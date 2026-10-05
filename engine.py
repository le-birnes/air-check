# -*- coding: utf-8 -*-
"""Audio capture + live EQ + monitor + record engine.

Pipeline: loopback capture device -> DeHummer -> 5-band EQ -> per-channel mute
-> (a) monitor output device (so you hear it live) and (b) WAV file while
recording is active. A 10 s rolling timeline (waveform min/max + spectrogram
columns) is kept for the scrolling monitors.
"""
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import soundcard as sc
import sounddevice as sd
import soundfile as sf

from eq import FiveBandEQ, DeHummer

SAMPLE_RATE = 48000
BLOCK_SIZE = 1024
WAVEFORM_HISTORY = 2048  # samples kept for the oscilloscope trace

SPECTRUM_BINS = 96  # default/base resolution; UI can rescale via set_spectrum_bin_count
SPECTRUM_FREQ_MIN = 30.0
SPECTRUM_FREQ_MAX = 18000.0
SPECTRUM_FREQS = np.logspace(np.log10(SPECTRUM_FREQ_MIN), np.log10(SPECTRUM_FREQ_MAX), SPECTRUM_BINS)
SPECTRUM_RANGE_DB = 50.0  # dynamic range window shown behind the auto-tracked ceiling

# ---- scrolling timeline monitors --------------------------------------------
TIMELINE_SECONDS = 10.0
WAVE_CHUNK = 256  # samples per waveform min/max column (~5.3 ms)
WAVE_HISTORY = int(TIMELINE_SECONDS * SAMPLE_RATE / WAVE_CHUNK)
SPEC_FFT = 2048  # hop = one capture block
SPEC_HISTORY = int(np.ceil(TIMELINE_SECONDS * SAMPLE_RATE / BLOCK_SIZE))
SPEC_ROWS = 160
SPEC_FREQ_LO, SPEC_FREQ_HI = 40.0, 22000.0
SPEC_SCALE_OFFSET = 300.0  # log10(f + offset): Audition-like spectral ruler (1k sits ~1/3 up)
SPEC_SILENCE_DB = -200.0
MONO_HOLD_BLOCKS = 47  # ~1 s of identical L/R (non-silent) before showing a single lane


def spec_scale(freq):
    return np.log10(np.asarray(freq, dtype=np.float64) + SPEC_SCALE_OFFSET)


def spec_unscale(value):
    return 10 ** np.asarray(value, dtype=np.float64) - SPEC_SCALE_OFFSET


SPEC_ROW_EDGES = spec_unscale(np.linspace(spec_scale(SPEC_FREQ_LO), spec_scale(SPEC_FREQ_HI), SPEC_ROWS + 1))
SPEC_ROW_BINS = np.clip(np.round(SPEC_ROW_EDGES[:-1] * SPEC_FFT / SAMPLE_RATE).astype(int), 0, SPEC_FFT // 2)
SPEC_WINDOW = np.hanning(SPEC_FFT)[:, None]


def list_capture_devices():
    """Loopback-capable sources: any render device, tapped post-mix."""
    return [m.name for m in sc.all_microphones(include_loopback=True)]


def list_monitor_devices():
    """Real output devices you can actually listen on."""
    devices = sd.query_devices()
    names = []
    seen = set()
    for d in devices:
        if d["max_output_channels"] > 0 and d["name"] not in seen:
            names.append(d["name"])
            seen.add(d["name"])
    return names


class AudioEngine:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.eq = FiveBandEQ(SAMPLE_RATE, channels=2)
        self.dehummer = DeHummer(SAMPLE_RATE, channels=2)
        self.mute = [False, False]  # per-channel, applied to monitor + recording

        self._capture_name = None
        self._monitor_name = None
        self._thread = None
        self._running = threading.Event()
        self._recording = threading.Event()
        self._paused = threading.Event()

        self._wav_file = None
        self._wav_lock = threading.Lock()
        self.current_recording_path = None

        self._level_db_l = -60.0
        self._level_db_r = -60.0
        self._waveform = np.zeros(WAVEFORM_HISTORY, dtype=np.float32)
        self.spectrum_freqs = SPECTRUM_FREQS  # mutable: UI rescales via set_spectrum_bin_count
        self._spectrum_l = np.zeros(len(self.spectrum_freqs), dtype=np.float32)
        self._spectrum_r = np.zeros(len(self.spectrum_freqs), dtype=np.float32)
        self._spec_ceiling = -20.0

        self._wave_min = np.zeros((WAVE_HISTORY, 2), dtype=np.float32)
        self._wave_max = np.zeros((WAVE_HISTORY, 2), dtype=np.float32)
        self._wave_pos = 0
        self._spec_hist = np.full((SPEC_HISTORY, 2, SPEC_ROWS), SPEC_SILENCE_DB, dtype=np.float32)
        self._spec_pos = 0
        self._fft_buf = np.zeros((0, 2), dtype=np.float32)
        self._stereo_hold = MONO_HOLD_BLOCKS
        self._is_mono = False
        self._state_lock = threading.Lock()

        self.master_volume = 1.0  # monitor-only gain; recorded file is unaffected

        self._monitor_stream = None
        self._last_error = None

    # ---- device selection -------------------------------------------------
    def set_capture_device(self, name):
        self._capture_name = name

    def set_monitor_device(self, name):
        self._monitor_name = name

    def set_master_volume(self, gain):
        self.master_volume = float(np.clip(gain, 0.0, 1.5))

    def set_channel_mute(self, channel, muted):
        self.mute[channel] = bool(muted)

    def set_spectrum_bin_count(self, n):
        """Rescale the spectrum resolution (called by the UI as the window is resized)."""
        with self._state_lock:
            if n == len(self.spectrum_freqs):
                return
            self.spectrum_freqs = np.logspace(
                np.log10(SPECTRUM_FREQ_MIN), np.log10(SPECTRUM_FREQ_MAX), n
            )
            self._spectrum_l = np.zeros(n, dtype=np.float32)
            self._spectrum_r = np.zeros(n, dtype=np.float32)

    # ---- lifecycle ----------------------------------------------------------
    def start(self):
        if self._running.is_set():
            return
        if not self._capture_name:
            raise RuntimeError("No capture device selected")
        self._running.set()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._close_wav()

    # ---- recording controls -------------------------------------------------
    def start_recording(self):
        with self._wav_lock:
            if self._wav_file is not None:
                return self.current_recording_path
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = self.output_dir / f"recording_{ts}.wav"
            self._wav_file = sf.SoundFile(
                str(path), mode="w", samplerate=SAMPLE_RATE,
                channels=2, subtype="FLOAT"
            )
            self.current_recording_path = path
        self._paused.clear()
        self._recording.set()
        return path

    def pause_recording(self):
        self._paused.set()

    def resume_recording(self):
        self._paused.clear()

    def stop_recording(self):
        self._recording.clear()
        self._paused.clear()
        self._close_wav()
        path = self.current_recording_path
        self.current_recording_path = None
        return path

    def _close_wav(self):
        with self._wav_lock:
            if self._wav_file is not None:
                self._wav_file.close()
                self._wav_file = None

    @property
    def is_recording(self):
        return self._recording.is_set() and not self._paused.is_set()

    @property
    def is_paused(self):
        return self._recording.is_set() and self._paused.is_set()

    # ---- meters for the UI ---------------------------------------------------
    def snapshot(self):
        with self._state_lock:
            return (self._level_db_l, self._level_db_r, self._waveform.copy(),
                    self._spectrum_l.copy(), self._spectrum_r.copy())

    def timeline_snapshot(self):
        """Oldest -> newest: waveform min/max (WAVE_HISTORY, 2), spectrogram dB
        (SPEC_HISTORY, 2, SPEC_ROWS), and whether the input currently looks mono.
        Pre-mute data, so a channel's history is there the moment it's unmuted."""
        with self._state_lock:
            wp = self._wave_pos % WAVE_HISTORY
            sp = self._spec_pos % SPEC_HISTORY
            return (np.roll(self._wave_min, -wp, axis=0), np.roll(self._wave_max, -wp, axis=0),
                    np.roll(self._spec_hist, -sp, axis=0), self._is_mono)

    @property
    def last_error(self):
        return self._last_error

    # ---- worker loop ------------------------------------------------------
    def _run_loop(self):
        try:
            mic = sc.get_microphone(id=self._capture_name, include_loopback=True)
        except Exception as e:
            self._last_error = f"Capture device error: {e}"
            self._running.clear()
            return

        out_stream = None
        if self._monitor_name:
            try:
                dev_index = self._resolve_output_index(self._monitor_name)
                print(f"[engine] monitor device '{self._monitor_name}' resolved to index {dev_index}", flush=True)
                out_stream = sd.OutputStream(
                    samplerate=SAMPLE_RATE, channels=2, device=dev_index,
                    blocksize=BLOCK_SIZE, dtype="float32", latency="low",
                )
                out_stream.start()
                print("[engine] monitor stream started OK", flush=True)
            except Exception as e:
                self._last_error = f"Monitor device error: {e}"
                print(f"[engine] MONITOR START FAILED: {e!r}", flush=True)
                out_stream = None
        else:
            print("[engine] no monitor device set", flush=True)

        try:
            with mic.recorder(samplerate=SAMPLE_RATE, channels=2, blocksize=BLOCK_SIZE) as rec:
                while self._running.is_set():
                    block = rec.record(numframes=BLOCK_SIZE).astype(np.float32)
                    x = self.dehummer.process(block.astype(np.float64))
                    x = self.eq.process(x)
                    processed = np.ascontiguousarray(np.clip(x, -1.0, 1.0).astype(np.float32))
                    self._update_timeline(processed)
                    if self.mute[0] or self.mute[1]:
                        processed = processed.copy()
                        for ch in (0, 1):
                            if self.mute[ch]:
                                processed[:, ch] = 0.0

                    if out_stream is not None:
                        try:
                            monitor_block = np.clip(processed * self.master_volume, -1.0, 1.0).astype(np.float32)
                            out_stream.write(np.ascontiguousarray(monitor_block))
                        except Exception as e:
                            if self._last_error != f"Monitor write error: {e}":
                                print(f"[engine] MONITOR WRITE FAILED: {e!r}", flush=True)
                            self._last_error = f"Monitor write error: {e}"

                    if self._recording.is_set() and not self._paused.is_set():
                        with self._wav_lock:
                            if self._wav_file is not None:
                                self._wav_file.write(processed)

                    self._update_meters(processed)
        except Exception as e:
            self._last_error = f"Capture loop error: {e}"
        finally:
            if out_stream is not None:
                out_stream.stop()
                out_stream.close()
            self._running.clear()

    def _update_timeline(self, block):
        n = len(block)
        nchunks = n // WAVE_CHUNK
        seg = block[:nchunks * WAVE_CHUNK].reshape(nchunks, WAVE_CHUNK, 2)
        mins = seg.min(axis=1)
        maxs = seg.max(axis=1)

        self._fft_buf = np.concatenate([self._fft_buf, block])[-SPEC_FFT:]
        spec = None
        if len(self._fft_buf) == SPEC_FFT:
            # scaled so a full-scale sine reads ~0 dBFS
            mags = np.abs(np.fft.rfft(self._fft_buf * SPEC_WINDOW, axis=0)) / (SPEC_FFT / 4)
            rows = np.maximum.reduceat(mags, SPEC_ROW_BINS, axis=0)  # (ROWS, 2)
            spec = (20 * np.log10(rows + 1e-10)).T.astype(np.float32)

        peak = float(np.abs(block).max()) if n else 0.0
        if peak > 1e-5:  # silence says nothing about mono vs stereo
            diff = float(np.abs(block[:, 0] - block[:, 1]).max())
            if diff > 1e-4 * max(peak, 1e-3):
                self._stereo_hold = MONO_HOLD_BLOCKS
            elif self._stereo_hold > 0:
                self._stereo_hold -= 1

        with self._state_lock:
            if nchunks:
                idx = (self._wave_pos + np.arange(nchunks)) % WAVE_HISTORY
                self._wave_min[idx] = mins
                self._wave_max[idx] = maxs
                self._wave_pos += nchunks
            if spec is not None:
                self._spec_hist[self._spec_pos % SPEC_HISTORY] = spec
                self._spec_pos += 1
            self._is_mono = self._stereo_hold == 0

    def _resolve_output_index(self, name):
        for idx, d in enumerate(sd.query_devices()):
            if d["name"] == name and d["max_output_channels"] > 0:
                return idx
        return None

    @staticmethod
    def _channel_level_db(samples):
        rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)) + 1e-12)
        level_db = 20 * np.log10(rms) if rms > 1e-9 else -60.0
        return max(-60.0, min(0.0, level_db))

    def _update_meters(self, block):
        mono = block.mean(axis=1)
        level_db_l = self._channel_level_db(block[:, 0])
        level_db_r = self._channel_level_db(block[:, 1])

        n = len(mono)
        spectrum_freqs = self.spectrum_freqs
        # capture local references once -- self._spectrum_l/r can be reassigned
        # to a differently-sized array at any moment by set_spectrum_bin_count()
        # running on the UI thread, so re-reading them mid-function would race
        prev_spec_l = self._spectrum_l
        prev_spec_r = self._spectrum_r
        spec_l = prev_spec_l
        spec_r = prev_spec_r
        if n >= 64 and len(prev_spec_l) == len(spectrum_freqs):
            window = np.hanning(n)
            fft_freqs = np.fft.rfftfreq(n, d=1.0 / SAMPLE_RATE)
            mags_l = np.abs(np.fft.rfft(block[:, 0] * window))
            mags_r = np.abs(np.fft.rfft(block[:, 1] * window))
            sampled_l = np.interp(spectrum_freqs, fft_freqs, mags_l)
            sampled_r = np.interp(spectrum_freqs, fft_freqs, mags_r)
            db_l = 20 * np.log10(sampled_l + 1e-6)
            db_r = 20 * np.log10(sampled_r + 1e-6)

            # auto-tracking ceiling shared by both channels, so their relative
            # strength stays meaningful (independent per-channel normalization
            # would hide which side is actually louder at a given frequency).
            peak_now = float(max(db_l.max(), db_r.max()))
            self._spec_ceiling = max(peak_now, self._spec_ceiling - 0.15)
            floor = self._spec_ceiling - SPECTRUM_RANGE_DB

            new_l = np.clip((db_l - floor) / SPECTRUM_RANGE_DB, 0.0, 1.0).astype(np.float32)
            new_r = np.clip((db_r - floor) / SPECTRUM_RANGE_DB, 0.0, 1.0).astype(np.float32)
            spec_l = 0.55 * prev_spec_l + 0.45 * new_l  # temporal smoothing
            spec_r = 0.55 * prev_spec_r + 0.45 * new_r

        take = min(n, WAVEFORM_HISTORY)
        with self._state_lock:
            self._level_db_l = level_db_l
            self._level_db_r = level_db_r
            self._waveform = np.roll(self._waveform, -take)
            self._waveform[-take:] = mono[-take:]
            self._spectrum_l = spec_l
            self._spectrum_r = spec_r
