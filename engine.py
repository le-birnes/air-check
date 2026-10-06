# -*- coding: utf-8 -*-
"""Audio capture + live EQ + monitor + record engine.

Pipeline: loopback capture device -> DeHummer -> 5-band EQ -> per-channel mute
-> (a) monitor output device (so you hear it live) and (b) WAV file while
recording is active. A 10 s history of waveform + spectrum frames (L, R and
mix) feeds the 3D scope.
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

# ---- 3D scope history ---------------------------------------------------------
TIMELINE_SECONDS = 10.0
SCOPE_POINTS = 128  # envelope points per history frame (must divide SCOPE_WINDOW)
SCOPE_FRAME_BLOCKS = 4  # one history frame every 4 blocks (~85 ms): dense enough to read as terrain
SCOPE_WINDOW = SCOPE_FRAME_BLOCKS * BLOCK_SIZE  # samples one frame / the live trace covers
SCOPE_FRAMES = int(np.ceil(TIMELINE_SECONDS * SAMPLE_RATE / BLOCK_SIZE / SCOPE_FRAME_BLOCKS))
SPEC_FFT = 2048  # analysis window = the last two capture blocks
SPEC_ROWS = 128
SPEC_FREQ_LO, SPEC_FREQ_HI = 40.0, 20000.0
SPEC_SCALE_OFFSET = 0.0  # plain log frequency axis
SPEC_SILENCE_DB = -200.0
MONO_HOLD_BLOCKS = 47  # ~1 s of identical L/R (non-silent) before showing a single view


def spec_scale(freq):
    return np.log10(np.asarray(freq, dtype=np.float64) + SPEC_SCALE_OFFSET)


def spec_unscale(value):
    return 10 ** np.asarray(value, dtype=np.float64) - SPEC_SCALE_OFFSET


SPEC_ROW_EDGES = spec_unscale(np.linspace(spec_scale(SPEC_FREQ_LO), spec_scale(SPEC_FREQ_HI), SPEC_ROWS + 1))
SPEC_ROW_BINS = np.clip(np.round(SPEC_ROW_EDGES[:-1] * SPEC_FFT / SAMPLE_RATE).astype(int), 0, SPEC_FFT // 2)
SPEC_WINDOW = np.hanning(SPEC_FFT)[:, None]


def peak_decimate(x, points):
    """(n, c) -> (points, c), keeping the largest-magnitude sample of each
    segment (with its sign) so peaks survive the downsampling."""
    k = len(x) // points
    seg = x[len(x) - points * k:].reshape(points, k, x.shape[1])
    idx = np.abs(seg).argmax(axis=1)[:, None, :]
    return np.take_along_axis(seg, idx, axis=1)[:, 0, :]


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

        self._fft_buf = np.zeros((0, 2), dtype=np.float32)
        self._live = np.zeros((SCOPE_WINDOW, 2), dtype=np.float32)
        self._block_count = 0
        self._scope_wave = np.zeros((SCOPE_FRAMES, 3, 2, SCOPE_POINTS), dtype=np.float32)
        self._scope_spec = np.full((SCOPE_FRAMES, 3, SPEC_ROWS), SPEC_SILENCE_DB, dtype=np.float32)
        self._scope_block = np.full(SCOPE_FRAMES, -10 ** 9, dtype=np.int64)
        self._scope_pos = 0
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

    def scope_snapshot(self):
        """Live trace (last SCOPE_WINDOW samples, (n, 2)), then history frames
        oldest -> newest: waveform envelopes (F, 3, [max, min], SCOPE_POINTS) and spectra in dB (F, 3,
        SPEC_ROWS) for [L, R, mix], each frame's age in seconds, and whether
        the input looks mono. Pre-mute, so a channel shows the moment it's unmuted."""
        with self._state_lock:
            order = (np.arange(SCOPE_FRAMES) + self._scope_pos) % SCOPE_FRAMES
            ages = (self._block_count - self._scope_block[order]) * BLOCK_SIZE / SAMPLE_RATE
            return (self._live.copy(), self._scope_wave[order], self._scope_spec[order],
                    ages, self._is_mono)

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
        buf = np.concatenate([self._fft_buf, block])[-SPEC_FFT:]
        self._fft_buf = buf
        live = np.concatenate([self._live, block])[-SCOPE_WINDOW:]
        self._block_count += 1

        n = len(block)
        peak = float(np.abs(block).max()) if n else 0.0
        if peak > 1e-5:  # silence says nothing about mono vs stereo
            diff = float(np.abs(block[:, 0] - block[:, 1]).max())
            if diff > 1e-4 * max(peak, 1e-3):
                self._stereo_hold = MONO_HOLD_BLOCKS
            elif self._stereo_hold > 0:
                self._stereo_hold -= 1

        frame = None
        if len(buf) == SPEC_FFT and self._block_count % SCOPE_FRAME_BLOCKS == 0:
            three = np.column_stack([buf, buf.mean(axis=1)])  # L, R, mix
            # scaled so a full-scale sine reads ~0 dBFS
            mags = np.abs(np.fft.rfft(three * SPEC_WINDOW, axis=0)) / (SPEC_FFT / 4)
            rows = np.maximum.reduceat(mags, SPEC_ROW_BINS, axis=0)  # (ROWS, 3)
            win = np.column_stack([live, live.mean(axis=1)])
            seg = win.reshape(SCOPE_POINTS, -1, 3)
            envelope = np.stack([seg.max(axis=1).T, seg.min(axis=1).T], axis=1)  # (3, 2, P)
            frame = (envelope, (20 * np.log10(rows + 1e-10)).T)

        with self._state_lock:
            self._live = live
            if frame is not None:
                i = self._scope_pos % SCOPE_FRAMES
                self._scope_wave[i], self._scope_spec[i] = frame
                self._scope_block[i] = self._block_count
                self._scope_pos += 1
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
