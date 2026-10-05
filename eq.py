# -*- coding: utf-8 -*-
"""Real-time EQ + DeHummer, both built from RBJ-cookbook biquads.

EQ: 5 fixed bands -- high-pass, low-mid / mid / high-mid peaking bells and a
low-pass -- each with its own adjustable Q (narrowness). HP parked at the very
bottom of the display range and LP at the very top are bypassed ("off").

DeHummer (Adobe Audition style): a stack of narrow peaking cuts at a base
frequency and its harmonics. Base frequency, Q, gain, number of harmonics,
harmonic slope (how much shallower each higher harmonic's cut gets) and an
'output hum only' mode that returns exactly what the notches take out."""
import numpy as np
from scipy.signal import lfilter

FREQ_MIN, FREQ_MAX = 30.0, 18000.0
GAIN_MIN_DB, GAIN_MAX_DB = -12.0, 12.0
PEAK_Q_MIN, PEAK_Q_MAX = 0.1, 24.0
PASS_Q_MIN, PASS_Q_MAX = 0.4, 6.0

BAND_DEFS = [  # (name, kind, default freq, default q)
    ("HP", "highpass", FREQ_MIN, 0.707),
    ("LOW", "peak", 150.0, 1.0),
    ("MID", "peak", 1000.0, 1.0),
    ("HIGH", "peak", 5000.0, 1.0),
    ("LP", "lowpass", FREQ_MAX, 0.707),
]

HUM_FREQ_MIN, HUM_FREQ_MAX = 20.0, 4000.0
HUM_Q_MIN, HUM_Q_MAX = 1.0, 5000.0
HUM_GAIN_MIN, HUM_GAIN_MAX = -80.0, 0.0
HUM_MAX_HARMONICS = 10

# Built-in DeHummer presets (mirrors Audition's list)
DEHUMMER_PRESETS = {
    "Remove 60 Hz and Harmonics": dict(freq=60.0, q=60.0, gain_db=-60.0, harmonics=8, slope=50.0, hum_only=False),
    "Remove 50 Hz and Harmonics": dict(freq=50.0, q=60.0, gain_db=-60.0, harmonics=8, slope=50.0, hum_only=False),
    "Remove 400 Hz and Harmonics": dict(freq=400.0, q=60.0, gain_db=-60.0, harmonics=8, slope=50.0, hum_only=False),
    "60 Hz Notch": dict(freq=60.0, q=60.0, gain_db=-60.0, harmonics=1, slope=50.0, hum_only=False),
    "50 Hz Notch": dict(freq=50.0, q=60.0, gain_db=-60.0, harmonics=1, slope=50.0, hum_only=False),
    "120 Hz Notch": dict(freq=120.0, q=60.0, gain_db=-60.0, harmonics=1, slope=50.0, hum_only=False),
    "200 Hz Narrow Notch": dict(freq=200.0, q=3000.0, gain_db=-60.0, harmonics=1, slope=50.0, hum_only=False),
    "1000 Hz Resonance": dict(freq=1000.0, q=30.0, gain_db=-60.0, harmonics=1, slope=50.0, hum_only=True),
}


def peaking_coeffs(freq_hz, gain_db, q, sample_rate):
    a_amp = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * freq_hz / sample_rate
    alpha = np.sin(w0) / (2 * q)
    cos_w0 = np.cos(w0)
    b = np.array([1 + alpha * a_amp, -2 * cos_w0, 1 - alpha * a_amp])
    a = np.array([1 + alpha / a_amp, -2 * cos_w0, 1 - alpha / a_amp])
    return b / a[0], a / a[0]


def pass_coeffs(kind, freq_hz, q, sample_rate):
    w0 = 2 * np.pi * freq_hz / sample_rate
    alpha = np.sin(w0) / (2 * q)
    cos_w0 = np.cos(w0)
    if kind == "highpass":
        b = np.array([(1 + cos_w0) / 2, -(1 + cos_w0), (1 + cos_w0) / 2])
    else:
        b = np.array([(1 - cos_w0) / 2, 1 - cos_w0, (1 - cos_w0) / 2])
    a = np.array([1 + alpha, -2 * cos_w0, 1 - alpha])
    return b / a[0], a / a[0]


def biquad_response_db(b, a, freqs_hz, sample_rate):
    z1 = np.exp(-2j * np.pi * np.asarray(freqs_hz, dtype=np.float64) / sample_rate)
    z2 = z1 * z1
    h = (b[0] + b[1] * z1 + b[2] * z2) / (a[0] + a[1] * z1 + a[2] * z2)
    return 20 * np.log10(np.maximum(np.abs(h), 1e-9))


class Biquad:
    """One stereo biquad section with its own filter state."""

    def __init__(self, channels):
        self.channels = channels
        self.coeffs = (np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))
        self.bypass = True
        self.zi = np.zeros((2, channels))

    def set(self, b, a, bypass=False):
        if self.bypass and not bypass:
            self.zi = np.zeros((2, self.channels))  # stale state would click on re-entry
        self.coeffs = (b, a)
        self.bypass = bypass

    def reset(self):
        self.zi = np.zeros((2, self.channels))

    def process(self, x):
        if self.bypass:
            return x
        b, a = self.coeffs
        y, self.zi = lfilter(b, a, x, axis=0, zi=self.zi)
        return y


class EQBand:
    def __init__(self, name, kind, freq, q, sample_rate, channels):
        self.name = name
        self.kind = kind
        self.sample_rate = sample_rate
        self.freq = float(freq)
        self.q = float(q)
        self.gain_db = 0.0
        self.biquad = Biquad(channels)
        self._update()

    def is_off(self):
        if self.kind == "highpass":
            return self.freq <= FREQ_MIN * 1.001
        if self.kind == "lowpass":
            return self.freq >= FREQ_MAX / 1.001
        return abs(self.gain_db) < 1e-3

    def set_params(self, freq=None, gain_db=None, q=None):
        if freq is not None:
            self.freq = float(np.clip(freq, FREQ_MIN, FREQ_MAX))
        if gain_db is not None and self.kind == "peak":
            self.gain_db = float(np.clip(gain_db, GAIN_MIN_DB, GAIN_MAX_DB))
        if q is not None:
            lo, hi = (PEAK_Q_MIN, PEAK_Q_MAX) if self.kind == "peak" else (PASS_Q_MIN, PASS_Q_MAX)
            self.q = float(np.clip(q, lo, hi))
        self._update()

    def _update(self):
        if self.kind == "peak":
            b, a = peaking_coeffs(self.freq, self.gain_db, self.q, self.sample_rate)
        else:
            b, a = pass_coeffs(self.kind, self.freq, self.q, self.sample_rate)
        self.biquad.set(b, a, bypass=self.is_off())


class FiveBandEQ:
    def __init__(self, sample_rate, channels=2):
        self.sample_rate = sample_rate
        self.bands = [EQBand(n, k, f, q, sample_rate, channels) for n, k, f, q in BAND_DEFS]

    def reset_flat(self):
        """Bells to 0 dB, HP/LP parked off. Bell frequencies and Qs are kept."""
        for band in self.bands:
            if band.kind == "highpass":
                band.set_params(freq=FREQ_MIN)
            elif band.kind == "lowpass":
                band.set_params(freq=FREQ_MAX)
            else:
                band.set_params(gain_db=0.0)

    def reset_band(self, index):
        name, kind, freq, q = BAND_DEFS[index]
        band = self.bands[index]
        if kind == "peak":
            band.set_params(gain_db=0.0)
        else:
            band.set_params(freq=freq)

    def get_state(self):
        return [dict(freq=b.freq, gain_db=b.gain_db, q=b.q) for b in self.bands]

    def set_state(self, state):
        for band, d in zip(self.bands, state or []):
            band.set_params(freq=d.get("freq"), gain_db=d.get("gain_db"), q=d.get("q"))

    def response_db(self, freqs_hz):
        total = np.zeros(len(freqs_hz))
        for band in self.bands:
            if not band.biquad.bypass:
                total += biquad_response_db(*band.biquad.coeffs, freqs_hz, self.sample_rate)
        return total

    def process(self, x):
        for band in self.bands:
            x = band.biquad.process(x)
        return x


class DeHummer:
    def __init__(self, sample_rate, channels=2):
        self.sample_rate = sample_rate
        self.enabled = False
        self.freq = 60.0
        self.q = 60.0
        self.gain_db = -60.0
        self.harmonics = 8
        self.slope = 50.0  # percent
        self.hum_only = False
        self._filters = [Biquad(channels) for _ in range(HUM_MAX_HARMONICS)]
        self._rebuild()

    def notch_specs(self):
        """(freq, gain_db) for the base notch and each harmonic. Harmonic k's cut
        shrinks linearly toward the top one by `slope` percent."""
        specs = []
        n = self.harmonics
        for k in range(1, n + 1):
            f = self.freq * k
            if f >= 0.45 * self.sample_rate:
                break
            specs.append((f, self.gain_db * (1.0 - (self.slope / 100.0) * (k - 1) / n)))
        return specs

    def _rebuild(self):
        specs = self.notch_specs()
        for i, filt in enumerate(self._filters):
            if i < len(specs):
                f, g = specs[i]
                filt.set(*peaking_coeffs(f, g, self.q, self.sample_rate))
            else:
                filt.set(np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), bypass=True)

    def set_params(self, freq=None, q=None, gain_db=None, harmonics=None, slope=None, hum_only=None):
        if freq is not None:
            self.freq = float(np.clip(freq, HUM_FREQ_MIN, HUM_FREQ_MAX))
        if q is not None:
            self.q = float(np.clip(q, HUM_Q_MIN, HUM_Q_MAX))
        if gain_db is not None:
            self.gain_db = float(np.clip(gain_db, HUM_GAIN_MIN, HUM_GAIN_MAX))
        if harmonics is not None:
            self.harmonics = int(np.clip(harmonics, 1, HUM_MAX_HARMONICS))
        if slope is not None:
            self.slope = float(np.clip(slope, 0.0, 100.0))
        if hum_only is not None:
            self.hum_only = bool(hum_only)
        self._rebuild()

    def set_enabled(self, enabled):
        enabled = bool(enabled)
        if enabled and not self.enabled:
            for filt in self._filters:
                filt.reset()
        self.enabled = enabled

    def get_state(self):
        return dict(enabled=self.enabled, freq=self.freq, q=self.q, gain_db=self.gain_db,
                    harmonics=self.harmonics, slope=self.slope, hum_only=self.hum_only)

    def set_state(self, state):
        if not state:
            return
        self.set_params(**{k: state.get(k) for k in
                           ("freq", "q", "gain_db", "harmonics", "slope", "hum_only")})
        if "enabled" in state:
            self.set_enabled(state["enabled"])

    def response_db(self, freqs_hz):
        total = np.zeros(len(freqs_hz))
        for f, g in self.notch_specs():
            total += biquad_response_db(*peaking_coeffs(f, g, self.q, self.sample_rate),
                                        freqs_hz, self.sample_rate)
        return total

    def process(self, x):
        if not self.enabled:
            return x
        y = x
        for filt in self._filters:
            y = filt.process(y)
        return x - y if self.hum_only else y
