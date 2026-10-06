# -*- coding: utf-8 -*-
"""Floating retro-80s 'air-check' recorder: per-device audio tap, 5-band EQ curve,
DeHummer, and a 3D waveform / spectral scope."""
import os
import struct
import sys
from pathlib import Path

import numpy as np

from PySide6.QtCore import Qt, QTimer, QPointF, QRectF, QDataStream, QByteArray
from PySide6.QtGui import (
    QPainter, QColor, QLinearGradient, QPen, QFont, QBrush, QPainterPath, QRadialGradient,
    QIcon, QPixmap, QAction, QImage,
)
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QSystemTrayIcon, QMenu, QSlider, QGridLayout, QCheckBox, QDoubleSpinBox, QInputDialog,
)

sys.path.insert(0, str(Path(__file__).parent))
from engine import (
    AudioEngine, list_capture_devices, list_monitor_devices, peak_decimate, TIMELINE_SECONDS,
    SCOPE_FRAMES, SCOPE_POINTS, SCOPE_WINDOW, SPEC_ROWS, SPEC_FREQ_LO, SPEC_FREQ_HI,
)
from eq import (
    BAND_DEFS, FREQ_MIN, FREQ_MAX, DEHUMMER_PRESETS, HUM_FREQ_MIN, HUM_FREQ_MAX, HUM_Q_MIN,
    HUM_Q_MAX, HUM_GAIN_MIN, HUM_GAIN_MAX, HUM_MAX_HARMONICS,
)
import settings as settings_store

OUTPUT_DIR = Path(__file__).parent / "recordings"

# --- matrix-red palette ------------------------------------------------------
BG_TOP = QColor(10, 0, 0)
BG_BOTTOM = QColor(0, 0, 0)
PANEL_BORDER = QColor(255, 25, 25)
GRID_COLOR = QColor(255, 25, 25, 55)
GRID_COLOR_DIM = QColor(120, 20, 20, 60)
ZERO_LINE = QColor(255, 255, 255, 90)
TEXT_DIM = QColor(200, 70, 70, 180)
TEXT_BRIGHT = QColor(255, 190, 190)

NODE_COLORS = [
    QColor(255, 45, 149),   # pink
    QColor(255, 140, 26),   # orange
    QColor(196, 255, 26),   # yellow-green
    QColor(26, 240, 255),   # cyan
    QColor(176, 38, 255),   # purple
    QColor(45, 120, 255),   # electric blue
    QColor(255, 80, 80),    # coral red
    QColor(120, 255, 170),  # mint
    QColor(255, 210, 40),   # gold
    QColor(140, 140, 255),  # periwinkle
    QColor(255, 120, 220),  # orchid
    QColor(90, 220, 220),   # teal
]

DB_RANGE = 15.0  # display range +-15dB (filters clamp at +-12, gives a little headroom)

SPECTRUM_BIN_COUNT = 200  # fixed spectrum resolution
SIDE_PANEL_WIDTH = 250


def freq_to_x(freq, rect):
    lo, hi = np.log10(FREQ_MIN), np.log10(FREQ_MAX)
    frac = (np.log10(freq) - lo) / (hi - lo)
    return rect.left() + frac * rect.width()


def db_to_y(db, rect):
    frac = (db + DB_RANGE) / (2 * DB_RANGE)
    return rect.bottom() - frac * rect.height()


def x_to_freq(x, rect):
    lo, hi = np.log10(FREQ_MIN), np.log10(FREQ_MAX)
    frac = min(max((x - rect.left()) / rect.width(), 0.0), 1.0)
    return 10 ** (lo + frac * (hi - lo))


def fmt_freq(freq):
    return f"{freq/1000:.2f}k" if freq >= 1000 else f"{freq:.0f}"


HUM_NODE = "hum"
HUM_DB_MIN = -80.0  # DeHummer overlay uses its own Audition-style 0..-80 dB right-hand scale
HUM_CURVE_COLOR = QColor(225, 35, 35)
HUM_AXIS_COLOR = QColor(235, 90, 90, 200)
Q_WHEEL_STEP = 1.15  # Q multiplier per mouse-wheel notch


class ParametricEQ(QWidget):
    """Live spectrum (left +-15 dB scale) plus the DeHummer notch comb in red
    on its own 0..-80 dB right-hand scale. The 5-band EQ nodes live on the 3D
    scope's YZ plane. Drag the hum node = move it, mouse wheel = Q."""

    HIT_RADIUS = 14

    def __init__(self, engine, on_change=None, on_hum_change=None):
        super().__init__()
        self.engine = engine
        self.on_change = on_change
        self.on_hum_change = on_hum_change
        self.setMinimumHeight(160)

        self.spectrum_freqs = np.logspace(np.log10(FREQ_MIN), np.log10(FREQ_MAX), SPECTRUM_BIN_COUNT)
        engine.set_spectrum_bin_count(SPECTRUM_BIN_COUNT)
        self.spectrum_l = np.zeros(SPECTRUM_BIN_COUNT)
        self.spectrum_r = np.zeros(SPECTRUM_BIN_COUNT)
        self.level_db = -60.0
        self._dragging = False  # dragging the hum node
        self.hum_freqs = np.logspace(np.log10(FREQ_MIN), np.log10(FREQ_MAX), 900)
        self.setMouseTracking(True)

    def update_data(self, level_db, spectrum_l, spectrum_r):
        self.level_db = level_db
        self.spectrum_l = spectrum_l
        self.spectrum_r = spectrum_r
        self.update()

    def plot_rect(self):
        return QRectF(self.rect()).adjusted(46, 22, -44, -30)

    # ---- painting -----------------------------------------------------------
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        rect = self.plot_rect()

        p.fillRect(self.rect(), QColor(3, 2, 5))
        p.setPen(QPen(QColor(80, 40, 110), 1))
        p.drawRect(rect)

        self._draw_grid(p, rect)
        self._draw_spectrum(p, rect)
        self._draw_zero_line(p, rect)
        p.save()
        p.setClipRect(rect.adjusted(-1, -1, 1, 1))
        self._draw_hum(p, rect)
        p.restore()
        self._draw_hum_axis(p, rect)
        p.end()

    def _draw_grid(self, p, rect):
        p.setFont(QFont("Consolas", 8))
        for f in (50, 100, 200, 500, 1000, 2000, 5000, 10000, 18000):
            x = freq_to_x(f, rect)
            p.setPen(QPen(GRID_COLOR_DIM, 1))
            p.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
            p.setPen(TEXT_DIM)
            label = f"{f // 1000}k" if f >= 1000 else str(f)
            p.drawText(QPointF(x - 10, rect.bottom() + 16), label)

        for db in (-12, -6, 0, 6, 12):
            y = db_to_y(db, rect)
            p.setPen(QPen(GRID_COLOR_DIM, 1))
            p.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
            p.setPen(TEXT_DIM)
            p.drawText(QPointF(rect.left() - 38, y + 4), f"{db:+d}")

    def _draw_spectrum(self, p, rect):
        n = len(self.spectrum_l)
        if n == 0 or n != len(self.spectrum_freqs):
            return
        w = max(int(round(rect.width())), 1)
        h = max(int(round(rect.height())), 1)

        # Composite the whole bar chart as one RGBA image with numpy, then
        # hand it to Qt as a single drawImage() call -- thousands of individual
        # fillRect() calls stall the audio thread out of the GIL (glitches).
        px = rect.left() + np.arange(w) + 0.5
        lo, hi = np.log10(FREQ_MIN), np.log10(FREQ_MAX)
        frac = np.clip((px - rect.left()) / rect.width(), 0.0, 1.0)
        freqs_per_col = 10 ** (lo + frac * (hi - lo))
        bin_idx = np.clip(np.searchsorted(self.spectrum_freqs, freqs_per_col), 0, n - 1)

        mag_l = self.spectrum_l[bin_idx]
        mag_r = self.spectrum_r[bin_idx]
        height_l = mag_l * h * 0.92
        height_r = mag_r * h * 0.92

        from_bottom = (h - 1 - np.arange(h, dtype=np.float32))[:, None]  # (h,1), 0 at bottom
        lit_l = from_bottom < height_l[None, :]  # (h,w)
        lit_r = from_bottom < height_r[None, :]
        both = lit_l & lit_r

        # L drives red, R drives green -- additive light mixing, so the
        # overlap reads as yellow. 80% opacity for a single channel, 100% on
        # the overlap or where one side beats the other by > 50 points.
        diff = mag_l - mag_r
        l_boost = (diff > 0.5)[None, :]
        r_boost = (diff < -0.5)[None, :]
        alpha_l_only = np.where(l_boost, 255.0, 204.0)
        alpha_r_only = np.where(r_boost, 255.0, 204.0)

        img = np.zeros((h, w, 4), dtype=np.uint8)
        img[..., 0] = np.where(lit_l, 255, 0).astype(np.uint8)
        img[..., 1] = np.where(lit_r, 255, 0).astype(np.uint8)
        alpha = np.where(both, 255.0,
                          np.where(lit_l, alpha_l_only,
                                   np.where(lit_r, alpha_r_only, 0.0)))
        img[..., 3] = alpha.astype(np.uint8)

        img = np.ascontiguousarray(img)
        qimg = QImage(img.data, w, h, w * 4, QImage.Format_RGBA8888).copy()
        p.drawImage(QPointF(rect.left(), rect.top()), qimg)

    def _draw_zero_line(self, p, rect):
        y = db_to_y(0, rect)
        p.setPen(QPen(ZERO_LINE, 1, Qt.DotLine))
        p.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))

    @staticmethod
    def _path(freqs, ys, rect):
        path = QPainterPath()
        for i, (f, y) in enumerate(zip(freqs, ys)):
            x = freq_to_x(f, rect)
            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
        return path

    @staticmethod
    def _stroke(p, path, color):
        for width, alpha in ((7, 40), (4, 90), (1.6, 255)):
            pen = QPen(QColor(color.red(), color.green(), color.blue(), alpha), width)
            pen.setCapStyle(Qt.RoundCap)
            pen.setJoinStyle(Qt.RoundJoin)
            p.setPen(pen)
            p.drawPath(path)

    # DeHummer overlay: 0 dB at the top, HUM_DB_MIN at the bottom
    @staticmethod
    def _hum_y(db, rect):
        return rect.top() + (db / HUM_DB_MIN) * rect.height()

    @staticmethod
    def _hum_db(y, rect):
        return (y - rect.top()) / rect.height() * HUM_DB_MIN

    def _hum_node_pos(self, rect):
        hum = self.engine.dehummer
        freq = min(max(hum.freq, FREQ_MIN), FREQ_MAX)
        return freq_to_x(freq, rect), self._hum_y(hum.gain_db, rect)

    def _draw_hum(self, p, rect):
        hum = self.engine.dehummer
        if not hum.enabled:
            return
        # insert the exact notch centers so even a Q-3000 hairline reaches its depth
        centers = [f for f, _ in hum.notch_specs() if FREQ_MIN <= f <= FREQ_MAX]
        freqs = np.unique(np.concatenate([self.hum_freqs, centers]))
        db_vals = np.clip(hum.response_db(freqs), HUM_DB_MIN, 0.0)
        self._stroke(p, self._path(freqs, [self._hum_y(d, rect) for d in db_vals], rect),
                     HUM_CURVE_COLOR)

    def _draw_hum_axis(self, p, rect):
        hum = self.engine.dehummer
        if not hum.enabled:
            return
        p.setFont(QFont("Consolas", 7))
        p.setPen(HUM_AXIS_COLOR)
        p.drawText(QPointF(rect.right() + 6, rect.top() - 8), "dB")
        for db in range(0, int(HUM_DB_MIN) - 1, -10):
            y = self._hum_y(db, rect)
            p.drawLine(QPointF(rect.right(), y), QPointF(rect.right() + 3, y))
            p.drawText(QPointF(rect.right() + 6, y + 3), str(db))
        x, y = self._hum_node_pos(rect)
        p.setPen(QPen(QColor(20, 0, 0), 1))
        p.setBrush(QColor(255, 255, 255))
        p.drawRect(QRectF(x - 4, y - 4, 8, 8))
        p.setFont(QFont("Consolas", 8, QFont.Bold))
        p.setPen(QColor(255, 120, 120))
        p.drawText(QRectF(x - 60, y + 6, 120, 13), Qt.AlignCenter,
                   f"{fmt_freq(hum.freq)}Hz Q{hum.q:.0f} {hum.gain_db:.0f}dB")

    # ---- mouse interaction (hum node only) -------------------------------------
    def _hit_hum(self, pos, rect):
        if not self.engine.dehummer.enabled:
            return False
        x, y = self._hum_node_pos(rect)
        return (pos.x() - x) ** 2 + (pos.y() - y) ** 2 <= self.HIT_RADIUS ** 2

    def mousePressEvent(self, event):
        self._dragging = self._hit_hum(event.position(), self.plot_rect())

    def mouseMoveEvent(self, event):
        if not self._dragging:
            return
        rect = self.plot_rect()
        pos = event.position()
        self.engine.dehummer.set_params(freq=x_to_freq(pos.x(), rect), gain_db=self._hum_db(pos.y(), rect))
        if self.on_hum_change:
            self.on_hum_change()
        self.update()

    def mouseReleaseEvent(self, event):
        if self._dragging and self.on_change:
            self.on_change()
        self._dragging = False

    def wheelEvent(self, event):
        if not self._hit_hum(event.position(), self.plot_rect()):
            return
        hum = self.engine.dehummer
        hum.set_params(q=hum.q * Q_WHEEL_STEP ** (event.angleDelta().y() / 120.0))
        if self.on_hum_change:
            self.on_hum_change()
        self.update()
        if self.on_change:
            self.on_change()
        event.accept()


class VuStrip(QWidget):
    """Stereo pixelated LED meter: each cell splits into 4 pieces (2 per channel row),
    1px gap between the L/R sub-pieces within a row, 2px gap between the two channel rows."""

    SEGMENTS = 40
    ROW_GAP = 2
    COL_GAP = 1

    def __init__(self):
        super().__init__()
        self.setFixedHeight(30)
        self.level_l = -60.0
        self.level_r = -60.0

    def set_levels(self, level_l, level_r):
        self.level_l = level_l
        self.level_r = level_r
        self.update()

    @staticmethod
    def _segment_color(frac, lit):
        if frac < 0.6:
            color = QColor(80, 240, 160)
        elif frac < 0.85:
            color = QColor(255, 220, 60)
        else:
            color = QColor(255, 60, 90)
        if not lit:
            color = QColor(color.red() // 4, color.green() // 4, color.blue() // 4)
        return color

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)
        r = self.rect()
        segs = self.SEGMENTS
        seg_w = r.width() / segs
        cell_w = seg_w * 0.82  # leaves a gap between adjacent segment columns
        half_w = (cell_w - self.COL_GAP) / 2
        row_h = (r.height() - self.ROW_GAP) / 2

        for row, level in enumerate((self.level_l, self.level_r)):
            lit_count = int(((level + 60) / 60.0) * segs)
            y0 = 0 if row == 0 else row_h + self.ROW_GAP
            for i in range(segs):
                frac = i / segs
                color = self._segment_color(frac, i < lit_count)
                p.setPen(Qt.NoPen)
                p.setBrush(color)
                x0 = i * seg_w
                p.drawRect(int(x0), int(y0), int(half_w), int(row_h))
                p.drawRect(int(x0 + half_w + self.COL_GAP), int(y0), int(half_w), int(row_h))
        p.end()


class BandListPanel(QWidget):
    """One row per EQ band ('name gain freq Q'), colored like its node."""

    def __init__(self):
        super().__init__()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 2, 2, 2)
        outer.setSpacing(2)

        title = QLabel("EQ BANDS")
        title.setStyleSheet("color:#ff3030; font-size:9px; font-weight:bold; letter-spacing:1px;")
        outer.addWidget(title)

        self._rows = []
        for i in range(len(BAND_DEFS)):
            row = QLabel("")
            color = NODE_COLORS[i % len(NODE_COLORS)]
            row.setStyleSheet(
                f"color: rgb({color.red()},{color.green()},{color.blue()}); "
                "font-family:Consolas; font-size:10px;"
            )
            outer.addWidget(row)
            self._rows.append(row)

        hint = QLabel("drag=move  wheel=Q  dbl-click=reset")
        hint.setStyleSheet("color:#8a4545; font-family:Consolas; font-size:9px;")
        outer.addWidget(hint)

    def refresh(self, bands):
        for row, band in zip(self._rows, bands):
            row.setText(self._format(band))

    @staticmethod
    def _format(band):
        if band.kind == "peak":
            return f"{band.name:<4} {band.gain_db:+5.1f}dB {fmt_freq(band.freq):>6}Hz Q{band.q:.2f}"
        where = "off" if band.is_off() else f"{fmt_freq(band.freq)}Hz"
        return f"{band.name:<4} {'':7} {where:>8} Q{band.q:.2f}"


# ---- 3D scope ------------------------------------------------------------------
LANE_BTN_W = 20  # channel (mute) buttons column on the right
SCOPE_BG = QColor(2, 2, 9)
SCOPE_TEXT = QColor(170, 170, 215, 170)
SCOPE_AXIS = QColor(200, 200, 230, 80)
SCOPE_BTN = QColor(140, 160, 255)
SCOPE_FADE_SECONDS = 3.0  # history fades out over the last 3 s of the 10 s window
SCOPE_DEPTH = 1.5         # world units the 10 s history spans on each side of the live trace
SCOPE_Y = 1.6             # half-length of the Y axis (the live trace / frequency axis)
SCOPE_WAVE_Z = 0.55       # waveform full-scale height
SCOPE_SPEC_Z = 0.8        # spectrum full height
SCOPE_EQ_Z = 0.6          # EQ curve on the YZ plane: +-DB_RANGE maps to +-SCOPE_EQ_Z
SCOPE_DB_FLOOR, SCOPE_DB_CEIL = -90.0, -10.0
SCOPE_ALPHA_BUCKETS = 10
SCOPE_HZ_MARKS = (50, 100, 200, 500, 1000, 2000, 5000, 10000)
SCOPE_DEFAULT_VIEW = (16.0, 52.0, 1.0)  # yaw deg, pitch deg, zoom
SCOPE_PITCH_MAX = 90.0    # orbit can tilt the floor all the way to a top-down view
SCOPE_LACE_STEP = 4       # every 4th spectrum row also runs across time (terrain grid)
EQ_RGB = (255, 205, 60)
EQ_NODE_HALF = 7          # node square half-size, px
EQ_HIT_RADIUS = 12
SCOPE_CAMERA = 5.0
WAVE_RGB_NEW, WAVE_RGB_OLD = np.array([125, 175, 255]), np.array([150, 90, 255])
SPEC_RGB_NEW, SPEC_RGB_OLD = np.array([200, 140, 255]), np.array([80, 70, 230])
LIVE_RGB = (215, 230, 255)

_PATH_DTYPE = np.dtype([("c", ">i4"), ("x", ">f8"), ("y", ">f8")])


def polylines_path(xs, ys):
    """(k, m) screen coords -> one QPainterPath holding k polylines, built in
    bulk through QPainterPath's QDataStream format. Thousands of per-point
    lineTo() calls per frame would hog the GIL and glitch the audio thread."""
    k, m = xs.shape
    arr = np.empty((k, m), dtype=_PATH_DTYPE)
    arr["c"] = 1
    arr["c"][:, 0] = 0  # MoveTo starts each polyline
    arr["x"] = xs
    arr["y"] = ys
    buf = struct.pack(">i", k * m) + arr.tobytes() + struct.pack(">ii", 0, 0)
    path = QPainterPath()
    QDataStream(QByteArray(buf)) >> path
    return path


def _smoothstep(f):
    f = np.clip(f, 0.0, 1.0)
    return f * f * (3 - 2 * f)


class Scope3D(QWidget):
    """3D audio scope. Y axis: the live waveform. X+: the last 10 s of waveform
    frames receding behind it (topography). X-: the matching spectrum frames
    (spectral map, log Hz along Y) coming forward, laced into a terrain.
    Everything fades out smoothly at 10 s.

    MIX/CH toggle: one scene for the mix, or L | R side by side. Channel mute
    buttons on the right: a muted channel's scene disappears and the other
    takes the whole area; both muted = dark. Mono input = one scene.
    The 5-band EQ curve stands on the YZ plane (log Hz along Y, gain along Z):
    drag a node = move it, wheel over a node = Q, double-click a node = reset it.
    Elsewhere: drag = orbit, wheel = zoom, double-click = reset view."""

    def __init__(self, engine, mode="mix", on_mute_change=None, on_mode_change=None,
                 on_eq_change=None, on_eq_commit=None):
        super().__init__()
        self.engine = engine
        self.mode = mode if mode in ("mix", "channels") else "mix"
        self.on_mute_change = on_mute_change
        self.on_mode_change = on_mode_change
        self.on_eq_change = on_eq_change    # every drag/wheel step (band list refresh)
        self.on_eq_commit = on_eq_commit    # end of an edit (persist)
        self._cams = []                     # [(scene rect, camera)] from the last paint
        self._eq_drag = None                # (band index, camera)
        self._eq_freqs = np.logspace(np.log10(SPEC_FREQ_LO), np.log10(SPEC_FREQ_HI), 300)
        self.mono = False
        self.live = np.zeros((SCOPE_WINDOW, 2), dtype=np.float32)
        self.wave = np.zeros((SCOPE_FRAMES, 3, 2, SCOPE_POINTS), dtype=np.float32)
        self.spec = np.full((SCOPE_FRAMES, 3, SPEC_ROWS), -200.0, dtype=np.float32)
        self.ages = np.full(SCOPE_FRAMES, 1e9)
        self.yaw, self.pitch, self.zoom = SCOPE_DEFAULT_VIEW
        self._orbit = None
        self.setMinimumHeight(220)

    def update_data(self, live, wave, spec, ages, mono):
        self.live, self.wave, self.spec, self.ages, self.mono = live, wave, spec, ages, mono
        self.update()

    # ---- layout -----------------------------------------------------------------
    @staticmethod
    def _toggle_rects():
        return QRectF(6, 4, 34, 15), QRectF(40, 4, 34, 15)

    def _layout(self):
        """-> (area, [(label, data index, rect)], {button: rect})"""
        r = QRectF(self.rect()).adjusted(2, 2, -(LANE_BTN_W + 6), -2)
        mute = self.engine.mute
        bx = r.right() + 4

        def btn(y):
            return QRectF(bx, y, 16, 16)

        if self.mono:
            scenes = [] if all(mute) else [("MONO", 0, r)]
            return r, scenes, {"M": btn(r.top() + 22)}
        if self.mode == "mix":
            scenes = []
            if not all(mute):
                idx = 2 if not any(mute) else (1 if mute[0] else 0)
                scenes = [("MIX" if idx == 2 else "LR"[idx] + " ONLY", idx, r)]
            return r, scenes, {0: btn(r.top() + 22), 1: btn(r.top() + 40)}
        # channels: L | R side by side; a muted side's scene goes, the other takes the width
        visible = [c for c in (0, 1) if not mute[c]]
        scenes = []
        if visible:
            w = r.width() / len(visible)
            scenes = [("LR"[c], c, QRectF(r.left() + i * w, r.top(), w, r.height()))
                      for i, c in enumerate(visible)]
        return r, scenes, {0: btn(r.top() + 22), 1: btn(r.top() + 40)}

    # ---- projection -------------------------------------------------------------
    def _camera(self, rect):
        """Fit the world box into rect -> (scale, cx, cy) for _project."""
        corners = np.array([(x, y, z) for x in (-SCOPE_DEPTH - 0.1, SCOPE_DEPTH + 0.1)
                            for y in (-SCOPE_Y - 0.2, SCOPE_Y + 0.1)
                            for z in (-max(SCOPE_WAVE_Z, SCOPE_EQ_Z), max(SCOPE_SPEC_Z, SCOPE_EQ_Z))])
        sx, sy = self._project(corners[:, 0], corners[:, 1], corners[:, 2], 1.0, 0.0, 0.0)
        # the box's empty corners overstate its height: favour filling the width
        scale = min((rect.width() - 20) / np.ptp(sx), 1.35 * (rect.height() - 20) / np.ptp(sy)) * self.zoom
        cx = rect.center().x() - scale * (sx.max() + sx.min()) / 2
        cy = rect.center().y() + 6 - scale * (sy.max() + sy.min()) / 2
        return scale, cx, cy

    def _project(self, x, y, z, scale, cx, cy):
        yaw, pitch = np.radians(self.yaw), np.radians(self.pitch)
        depth = x * np.cos(yaw) - y * np.sin(yaw)   # + = away from the viewer
        right = x * np.sin(yaw) + y * np.cos(yaw)
        up = z * np.cos(pitch) + depth * np.sin(pitch)
        toward = -depth * np.cos(pitch) + z * np.sin(pitch)
        f = SCOPE_CAMERA / (SCOPE_CAMERA - toward)
        return cx + scale * right * f, cy - scale * up * f

    # ---- painting ---------------------------------------------------------------
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(3, 2, 5))
        area, scenes, buttons = self._layout()
        p.fillRect(area, QColor(0, 0, 0))
        self._cams = []
        for label, idx, rect in scenes:
            p.save()
            p.setClipRect(rect)
            self._paint_scene(p, idx, rect, label)
            p.restore()
            p.setPen(QPen(QColor(50, 45, 90), 1))
            p.setBrush(Qt.NoBrush)
            p.drawRect(rect)
        for ch, b in buttons.items():
            self._paint_button(p, ch, b)
        self._paint_toggle(p)
        p.end()

    def _paint_scene(self, p, idx, rect, label):
        p.fillRect(rect, SCOPE_BG)
        cam = self._camera(rect)
        self._cams.append((rect, cam))

        def proj(x, y, z):
            return self._project(np.asarray(x, float), np.asarray(y, float), np.asarray(z, float), *cam)

        self._paint_axes(p, proj)

        ages = self.ages
        valid = ages < TIMELINE_SECONDS
        fade = (_smoothstep((TIMELINE_SECONDS - ages) / SCOPE_FADE_SECONDS)
                * (1.0 - 0.45 * np.clip(ages / TIMELINE_SECONDS, 0, 1)))
        # X+: waveform topography -- each frame's max/min envelope as one
        # closed silhouette, receding behind the live trace
        if valid.any():
            ys = np.linspace(-SCOPE_Y, SCOPE_Y, SCOPE_POINTS)
            ys = np.concatenate([ys, ys[::-1]])
            env = np.clip(self.wave[valid, idx], -1, 1)  # (k, 2, P)
            zs = np.concatenate([env[:, 0], env[:, 1, ::-1]], axis=1) * SCOPE_WAVE_Z
            xs = (ages[valid] / TIMELINE_SECONDS * SCOPE_DEPTH + 0.02)[:, None] * np.ones_like(ys)
            sx, sy = proj(xs, np.broadcast_to(ys, xs.shape), zs)
            self._stroke_buckets(p, sx, sy, fade[valid], ages[valid], WAVE_RGB_NEW, WAVE_RGB_OLD)

        # live trace on the Y axis: the same window a history frame covers
        three = np.column_stack([self.live, self.live.mean(axis=1)])
        live = peak_decimate(three, SCOPE_POINTS * 4)[:, idx]
        ly = np.linspace(-SCOPE_Y, SCOPE_Y, len(live))
        sx, sy = proj(np.zeros_like(ly), ly, np.clip(live, -1, 1) * SCOPE_WAVE_Z)
        path = polylines_path(sx[None, :], sy[None, :])
        p.setCompositionMode(QPainter.CompositionMode_Plus)
        for width, alpha in ((4.0, 45), (1.2, 230)):
            p.setPen(QPen(QColor(*LIVE_RGB, alpha), width))
            p.drawPath(path)
        p.setCompositionMode(QPainter.CompositionMode_SourceOver)

        # X-: spectral map coming forward, laced into a terrain
        if valid.any():
            ry = np.linspace(-SCOPE_Y, SCOPE_Y, SPEC_ROWS)
            level = np.clip((self.spec[valid, idx, :] - SCOPE_DB_FLOOR) / (SCOPE_DB_CEIL - SCOPE_DB_FLOOR), 0, 1) ** 1.6
            xs = -(ages[valid] / TIMELINE_SECONDS * SCOPE_DEPTH + 0.04)[:, None] * np.ones_like(ry)
            sx, sy = proj(xs, np.broadcast_to(ry, xs.shape), level * SCOPE_SPEC_Z)
            self._stroke_buckets(p, sx, sy, fade[valid], ages[valid], SPEC_RGB_NEW, SPEC_RGB_OLD)
            if valid.sum() > 1:
                lace = slice(SCOPE_LACE_STEP // 2, SPEC_ROWS, SCOPE_LACE_STEP)  # constant-frequency lines across time
                p.setPen(QPen(QColor(110, 100, 240, 45), 1.0))
                p.drawPath(polylines_path(np.ascontiguousarray(sx[:, lace].T),
                                          np.ascontiguousarray(sy[:, lace].T)))

        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
        self._paint_eq(p, proj)
        p.setFont(QFont("Consolas", 8, QFont.Bold))
        p.setPen(SCOPE_TEXT)
        p.drawText(QRectF(rect.right() - 90, rect.top() + 4, 84, 14), Qt.AlignRight, label)

    # ---- EQ on the YZ plane -----------------------------------------------------
    @staticmethod
    def _freq_to_y(freq):
        lo, hi = np.log10(SPEC_FREQ_LO), np.log10(SPEC_FREQ_HI)
        return SCOPE_Y * (-1.0 + 2.0 * (np.log10(freq) - lo) / (hi - lo))

    @staticmethod
    def _y_to_freq(y):
        lo, hi = np.log10(SPEC_FREQ_LO), np.log10(SPEC_FREQ_HI)
        return 10 ** (lo + (y / SCOPE_Y + 1.0) / 2.0 * (hi - lo))

    @staticmethod
    def _band_yz(band):
        gain = band.gain_db if band.kind == "peak" else 0.0
        y = float(np.clip(Scope3D._freq_to_y(band.freq), -SCOPE_Y, SCOPE_Y))
        return y, gain / DB_RANGE * SCOPE_EQ_Z

    def _paint_eq(self, p, proj):
        yl, ez = SCOPE_Y, SCOPE_EQ_Z
        # faint frame of the plane + the +-12 dB lines
        p.setPen(QPen(QColor(*EQ_RGB, 40), 1.0))
        for db in (-12, 12):
            (x0, x1), (y0, y1) = proj([0, 0], [-yl, yl], [db / DB_RANGE * ez] * 2)
            p.drawLine(QPointF(x0, y0), QPointF(x1, y1))
        for y in (-yl, yl):
            (x0, x1), (y0, y1) = proj([0, 0], [y, y], [-ez, ez])
            p.drawLine(QPointF(x0, y0), QPointF(x1, y1))

        db = np.clip(self.engine.eq.response_db(self._eq_freqs), -DB_RANGE, DB_RANGE)
        ys = self._freq_to_y(self._eq_freqs)
        sx, sy = proj(np.zeros_like(ys), ys, db / DB_RANGE * ez)
        path = polylines_path(sx[None, :], sy[None, :])
        for width, alpha in ((5, 40), (1.6, 235)):
            pen = QPen(QColor(*EQ_RGB, alpha), width)
            pen.setJoinStyle(Qt.RoundJoin)
            p.setPen(pen)
            p.drawPath(path)

        half = EQ_NODE_HALF
        for i, band in enumerate(self.engine.eq.bands):
            y, z = self._band_yz(band)
            x, yy = (float(v) for v in proj(0.0, y, z))
            color = QColor(NODE_COLORS[i % len(NODE_COLORS)])
            off = band.is_off() and band.kind != "peak"
            if off:
                color.setAlpha(110)
            square = QRectF(x - half, yy - half, half * 2, half * 2)
            p.setPen(QPen(QColor(10, 0, 0), 1.2))
            p.setBrush(color)
            p.drawRect(square)
            p.setFont(QFont("Consolas", 6, QFont.Bold))
            p.setPen(QColor(15, 0, 0))
            p.drawText(square, Qt.AlignCenter, f"{band.gain_db:+.0f}" if band.kind == "peak" else band.name)
            p.setFont(QFont("Consolas", 8, QFont.Bold))
            label = "OFF" if off else f"{fmt_freq(band.freq)}Hz"
            label_rect = QRectF(x - 30, yy - half - 15, 60, 13)
            p.setPen(QColor(0, 0, 0, 210))
            p.drawText(label_rect.translated(0, 1), Qt.AlignCenter, label)
            p.setPen(TEXT_BRIGHT)
            p.drawText(label_rect, Qt.AlignCenter, label)
            if not off:
                p.setFont(QFont("Consolas", 7))
                p.setPen(TEXT_DIM)
                p.drawText(QRectF(x - 30, yy + half + 1, 60, 11), Qt.AlignCenter, f"Q{band.q:.2f}")

    def _eq_hit(self, pos):
        """-> (band index, camera) of the nearest node under pos, or None."""
        best, best_d = None, EQ_HIT_RADIUS ** 2
        for rect, cam in self._cams:
            if not rect.contains(pos):
                continue
            for i, band in enumerate(self.engine.eq.bands):
                y, z = self._band_yz(band)
                x, yy = self._project(0.0, y, z, *cam)
                d = (pos.x() - x) ** 2 + (pos.y() - yy) ** 2
                if d <= best_d:
                    best, best_d = (i, cam), d
        return best

    def _unproject_yz(self, pos, cam, y, z):
        """Screen point -> (y, z) on the x = 0 plane, by a few Newton steps
        from the node's current position (lstsq copes with an edge-on plane)."""
        target = np.array([pos.x(), pos.y()])
        v = np.array([y, z], dtype=float)
        eps = 1e-3
        for _ in range(4):
            f0 = np.array(self._project(0.0, v[0], v[1], *cam), dtype=float)
            fy = np.array(self._project(0.0, v[0] + eps, v[1], *cam), dtype=float)
            fz = np.array(self._project(0.0, v[0], v[1] + eps, *cam), dtype=float)
            jac = np.column_stack([(fy - f0) / eps, (fz - f0) / eps])
            v = v + np.linalg.lstsq(jac, target - f0, rcond=None)[0]
        return float(v[0]), float(v[1])

    def _eq_changed(self, commit=False):
        if self.on_eq_change:
            self.on_eq_change()
        if commit and self.on_eq_commit:
            self.on_eq_commit()
        self.update()

    @staticmethod
    def _stroke_buckets(p, sx, sy, fade, ages, rgb_new, rgb_old):
        """One path per opacity bucket (not per frame) keeps draw calls few."""
        bucket = np.minimum((fade * SCOPE_ALPHA_BUCKETS).astype(int), SCOPE_ALPHA_BUCKETS - 1)
        for b in range(SCOPE_ALPHA_BUCKETS):
            sel = (bucket == b) & (fade > 0.01)
            if not sel.any():
                continue
            t = float(np.clip(ages[sel].mean() / TIMELINE_SECONDS, 0, 1))
            r, g, bl = (rgb_new * (1 - t) + rgb_old * t).astype(int)
            alpha = (b + 0.5) / SCOPE_ALPHA_BUCKETS
            p.setPen(QPen(QColor(r, g, bl, int(200 * alpha)), 1.0))
            p.drawPath(polylines_path(sx[sel], sy[sel]))

    def _paint_axes(self, p, proj):
        d, wz, yl = SCOPE_DEPTH, SCOPE_WAVE_Z, SCOPE_Y
        p.setFont(QFont("Consolas", 8))

        def line(a, b):
            (x0, x1), (y0, y1) = proj([a[0], b[0]], [a[1], b[1]], [a[2], b[2]])
            p.setPen(QPen(SCOPE_AXIS, 1.0))
            p.drawLine(QPointF(x0, y0), QPointF(x1, y1))

        def text(pt, s, color=SCOPE_TEXT):
            x, y = proj(*pt)
            p.setPen(color)
            p.drawText(QPointF(float(x), float(y)), s)

        line((-d - 0.15, 0, 0), (d + 0.15, 0, 0))
        line((0, -yl - 0.1, 0), (0, yl + 0.1, 0))
        line((0, 0, -wz - 0.1), (0, 0, SCOPE_SPEC_Z + 0.1))
        text((d + 0.2, 0, 0), "X+ WAVE")
        text((-d - 0.25, 0, 0), "X- SPECTRUM")
        text((0, yl + 0.15, 0), "Y")
        text((0, 0, SCOPE_SPEC_Z + 0.14), "Z")

        dim = QColor(SCOPE_TEXT.red(), SCOPE_TEXT.green(), SCOPE_TEXT.blue(), 110)
        for sec in (2, 4, 6, 8, 10):  # time ticks both ways along X
            x = sec / TIMELINE_SECONDS * d
            for sign in (1, -1):
                line((sign * x, -yl - 0.02, 0), (sign * x, -yl - 0.1, 0))
                text((sign * x, -yl - 0.3, 0), f"-{sec}s", dim)
        for amp, lbl in ((1.0, "0dB"), (0.5, "-6")):  # waveform full scale on the live trace
            for sign in (1, -1):
                line((0, -yl, sign * amp * wz), (0, -yl - 0.07, sign * amp * wz))
            text((0, -yl - 0.3, amp * wz), lbl, dim)
        lo, hi = np.log10(SPEC_FREQ_LO), np.log10(SPEC_FREQ_HI)
        for f in SCOPE_HZ_MARKS:  # frequency along Y on the near (oldest spectrum) edge
            y = yl * (-1.0 + 2.0 * (np.log10(f) - lo) / (hi - lo))
            line((-d - 0.04, y, 0), (-d - 0.12, y, 0))
            text((-d - 0.3, y - 0.03, 0), f"{f // 1000}k" if f >= 1000 else str(f), dim)

    def _paint_button(self, p, ch, rect):
        muted = all(self.engine.mute) if ch == "M" else self.engine.mute[ch]
        c = SCOPE_BTN
        p.setPen(QPen(c if not muted else QColor(70, 60, 90), 1))
        p.setBrush(QColor(c.red() // 3, c.green() // 3, c.blue() // 3) if not muted else QColor(12, 10, 20))
        p.drawRect(rect)
        p.setFont(QFont("Consolas", 8, QFont.Bold))
        p.setPen(QColor(235, 235, 255) if not muted else QColor(90, 80, 110))
        p.drawText(rect, Qt.AlignCenter, "M" if ch == "M" else "LR"[ch])

    def _paint_toggle(self, p):
        p.setFont(QFont("Consolas", 7, QFont.Bold))
        for rect, label, on in zip(self._toggle_rects(), ("MIX", "CH"),
                                   (self.mode == "mix", self.mode == "channels")):
            p.setPen(QPen(SCOPE_BTN if on else QColor(70, 60, 90), 1))
            p.setBrush(QColor(45, 50, 110) if on else QColor(12, 10, 20))
            p.drawRect(rect)
            p.setPen(QColor(235, 235, 255) if on else QColor(110, 100, 140))
            p.drawText(rect, Qt.AlignCenter, label)

    # ---- mouse ------------------------------------------------------------------
    def mousePressEvent(self, event):
        pos = event.position()
        for rect, mode in zip(self._toggle_rects(), ("mix", "channels")):
            if rect.contains(pos):
                if self.mode != mode:
                    self.mode = mode
                    if self.on_mode_change:
                        self.on_mode_change()
                    self.update()
                return
        _, _, buttons = self._layout()
        for ch, b in buttons.items():
            if b.contains(pos):
                if ch == "M":
                    state = not all(self.engine.mute)
                    self.engine.set_channel_mute(0, state)
                    self.engine.set_channel_mute(1, state)
                else:
                    self.engine.set_channel_mute(ch, not self.engine.mute[ch])
                if self.on_mute_change:
                    self.on_mute_change()
                self.update()
                return
        self._eq_drag = self._eq_hit(pos)
        if self._eq_drag is None:
            self._orbit = (pos, self.yaw, self.pitch)

    def mouseMoveEvent(self, event):
        if self._eq_drag is not None:
            idx, cam = self._eq_drag
            band = self.engine.eq.bands[idx]
            y, z = self._unproject_yz(event.position(), cam, *self._band_yz(band))
            freq = self._y_to_freq(y)
            if band.kind == "peak":
                band.set_params(freq=freq, gain_db=z / SCOPE_EQ_Z * DB_RANGE)
            else:
                band.set_params(freq=freq)
            self._eq_changed()
            return
        if self._orbit is None:
            return
        start, yaw, pitch = self._orbit
        delta = event.position() - start
        self.yaw = yaw + delta.x() * 0.3
        self.pitch = float(np.clip(pitch - delta.y() * 0.3, 0.0, SCOPE_PITCH_MAX))
        self.update()

    def mouseReleaseEvent(self, event):
        if self._eq_drag is not None:
            self._eq_changed(commit=True)
        self._eq_drag = None
        self._orbit = None

    def mouseDoubleClickEvent(self, event):
        hit = self._eq_hit(event.position())
        if hit is not None:
            self.engine.eq.reset_band(hit[0])
            self._eq_changed(commit=True)
            return
        self.yaw, self.pitch, self.zoom = SCOPE_DEFAULT_VIEW
        self.update()

    def wheelEvent(self, event):
        hit = self._eq_hit(event.position())
        if hit is not None:
            band = self.engine.eq.bands[hit[0]]
            band.set_params(q=band.q * Q_WHEEL_STEP ** (event.angleDelta().y() / 120.0))
            self._eq_changed(commit=True)
            event.accept()
            return
        self.zoom = float(np.clip(self.zoom * 1.1 ** (event.angleDelta().y() / 120.0), 0.4, 4.0))
        self.update()
        event.accept()


PANEL_STYLE = """
QLabel { color:#ffb0b0; font-family:Consolas; font-size:10px; }
QCheckBox { color:#ffd9d9; font-family:Consolas; font-size:10px; }
QCheckBox::indicator { width:11px; height:11px; border:1px solid #ff2020; background:#140505; }
QCheckBox::indicator:checked { background:#ff3030; }
QComboBox, QDoubleSpinBox { background:#140505; color:#ffd9d9; border:1px solid #ff2020;
    padding:1px 3px; border-radius:0px; font-family:Consolas; font-size:10px; }
QComboBox QAbstractItemView { background:#140505; color:#ffd9d9; selection-background-color:#5a1818; }
QSlider::groove:horizontal { background:#241313; height:4px; }
QSlider::sub-page:horizontal { background:#5a1f1f; }
QSlider::handle:horizontal { background:#ff3030; width:10px; margin:-5px 0; }
QPushButton { background:#2a0a0a; color:#ffd9d9; border:1px solid #ff2020; padding:2px 6px;
    font-family:Consolas; font-size:9px; font-weight:bold; border-radius:0px; }
"""


class DeHummerPanel(QWidget):
    """Audition-style DeHummer controls: presets (built-in + user), base
    frequency / Q / gain, number of harmonics, harmonic slope, hum-only."""

    CUSTOM = "(custom)"
    USER_PREFIX = "* "

    def __init__(self, dehummer, user_presets, on_change=None):
        super().__init__()
        self.hum = dehummer
        self.user_presets = user_presets  # dict, mutated in place; owner persists it
        self.on_change = on_change
        self._syncing = False
        self.setStyleSheet(PANEL_STYLE)

        grid = QGridLayout(self)
        grid.setContentsMargins(4, 6, 2, 2)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(4)

        title = QLabel("DE-HUMMER")
        title.setStyleSheet("color:#ff3030; font-size:9px; font-weight:bold; letter-spacing:1px;")
        self.enable_chk = QCheckBox("ON")
        grid.addWidget(title, 0, 0)
        grid.addWidget(self.enable_chk, 0, 1, alignment=Qt.AlignRight)

        self.preset_combo = QComboBox()
        grid.addWidget(self.preset_combo, 1, 0, 1, 2)
        btns = QHBoxLayout()
        self.save_btn = QPushButton("SAVE PRESET")
        self.del_btn = QPushButton("DELETE")
        btns.addWidget(self.save_btn)
        btns.addWidget(self.del_btn)
        grid.addLayout(btns, 2, 0, 1, 2)

        def spin(lo, hi, decimals, suffix, step):
            s = QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setDecimals(decimals)
            s.setSuffix(suffix)
            s.setSingleStep(step)
            s.setKeyboardTracking(False)
            return s

        self.freq_spin = spin(HUM_FREQ_MIN, HUM_FREQ_MAX, 1, " Hz", 1.0)
        self.q_spin = spin(HUM_Q_MIN, HUM_Q_MAX, 1, "", 5.0)
        self.gain_spin = spin(HUM_GAIN_MIN, HUM_GAIN_MAX, 1, " dB", 1.0)
        self.harm_combo = QComboBox()
        self.harm_combo.addItems([str(i) for i in range(1, HUM_MAX_HARMONICS + 1)])
        self.slope_slider = QSlider(Qt.Horizontal)
        self.slope_slider.setRange(0, 100)
        self.slope_label = QLabel("50%")
        slope_row = QHBoxLayout()
        slope_row.addWidget(self.slope_slider, stretch=1)
        slope_row.addWidget(self.slope_label)
        self.hum_only_chk = QCheckBox("Output Hum Only")

        rows = [("Frequency", self.freq_spin), ("Q", self.q_spin), ("Gain", self.gain_spin),
                ("Harmonics", self.harm_combo)]
        for i, (name, widget) in enumerate(rows, start=3):
            grid.addWidget(QLabel(name), i, 0)
            grid.addWidget(widget, i, 1)
        grid.addWidget(QLabel("Harm. Slope"), 7, 0)
        grid.addLayout(slope_row, 7, 1)
        grid.addWidget(self.hum_only_chk, 8, 0, 1, 2)

        self.enable_chk.toggled.connect(self._on_controls)
        self.freq_spin.valueChanged.connect(self._on_controls)
        self.q_spin.valueChanged.connect(self._on_controls)
        self.gain_spin.valueChanged.connect(self._on_controls)
        self.harm_combo.currentIndexChanged.connect(self._on_controls)
        self.slope_slider.valueChanged.connect(self._on_controls)
        self.hum_only_chk.toggled.connect(self._on_controls)
        self.preset_combo.currentIndexChanged.connect(self._on_preset)
        self.save_btn.clicked.connect(self._save_preset)
        self.del_btn.clicked.connect(self._delete_preset)

        self._reload_presets()
        self.sync_from_engine()

    # ---- presets ----------------------------------------------------------------
    def _reload_presets(self, select=None):
        self._syncing = True
        self.preset_combo.clear()
        self.preset_combo.addItem(self.CUSTOM)
        self.preset_combo.addItems(list(DEHUMMER_PRESETS))
        self.preset_combo.addItems([self.USER_PREFIX + n for n in sorted(self.user_presets)])
        idx = self.preset_combo.findText(select) if select else -1
        self.preset_combo.setCurrentIndex(max(idx, 0))
        self._syncing = False

    def _preset_params(self, text):
        if text in DEHUMMER_PRESETS:
            return DEHUMMER_PRESETS[text]
        if text.startswith(self.USER_PREFIX):
            return self.user_presets.get(text[len(self.USER_PREFIX):])
        return None

    def _matching_preset(self):
        state = self.hum.get_state()
        for i in range(1, self.preset_combo.count()):
            params = self._preset_params(self.preset_combo.itemText(i))
            if params and all(abs(float(params[k]) - float(state[k])) < 1e-6 for k in params):
                return i
        return 0

    def _on_preset(self, _idx):
        if self._syncing:
            return
        params = self._preset_params(self.preset_combo.currentText())
        if params is None:
            return
        self.hum.set_params(**params)
        self.hum.set_enabled(True)
        self.sync_from_engine()
        if self.on_change:
            self.on_change()

    def _save_preset(self):
        name, ok = QInputDialog.getText(self, "Save DeHummer preset", "Preset name:")
        name = name.strip()
        if not ok or not name:
            return
        state = self.hum.get_state()
        state.pop("enabled")
        self.user_presets[name] = state
        self._reload_presets(select=self.USER_PREFIX + name)
        if self.on_change:
            self.on_change()

    def _delete_preset(self):
        text = self.preset_combo.currentText()
        if not text.startswith(self.USER_PREFIX):
            return  # built-ins stay
        self.user_presets.pop(text[len(self.USER_PREFIX):], None)
        self._reload_presets()
        self.sync_from_engine()
        if self.on_change:
            self.on_change()

    # ---- controls <-> engine ----------------------------------------------------
    def _on_controls(self, *_):
        if self._syncing:
            return
        self.hum.set_params(freq=self.freq_spin.value(), q=self.q_spin.value(),
                            gain_db=self.gain_spin.value(),
                            harmonics=self.harm_combo.currentIndex() + 1,
                            slope=self.slope_slider.value(), hum_only=self.hum_only_chk.isChecked())
        self.hum.set_enabled(self.enable_chk.isChecked())
        self.sync_from_engine()
        if self.on_change:
            self.on_change()

    def sync_from_engine(self):
        """Pull values from the engine (e.g. after a drag/wheel on the EQ view)."""
        h = self.hum
        self._syncing = True
        self.enable_chk.setChecked(h.enabled)
        self.freq_spin.setValue(h.freq)
        self.q_spin.setValue(h.q)
        self.gain_spin.setValue(h.gain_db)
        self.harm_combo.setCurrentIndex(h.harmonics - 1)
        self.slope_slider.setValue(int(round(h.slope)))
        self.slope_label.setText(f"{h.slope:.0f}%")
        self.hum_only_chk.setChecked(h.hum_only)
        self.preset_combo.setCurrentIndex(self._matching_preset())
        self._syncing = False


def make_tray_icon():
    size = 64
    pix = QPixmap(size, size)
    pix.fill(Qt.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing)
    grad = QRadialGradient(size / 2, size / 2, size / 2)
    grad.setColorAt(0, QColor(255, 45, 149))
    grad.setColorAt(1, QColor(80, 20, 100))
    p.setBrush(QBrush(grad))
    p.setPen(QPen(QColor(80, 240, 255), 3))
    p.drawEllipse(3, 3, size - 6, size - 6)
    p.setPen(QColor(255, 255, 255))
    f = QFont("Consolas", 20, QFont.Bold)
    p.setFont(f)
    p.drawText(pix.rect(), Qt.AlignCenter, "FV")
    p.end()
    return QIcon(pix)


def retro_button(text, color="#3a1010"):
    btn = QPushButton(text)
    btn.setStyleSheet(f"""
        QPushButton {{
            background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                stop:0 #501818, stop:1 {color});
            color: #ffd9d9; border: 2px solid #ff2020; border-radius: 0px;
            padding: 6px 10px; font-weight: bold; font-size: 11px; font-family: Consolas;
        }}
        QPushButton:pressed {{ background: #200a0a; }}
        QPushButton:disabled {{ color: #7a5555; border-color: #5a2a2a; background: #1a0d0d; }}
    """)
    return btn


class VolumeFader(QWidget):
    """Lateral master-volume fader: monitor-only gain, does not touch the recording."""

    def __init__(self, on_change=None, on_fx_toggle=None):
        super().__init__()
        self.on_change = on_change
        self.setFixedWidth(64)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)

        title = QLabel("MASTER")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color:#ff3030; font-size:9px; font-weight:bold; letter-spacing:1px;")

        self.readout = QLabel("100%")
        self.readout.setAlignment(Qt.AlignCenter)
        self.readout.setStyleSheet("color:#ffd9d9; font-size:10px; font-family:Consolas;")

        self.slider = QSlider(Qt.Vertical)
        self.slider.setMinimum(0)
        self.slider.setMaximum(150)
        self.slider.setValue(100)
        self.slider.setStyleSheet("""
            QSlider::groove:vertical { background:#1a0a0a; width:8px; border-radius:0px; }
            QSlider::handle:vertical {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 #ff2020, stop:0.5 #ffffff, stop:1 #ff2020);
                border: 1px solid #200a0a; height:16px; margin:0 -8px; border-radius:0px;
            }
            QSlider::sub-page:vertical { background:#5a1f1f; border-radius:0px; }
            QSlider::add-page:vertical { background:#241313; border-radius:0px; }
        """)
        self.slider.valueChanged.connect(self._on_value_changed)

        # FX icon: shows/hides the effects sidebar (EQ bands, DeHummer, ...)
        self.fx_btn = QPushButton("FX")
        self.fx_btn.setCheckable(True)
        self.fx_btn.setFixedSize(26, 16)
        self.fx_btn.setToolTip("Show / hide effects (EQ bands, DeHummer)")
        self.fx_btn.setStyleSheet("""
            QPushButton { background:#2a0a0a; color:#ff8080; border:1px solid #ff2020;
                font-family:Consolas; font-size:8px; font-weight:bold; border-radius:0px; padding:0; }
            QPushButton:checked { background:#ff3030; color:#140505; }
        """)
        if on_fx_toggle:
            self.fx_btn.toggled.connect(on_fx_toggle)

        layout.addWidget(self.fx_btn, alignment=Qt.AlignHCenter)
        layout.addWidget(title)
        layout.addWidget(self.slider, stretch=1, alignment=Qt.AlignHCenter)
        layout.addWidget(self.readout)

    def _on_value_changed(self, value):
        self.readout.setText(f"{value}%")
        if self.on_change:
            self.on_change(value)

    def set_value(self, value):
        self.slider.setValue(int(value))

    def value(self):
        return self.slider.value()


class RadioRecorder(QWidget):
    WINDOW_WIDTH = 1240
    WINDOW_HEIGHT = 900

    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setFixedSize(self.WINDOW_WIDTH, self.WINDOW_HEIGHT)

        self.engine = AudioEngine(output_dir=OUTPUT_DIR)
        self._drag_pos = None
        self.settings = settings_store.load()
        self._build_ui()
        self._build_tray()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(33)

        if self.settings.get("auto_power_on", True) and self.capture_combo.count() > 0:
            self._toggle_power()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 12, 14, 12)
        outer.setSpacing(8)

        title_row = QHBoxLayout()
        title = QLabel("F E R R A R I V O Z   A I R - C H E C K   ' 8 4")
        title.setStyleSheet(
            "color:#ff2020; font-weight:bold; font-size:15px; letter-spacing:2px;"
        )
        min_btn = QPushButton("_")
        min_btn.setFixedSize(22, 22)
        min_btn.setStyleSheet("background:#2a0a0a; color:#ffd9d9; border-radius:0px; font-weight:bold;")
        min_btn.clicked.connect(self._minimize_to_tray)

        close_btn = QPushButton("X")
        close_btn.setFixedSize(22, 22)
        close_btn.setStyleSheet("background:#7a1010; color:white; border-radius:0px; font-weight:bold;")
        close_btn.clicked.connect(self._minimize_to_tray)
        title_row.addWidget(title)
        title_row.addStretch()
        title_row.addWidget(min_btn)
        title_row.addWidget(close_btn)
        outer.addLayout(title_row)

        dev_row = QHBoxLayout()
        src_lbl = QLabel("SOURCE")
        mon_lbl = QLabel("LISTEN ON")
        for lbl in (src_lbl, mon_lbl):
            lbl.setStyleSheet("color:#ff3030; font-size:10px; font-weight:bold;")
        self.capture_combo = QComboBox()
        self.capture_combo.addItems(list_capture_devices())
        self.monitor_combo = QComboBox()
        self.monitor_combo.addItems(list_monitor_devices())
        combo_style = ("background:#140505; color:#ffd9d9; border:1px solid #ff2020; "
                       "padding:3px; border-radius:0px;")
        self.capture_combo.setStyleSheet(combo_style)
        self.monitor_combo.setStyleSheet(combo_style)
        self._select_combo_default(self.capture_combo, self.settings.get("capture_device"),
                                    prefer_substrings=["CABLE Output"])
        self._select_combo_default(self.monitor_combo, self.settings.get("monitor_device"),
                                    prefer_substrings=["Focusrite", "Speakers", "Headphones"])
        self.capture_combo.currentIndexChanged.connect(lambda _: self._save_settings())
        self.monitor_combo.currentIndexChanged.connect(lambda _: self._save_settings())
        dev_row.addWidget(src_lbl)
        dev_row.addWidget(self.capture_combo, stretch=1)
        dev_row.addSpacing(12)
        dev_row.addWidget(mon_lbl)
        dev_row.addWidget(self.monitor_combo, stretch=1)
        outer.addLayout(dev_row)

        self.engine.eq.set_state(self.settings.get("eq_bands"))
        self.engine.dehummer.set_state(self.settings.get("dehummer"))
        for ch, muted in enumerate((self.settings.get("mute") or [False, False])[:2]):
            self.engine.set_channel_mute(ch, muted)
        if not isinstance(self.settings.get("dehummer_presets"), dict):
            self.settings["dehummer_presets"] = {}

        eq_row = QHBoxLayout()
        self.band_list = BandListPanel()
        self.hum_panel = DeHummerPanel(self.engine.dehummer, self.settings["dehummer_presets"],
                                       on_change=self._on_hum_panel_change)
        self.eq_widget = ParametricEQ(self.engine, on_change=self._save_settings,
                                      on_hum_change=self.hum_panel.sync_from_engine)
        self.band_list.refresh(self.engine.eq.bands)
        self.scope = Scope3D(self.engine, mode=self.settings.get("scope_mode", "mix"),
                             on_mute_change=self._on_mute_change, on_mode_change=self._save_settings,
                             on_eq_change=lambda: self.band_list.refresh(self.engine.eq.bands),
                             on_eq_commit=self._save_settings)

        # EQ view + the 3D scope share what used to be the EQ area
        monitors = QVBoxLayout()
        monitors.setSpacing(4)
        monitors.addWidget(self.eq_widget, stretch=5)
        monitors.addWidget(self.scope, stretch=7)
        eq_row.addLayout(monitors, stretch=1)

        side = QWidget()
        side.setFixedWidth(SIDE_PANEL_WIDTH)
        side.setVisible(False)  # effects sidebar stays hidden until the FX icon calls it
        self.side_panel = side
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.addWidget(self.band_list)
        side_layout.addWidget(self.hum_panel)
        side_layout.addStretch(1)
        eq_row.addWidget(side)

        self.volume_fader = VolumeFader(on_change=self._on_volume_changed, on_fx_toggle=side.setVisible)
        saved_volume = self.settings.get("master_volume", 100)
        self.volume_fader.set_value(saved_volume)
        self.engine.set_master_volume(saved_volume / 100.0)
        eq_row.addWidget(self.volume_fader)
        outer.addLayout(eq_row, stretch=1)

        self.vu = VuStrip()
        outer.addWidget(self.vu)

        btn_row = QHBoxLayout()
        self.power_btn = retro_button("POWER", "#1f4a2e")
        self.rec_btn = retro_button("* REC", "#5a1024")
        self.pause_btn = retro_button("PAUSE")
        self.stop_btn = retro_button("STOP")
        self.flat_btn = retro_button("FLAT EQ")
        self.folder_btn = retro_button("OPEN FOLDER", "#1a2f5a")
        for b in (self.power_btn, self.rec_btn, self.pause_btn, self.stop_btn,
                  self.flat_btn, self.folder_btn):
            btn_row.addWidget(b)
        outer.addLayout(btn_row)

        self.status_label = QLabel("OFF AIR")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color:#c05050; font-size:10px; letter-spacing:1px;")
        outer.addWidget(self.status_label)

        self.power_btn.clicked.connect(self._toggle_power)
        self.rec_btn.clicked.connect(self._on_record)
        self.pause_btn.clicked.connect(self._on_pause)
        self.stop_btn.clicked.connect(self._on_stop)
        self.flat_btn.clicked.connect(self._reset_flat)
        self.folder_btn.clicked.connect(self._open_folder)

        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.rec_btn.setEnabled(False)

    # ---- window chrome (frameless drag) --------------------------------------
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and event.position().y() < 40:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event):
        if self._drag_pos is not None and event.buttons() & Qt.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)

    def mouseReleaseEvent(self, event):
        self._drag_pos = None

    def paintEvent(self, event):
        p = QPainter(self)
        grad = QLinearGradient(0, 0, 0, self.height())
        grad.setColorAt(0, BG_TOP)
        grad.setColorAt(1, BG_BOTTOM)
        p.setBrush(QBrush(grad))
        p.setPen(QPen(PANEL_BORDER, 2))
        p.drawRect(self.rect().adjusted(1, 1, -1, -1))
        p.end()

    # ---- controls -------------------------------------------------------------
    def _toggle_power(self):
        if self.power_btn.text() == "POWER":
            self.engine.set_capture_device(self.capture_combo.currentText())
            self.engine.set_monitor_device(self.monitor_combo.currentText())
            try:
                self.engine.start()
            except Exception as e:
                self.status_label.setText(f"ERROR: {e}")
                return
            self.power_btn.setText("POWER (ON)")
            self.rec_btn.setEnabled(True)
            self.capture_combo.setEnabled(False)
            self.monitor_combo.setEnabled(False)
            self.status_label.setText("MONITORING (not recording)")
        else:
            self._on_stop()
            self.engine.stop()
            self.power_btn.setText("POWER")
            self.rec_btn.setEnabled(False)
            self.capture_combo.setEnabled(True)
            self.monitor_combo.setEnabled(True)
            self.status_label.setText("OFF AIR")

    def _on_record(self):
        self.engine.start_recording()
        self.rec_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("RECORDING")

    def _on_pause(self):
        if self.engine.is_paused:
            self.engine.resume_recording()
            self.pause_btn.setText("PAUSE")
            self.status_label.setText("RECORDING")
        else:
            self.engine.pause_recording()
            self.pause_btn.setText("RESUME")
            self.status_label.setText("PAUSED")

    def _on_stop(self):
        if self.engine.current_recording_path or self.engine.is_recording:
            self.engine.stop_recording()
        self.rec_btn.setEnabled(self.power_btn.text() != "POWER")
        self.pause_btn.setEnabled(False)
        self.pause_btn.setText("PAUSE")
        self.stop_btn.setEnabled(False)
        if self.power_btn.text() != "POWER":
            self.status_label.setText("MONITORING (not recording)")

    def _reset_flat(self):
        self.engine.eq.reset_flat()
        self.band_list.refresh(self.engine.eq.bands)
        self.scope.update()
        self._save_settings()

    def _open_folder(self):
        os.startfile(str(OUTPUT_DIR))

    # ---- persisted settings ---------------------------------------------------
    def _select_combo_default(self, combo, saved_name, prefer_substrings):
        if saved_name:
            idx = combo.findText(saved_name)
            if idx >= 0:
                combo.setCurrentIndex(idx)
                return
        for pref in prefer_substrings:
            # Windows lists the same physical device once per audio backend
            # (MME/DirectSound truncate the name, WASAPI doesn't) -- prefer
            # the longest (untruncated) match, which is the reliable one.
            candidates = [i for i in range(combo.count()) if pref.lower() in combo.itemText(i).lower()]
            if candidates:
                best = max(candidates, key=lambda i: len(combo.itemText(i)))
                combo.setCurrentIndex(best)
                return

    def _on_volume_changed(self, value):
        self.engine.set_master_volume(value / 100.0)
        self._save_settings()

    def _on_hum_panel_change(self):
        self.eq_widget.update()
        self._save_settings()

    def _on_mute_change(self):
        self.scope.update()
        self._save_settings()

    def _save_settings(self):
        self.settings["capture_device"] = self.capture_combo.currentText()
        self.settings["monitor_device"] = self.monitor_combo.currentText()
        self.settings.pop("gains", None)  # pre-5-band layout
        self.settings.pop("freqs", None)
        self.settings["eq_bands"] = self.engine.eq.get_state()
        self.settings["dehummer"] = self.engine.dehummer.get_state()
        self.settings["mute"] = list(self.engine.mute)
        self.settings["scope_mode"] = self.scope.mode
        self.settings["master_volume"] = self.volume_fader.value()
        settings_store.save(self.settings)

    # ---- system tray ----------------------------------------------------------
    def _build_tray(self):
        self.tray = QSystemTrayIcon(make_tray_icon(), self)
        self.tray.setToolTip("FerrariVoz Air-Check Recorder")

        menu = QMenu()
        show_action = QAction("Show", self)
        show_action.triggered.connect(self._restore_from_tray)
        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self._quit)
        menu.addAction(show_action)
        menu.addSeparator()
        menu.addAction(quit_action)

        self.tray.setContextMenu(menu)
        self.tray.activated.connect(self._on_tray_activated)
        self.tray.show()
        self._tray_notified = False

    def _minimize_to_tray(self):
        self.hide()
        if not self._tray_notified:
            self.tray.showMessage(
                "FerrariVoz Air-Check Recorder",
                "Still running in the tray. Recording continues if it's active. "
                "Right-click the tray icon to Quit.",
                QSystemTrayIcon.Information, 4000,
            )
            self._tray_notified = True

    def _restore_from_tray(self):
        self.show()
        self.raise_()
        self.activateWindow()

    def _on_tray_activated(self, reason):
        if reason == QSystemTrayIcon.Trigger:
            if self.isVisible():
                self.hide()
            else:
                self._restore_from_tray()

    def _quit(self):
        self._save_settings()
        self.engine.stop()
        self.tray.hide()
        QApplication.instance().quit()

    def _tick(self):
        db_l, db_r, _waveform, spectrum_l, spectrum_r = self.engine.snapshot()
        self.eq_widget.update_data(max(db_l, db_r), spectrum_l, spectrum_r)
        self.vu.set_levels(db_l, db_r)
        self.scope.update_data(*self.engine.scope_snapshot())
        if self.engine.last_error:
            self.status_label.setText(f"ERROR: {self.engine.last_error}")

    def closeEvent(self, event):
        self.engine.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    win = RadioRecorder()
    if "--start-hidden" in sys.argv:
        win.tray.showMessage(
            "FerrariVoz Air-Check Recorder",
            "Running in the tray. Click the icon to open it.",
            QSystemTrayIcon.Information, 4000,
        )
    else:
        win.show()
    sys.exit(app.exec())
