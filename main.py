# -*- coding: utf-8 -*-
"""Floating retro-80s 'air-check' recorder: per-device audio tap, 5-band EQ curve,
DeHummer, and scrolling waveform / spectral timeline monitors."""
import os
import sys
from pathlib import Path

import numpy as np

from PySide6.QtCore import Qt, QTimer, QPointF, QRectF
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
    AudioEngine, list_capture_devices, list_monitor_devices, TIMELINE_SECONDS, WAVE_HISTORY,
    SPEC_HISTORY, SPEC_ROWS, SPEC_FREQ_LO, SPEC_FREQ_HI, spec_scale,
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
CURVE_COLOR = QColor(255, 70, 70)
CURVE_GLOW = QColor(255, 70, 70, 60)
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


def y_to_db(y, rect):
    frac = (rect.bottom() - y) / rect.height()
    return frac * 2 * DB_RANGE - DB_RANGE


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
    """Live spectrum behind the 5-band EQ curve (left +-15 dB scale), plus the
    DeHummer notch comb in red on its own 0..-80 dB right-hand scale.
    Drag a node = move it, mouse wheel over a node = narrower/wider (Q),
    double-click = reset that band."""

    HIT_RADIUS = 14

    def __init__(self, engine, on_change=None, band_list=None, on_hum_change=None):
        super().__init__()
        self.engine = engine
        self.on_change = on_change
        self.on_hum_change = on_hum_change
        self.band_list = band_list
        self.setMinimumHeight(220)

        self.spectrum_freqs = np.logspace(np.log10(FREQ_MIN), np.log10(FREQ_MAX), SPECTRUM_BIN_COUNT)
        engine.set_spectrum_bin_count(SPECTRUM_BIN_COUNT)
        self.spectrum_l = np.zeros(SPECTRUM_BIN_COUNT)
        self.spectrum_r = np.zeros(SPECTRUM_BIN_COUNT)
        self.level_db = -60.0
        self._dragging = None  # band index, or HUM_NODE
        self.curve_freqs = np.logspace(np.log10(FREQ_MIN), np.log10(FREQ_MAX), 400)
        self.hum_freqs = np.logspace(np.log10(FREQ_MIN), np.log10(FREQ_MAX), 900)
        self.setMouseTracking(True)

    def refresh(self):
        if self.band_list is not None:
            self.band_list.refresh(self.engine.eq.bands)
        self.update()

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
        self._draw_curve(p, rect)
        p.restore()
        self._draw_hum_axis(p, rect)
        self._draw_nodes(p, rect)
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

    def _draw_curve(self, p, rect):
        db_vals = np.clip(self.engine.eq.response_db(self.curve_freqs), -DB_RANGE, DB_RANGE)
        self._stroke(p, self._path(self.curve_freqs, [db_to_y(d, rect) for d in db_vals], rect),
                     CURVE_COLOR)

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

    def _band_pos(self, band, rect):
        gain = band.gain_db if band.kind == "peak" else 0.0
        return freq_to_x(band.freq, rect), db_to_y(gain, rect)

    def _draw_nodes(self, p, rect):
        half = 9  # node square half-size
        for i, band in enumerate(self.engine.eq.bands):
            x, y = self._band_pos(band, rect)
            color = QColor(NODE_COLORS[i % len(NODE_COLORS)])
            off = band.is_off() and band.kind != "peak"
            if off:
                color.setAlpha(110)
            square = QRectF(x - half, y - half, half * 2, half * 2)

            p.setPen(QPen(QColor(20, 0, 0), 1.5))
            p.setBrush(color)
            p.drawRect(square)

            # gain (bells) or HP/LP tag sits inside the square itself
            p.setFont(QFont("Consolas", 7, QFont.Bold))
            p.setPen(QColor(15, 0, 0))
            inner = f"{band.gain_db:+.0f}" if band.kind == "peak" else band.name
            p.drawText(square, Qt.AlignCenter, inner)

            # frequency floats above, Q (narrowness) below
            p.setFont(QFont("Consolas", 9, QFont.Bold))
            label = "OFF" if off else f"{fmt_freq(band.freq)}Hz"
            label_rect = QRectF(x - 30, y - half - 17, 60, 14)
            p.setPen(QColor(10, 0, 0, 210))
            p.drawText(label_rect.translated(0, 1), Qt.AlignCenter, label)
            p.setPen(TEXT_BRIGHT)
            p.drawText(label_rect, Qt.AlignCenter, label)
            if not off:
                p.setFont(QFont("Consolas", 7))
                p.setPen(TEXT_DIM)
                p.drawText(QRectF(x - 30, y + half + 1, 60, 11), Qt.AlignCenter, f"Q{band.q:.2f}")

    # ---- mouse interaction ---------------------------------------------------
    def _hit(self, pos, rect):
        r2 = self.HIT_RADIUS ** 2
        if self.engine.dehummer.enabled:
            x, y = self._hum_node_pos(rect)
            if (pos.x() - x) ** 2 + (pos.y() - y) ** 2 <= r2:
                return HUM_NODE
        best, best_d = None, None
        for i, band in enumerate(self.engine.eq.bands):
            x, y = self._band_pos(band, rect)
            d = (pos.x() - x) ** 2 + (pos.y() - y) ** 2
            if d <= r2 and (best_d is None or d < best_d):
                best, best_d = i, d
        return best

    def mousePressEvent(self, event):
        self._dragging = self._hit(event.position(), self.plot_rect())

    def mouseDoubleClickEvent(self, event):
        idx = self._hit(event.position(), self.plot_rect())
        if idx is None or idx == HUM_NODE:
            return
        self.engine.eq.reset_band(idx)
        self.refresh()
        if self.on_change:
            self.on_change()

    def mouseMoveEvent(self, event):
        if self._dragging is None:
            return
        rect = self.plot_rect()
        pos = event.position()
        freq = x_to_freq(pos.x(), rect)
        if self._dragging == HUM_NODE:
            self.engine.dehummer.set_params(freq=freq, gain_db=self._hum_db(pos.y(), rect))
            if self.on_hum_change:
                self.on_hum_change()
            self.update()
            return
        band = self.engine.eq.bands[self._dragging]
        if band.kind == "peak":
            band.set_params(freq=freq, gain_db=y_to_db(pos.y(), rect))
        else:
            band.set_params(freq=freq)
        self.refresh()

    def mouseReleaseEvent(self, event):
        if self._dragging is not None and self.on_change:
            self.on_change()
        self._dragging = None

    def wheelEvent(self, event):
        target = self._hit(event.position(), self.plot_rect())
        if target is None:
            return
        factor = Q_WHEEL_STEP ** (event.angleDelta().y() / 120.0)
        if target == HUM_NODE:
            hum = self.engine.dehummer
            hum.set_params(q=hum.q * factor)
            if self.on_hum_change:
                self.on_hum_change()
            self.update()
        else:
            band = self.engine.eq.bands[target]
            band.set_params(q=band.q * factor)
            self.refresh()
        if self.on_change:
            self.on_change()
        event.accept()

    def reset_flat(self):
        self.engine.eq.reset_flat()
        self.refresh()
        if self.on_change:
            self.on_change()


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


# ---- scrolling timeline monitors ---------------------------------------------
LANE_RULER_W = 44   # dB / Hz scale column, right of the lanes
LANE_BTN_W = 20     # channel (mute) buttons column, right of the ruler
LANE_BG = QColor(8, 3, 4)
CH_COLORS = [QColor(255, 70, 70), QColor(80, 240, 160)]  # L red, R green (matches the EQ spectrum)
MONO_COLOR = QColor(255, 210, 60)
WAVE_DB_MARKS = (-3, -6, -12, -18)
SPEC_HZ_MARKS = (100, 200, 400, 600, 800, 1000, 2000, 3000, 4000, 5000, 7000, 10000)
SPEC_DB_FLOOR, SPEC_DB_CEIL = -105.0, -5.0
HEAT_FADE_SECONDS = 3.0  # columns fade out over the last 3 s of the 10 s window
HEAT_LUT = np.stack([
    np.interp(np.linspace(0, 1, 256), [0, .15, .35, .55, .75, .9, 1], ch)
    for ch in ([0, 30, 90, 180, 250, 255, 255],     # R
               [0, 8, 30, 40, 120, 225, 255],      # G
               [0, 50, 130, 140, 40, 80, 230])     # B
], axis=1).astype(np.uint8)  # dark -> purple -> magenta -> orange -> yellow -> white (Audition heat)


class ChannelLanes(QWidget):
    """Base for the per-channel timeline monitors. Stereo: L over R. A muted
    channel's lane disappears (the other takes the whole area) and only its
    button stays; both muted = dark. Mono input: one lane, one 'M' button."""

    TOP = 2  # space above the lanes (subclasses may reserve a time ruler)

    def __init__(self, engine, on_mute_change=None):
        super().__init__()
        self.engine = engine
        self.on_mute_change = on_mute_change
        self.mono = False
        self.setMinimumHeight(110)

    def content_rect(self):
        return QRectF(self.rect()).adjusted(46, self.TOP, -(LANE_RULER_W + LANE_BTN_W), -2)

    def _is_muted(self, ch):
        return all(self.engine.mute) if ch == "M" else self.engine.mute[ch]

    def _layout(self):
        r = self.content_rect()
        chans = ["M"] if self.mono else [0, 1]
        visible = [c for c in chans if not self._is_muted(c)]
        hidden = [c for c in chans if self._is_muted(c)]
        lanes = []
        if visible:
            h = r.height() / len(visible)
            lanes = [(c, QRectF(r.left(), r.top() + i * h, r.width(), h)) for i, c in enumerate(visible)]
        bx = r.right() + LANE_RULER_W + 2
        buttons = {c: QRectF(bx, lane.top() + 2, 16, 16) for c, lane in lanes}
        for j, c in enumerate(reversed(hidden)):  # muted buttons park at the bottom, L above R
            buttons[c] = QRectF(bx, r.bottom() - 18 * (j + 1), 16, 16)
        return r, lanes, buttons

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(3, 2, 5))
        r, lanes, buttons = self._layout()
        p.fillRect(r, QColor(0, 0, 0))
        self.paint_header(p, r)
        for ch, lane in lanes:
            data_ch = 0 if ch == "M" else ch
            p.fillRect(lane, LANE_BG)
            self.paint_lane(p, data_ch, lane)
            self.paint_ruler(p, lane, QRectF(lane.right(), lane.top(), LANE_RULER_W, lane.height()))
            p.setPen(QPen(QColor(90, 30, 30), 1))
            p.drawRect(lane)
        for ch, b in buttons.items():
            self._paint_button(p, ch, b)
        p.end()

    def _paint_button(self, p, ch, rect):
        muted = self._is_muted(ch)
        color = MONO_COLOR if ch == "M" else CH_COLORS[ch]
        p.setPen(QPen(color if not muted else QColor(110, 40, 40), 1))
        p.setBrush(QColor(color.red() // 3, color.green() // 3, color.blue() // 3) if not muted
                   else QColor(25, 8, 8))
        p.drawRect(rect)
        p.setFont(QFont("Consolas", 8, QFont.Bold))
        p.setPen(QColor(255, 235, 235) if not muted else QColor(120, 60, 60))
        p.drawText(rect, Qt.AlignCenter, "M" if ch == "M" else "LR"[ch])

    def mousePressEvent(self, event):
        _, _, buttons = self._layout()
        for ch, b in buttons.items():
            if b.contains(event.position()):
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

    @staticmethod
    def _blit(p, img, rect):
        img = np.ascontiguousarray(img)
        h, w = img.shape[:2]
        qimg = QImage(img.data, w, h, w * 4, QImage.Format_RGBA8888).copy()
        p.drawImage(QPointF(rect.left(), rect.top()), qimg)

    @staticmethod
    def _draw_marks(p, marks, lane):
        """marks: [(y, text)] -- skip any that would collide with one already drawn."""
        p.setFont(QFont("Consolas", 7))
        drawn = []
        for y, text in marks:
            if y < lane.top() + 4 or y > lane.bottom() - 2 or any(abs(y - d) < 10 for d in drawn):
                continue
            drawn.append(y)
            p.setPen(QPen(TEXT_DIM, 1))
            p.drawLine(QPointF(lane.left(), y), QPointF(lane.left() + 4, y))
            p.drawText(QPointF(lane.left() + 7, y + 3), text)

    def paint_header(self, p, r):
        pass

    def paint_lane(self, p, ch, lane):
        raise NotImplementedError

    def paint_ruler(self, p, lane, ruler):
        raise NotImplementedError


class WaveformLanes(ChannelLanes):
    """Rolling 10 s min/max waveform per channel, newest on the right, with a
    dB ruler (-3/-6/-12/-18 mirrored around -inf) and a seconds timeline on top."""

    TOP = 16

    def __init__(self, engine, on_mute_change=None):
        super().__init__(engine, on_mute_change)
        self.wmin = np.zeros((WAVE_HISTORY, 2), dtype=np.float32)
        self.wmax = np.zeros((WAVE_HISTORY, 2), dtype=np.float32)

    def update_data(self, wmin, wmax, mono):
        self.wmin, self.wmax, self.mono = wmin, wmax, mono
        self.update()

    def paint_header(self, p, r):
        p.setFont(QFont("Consolas", 7))
        for sec in range(int(TIMELINE_SECONDS) + 1):
            x = r.right() - sec / TIMELINE_SECONDS * r.width()
            p.setPen(QPen(TEXT_DIM, 1))
            p.drawLine(QPointF(x, r.top() - 4), QPointF(x, r.top()))
            label = "now" if sec == 0 else f"-{sec}s"
            p.drawText(QRectF(x - 20, 0, 40, r.top() - 4), Qt.AlignCenter, label)

    def paint_lane(self, p, ch, lane):
        w, h = int(lane.width()), int(lane.height())
        if w < 2 or h < 4:
            return
        half = (h - 1) / 2.0
        # grid: dB levels + one vertical per second
        p.setPen(QPen(GRID_COLOR_DIM, 1))
        for db in WAVE_DB_MARKS:
            amp = 10 ** (db / 20.0)
            for sign in (1, -1):
                y = lane.top() + half - sign * amp * half
                p.drawLine(QPointF(lane.left(), y), QPointF(lane.right(), y))
        for sec in range(1, int(TIMELINE_SECONDS)):
            x = lane.right() - sec / TIMELINE_SECONDS * lane.width()
            p.drawLine(QPointF(x, lane.top()), QPointF(x, lane.bottom()))

        n = len(self.wmin)
        starts = np.arange(w) * n // w
        mins = np.minimum.reduceat(self.wmin[:, ch], starts)
        maxs = np.maximum.reduceat(self.wmax[:, ch], starts)
        y_top = np.floor(half - np.clip(maxs, -1, 1) * half)
        y_bot = np.ceil(half - np.clip(mins, -1, 1) * half)
        rows = np.arange(h, dtype=np.float32)[:, None]
        lit = (rows >= y_top[None, :]) & (rows <= y_bot[None, :])

        color = MONO_COLOR if self.mono else CH_COLORS[ch]
        img = np.zeros((h, w, 4), dtype=np.uint8)
        img[lit] = (color.red(), color.green(), color.blue(), 235)
        self._blit(p, img, lane)

        p.setPen(QPen(QColor(255, 255, 255, 50), 1))
        p.drawLine(QPointF(lane.left(), lane.top() + half), QPointF(lane.right(), lane.top() + half))

    def paint_ruler(self, p, lane, ruler):
        half = (lane.height() - 1) / 2.0
        center = lane.top() + half
        marks = [(center, "-inf")]
        for db in WAVE_DB_MARKS[::-1]:  # quietest first: near-center labels win collisions
            amp = 10 ** (db / 20.0)
            marks += [(center - amp * half, str(db)), (center + amp * half, str(db))]
        p.setFont(QFont("Consolas", 7))
        p.setPen(TEXT_DIM)
        p.drawText(QPointF(ruler.left() + 7, ruler.top() + 9), "dB")
        self._draw_marks(p, marks, QRectF(ruler.left(), ruler.top() + 10, ruler.width(), ruler.height() - 10))


class SpectrogramLanes(ChannelLanes):
    """Rolling 10 s spectral 'heat' display per channel (time -> X, frequency
    -> Y on an Audition-like Hz ruler, energy -> color). Columns fade smoothly
    to black over the last few seconds before they leave the window."""

    def __init__(self, engine, on_mute_change=None):
        super().__init__(engine, on_mute_change)
        self.spec = np.full((SPEC_HISTORY, 2, SPEC_ROWS), -200.0, dtype=np.float32)

    def update_data(self, spec, mono):
        self.spec, self.mono = spec, mono
        self.update()

    def paint_lane(self, p, ch, lane):
        w, h = int(lane.width()), int(lane.height())
        if w < 2 or h < 2:
            return
        cols = np.arange(w) * len(self.spec) // w
        rows = (h - 1 - np.arange(h)) * SPEC_ROWS // h  # top row = highest frequency
        db = self.spec[cols, ch, :][:, rows].T  # (h, w)
        level = np.clip((db - SPEC_DB_FLOOR) / (SPEC_DB_CEIL - SPEC_DB_FLOOR), 0.0, 1.0)
        rgb = HEAT_LUT[(level * 255).astype(np.int32)].astype(np.float32)  # (h, w, 3)

        age = (1.0 - (np.arange(w) + 0.5) / w) * TIMELINE_SECONDS  # seconds, oldest on the left
        f = np.clip((TIMELINE_SECONDS - age) / HEAT_FADE_SECONDS, 0.0, 1.0)
        fade = f * f * (3 - 2 * f)  # smoothstep
        rgb *= fade[None, :, None]

        img = np.empty((h, w, 4), dtype=np.uint8)
        img[..., :3] = rgb.astype(np.uint8)
        img[..., 3] = 255
        self._blit(p, img, lane)

    def paint_ruler(self, p, lane, ruler):
        lo, hi = spec_scale(SPEC_FREQ_LO), spec_scale(SPEC_FREQ_HI)
        marks = []
        for f in SPEC_HZ_MARKS[::-1]:  # top-down, so 10k wins over 7k when cramped
            y = lane.bottom() - (spec_scale(f) - lo) / (hi - lo) * lane.height()
            marks.append((y, f"{f // 1000}k" if f >= 1000 else str(f)))
        p.setFont(QFont("Consolas", 7))
        p.setPen(TEXT_DIM)
        p.drawText(QPointF(ruler.left() + 7, ruler.top() + 9), "Hz")
        self._draw_marks(p, marks, QRectF(ruler.left(), ruler.top() + 10, ruler.width(), ruler.height() - 10))


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

    def __init__(self, on_change=None):
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
        self.eq_widget = ParametricEQ(self.engine, on_change=self._save_settings, band_list=self.band_list,
                                      on_hum_change=self.hum_panel.sync_from_engine)
        self.eq_widget.refresh()
        self.wave_lanes = WaveformLanes(self.engine, on_mute_change=self._on_mute_change)
        self.spec_lanes = SpectrogramLanes(self.engine, on_mute_change=self._on_mute_change)

        # EQ view + the two scrolling monitors share what used to be the EQ area
        monitors = QVBoxLayout()
        monitors.setSpacing(4)
        monitors.addWidget(self.eq_widget, stretch=5)
        monitors.addWidget(self.wave_lanes, stretch=3)
        monitors.addWidget(self.spec_lanes, stretch=4)
        eq_row.addLayout(monitors, stretch=1)

        side = QWidget()
        side.setFixedWidth(SIDE_PANEL_WIDTH)
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.addWidget(self.band_list)
        side_layout.addWidget(self.hum_panel)
        side_layout.addStretch(1)
        eq_row.addWidget(side)

        self.volume_fader = VolumeFader(on_change=self._on_volume_changed)
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
        self.flat_btn.clicked.connect(self.eq_widget.reset_flat)
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
        self.wave_lanes.update()
        self.spec_lanes.update()
        self._save_settings()

    def _save_settings(self):
        self.settings["capture_device"] = self.capture_combo.currentText()
        self.settings["monitor_device"] = self.monitor_combo.currentText()
        self.settings.pop("gains", None)  # pre-5-band layout
        self.settings.pop("freqs", None)
        self.settings["eq_bands"] = self.engine.eq.get_state()
        self.settings["dehummer"] = self.engine.dehummer.get_state()
        self.settings["mute"] = list(self.engine.mute)
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
        wmin, wmax, spec, mono = self.engine.timeline_snapshot()
        self.wave_lanes.update_data(wmin, wmax, mono)
        self.spec_lanes.update_data(spec, mono)
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
