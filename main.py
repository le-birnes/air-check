# -*- coding: utf-8 -*-
"""Floating retro-80s 'air-check' recorder: per-device audio tap, a 3D waveform /
spectral scope with the 5-band EQ on its YZ plane, and a DeHummer."""
import ctypes
import struct
import sys
from pathlib import Path

import numpy as np
from OpenGL import GL
from OpenGL.GL import shaders as gl_shaders

from PySide6.QtCore import Qt, QTimer, QPointF, QRectF, QDataStream, QByteArray, QUrl
from PySide6.QtGui import (
    QPainter, QColor, QLinearGradient, QPen, QFont, QBrush, QPainterPath, QRadialGradient,
    QIcon, QPixmap, QAction, QSurfaceFormat, QDesktopServices,
)
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QSystemTrayIcon, QMenu, QSlider, QGridLayout, QCheckBox, QDoubleSpinBox, QInputDialog,
)

sys.path.insert(0, str(Path(__file__).parent))
from engine import (
    AudioEngine, list_capture_devices, list_monitor_devices, peak_decimate, TIMELINE_SECONDS,
    SCOPE_POINTS, SCOPE_WINDOW, SPEC_ROWS, SPEC_FREQ_LO, SPEC_FREQ_HI,
    SCOPE_HOPS, SCOPE_HOP_DEFAULT, SAMPLE_RATE,
)
from eq import (
    BAND_DEFS, DEHUMMER_PRESETS, HUM_FREQ_MIN, HUM_FREQ_MAX, HUM_Q_MIN,
    HUM_Q_MAX, HUM_GAIN_MIN, HUM_GAIN_MAX, HUM_MAX_HARMONICS,
)
import settings as settings_store

OUTPUT_DIR = Path(__file__).parent / "recordings"

# --- matrix-red palette ------------------------------------------------------
BG_TOP = QColor(10, 0, 0)
BG_BOTTOM = QColor(0, 0, 0)
PANEL_BORDER = QColor(255, 25, 25)
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
SIDE_PANEL_WIDTH = 250


def fmt_freq(freq):
    return f"{freq/1000:.2f}k" if freq >= 1000 else f"{freq:.0f}"


Q_WHEEL_STEP = 1.15  # Q multiplier per mouse-wheel notch


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
SCOPE_DEPTH = 1.5         # world units the 10 s history spans on each side of the live trace
SCOPE_Y = 1.6             # half-length of the Y axis (the live trace / frequency axis)
SCOPE_WAVE_Z = 0.55       # waveform full-scale height
SCOPE_SPEC_Z = 0.8        # spectrum full height
SCOPE_EQ_Z = 0.6          # EQ curve on the YZ plane: +-DB_RANGE maps to +-SCOPE_EQ_Z
SCOPE_DB_FLOOR, SCOPE_DB_CEIL = -90.0, -10.0
SCOPE_HZ_MARKS = (50, 100, 200, 500, 1000, 2000, 5000, 10000)
SCOPE_DEFAULT_VIEW = (16.0, 52.0, 1.0)  # yaw deg, pitch deg, zoom
SCOPE_PITCH_MAX = 90.0    # orbit can tilt the floor all the way to a top-down view
EQ_RGB = (255, 205, 60)
EQ_NODE_HALF_W, EQ_NODE_HALF_H = 17, 11  # node tag half-size, px (gain line + Q line)
EQ_HIT_RADIUS = 16
SCOPE_CAMERA = 5.0
# terrain color by height, bottom -> top: near-black, purple, red, yellow, near-white (never pure black/white)
TERRAIN_STOPS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
TERRAIN_RGB = np.array([[8, 4, 14], [58, 18, 108], [150, 24, 46], [205, 135, 24], [232, 222, 188]])
TERRAIN_GAMMA = 1.6        # > 1 keeps most of the terrain in the dark end; only real peaks get bright
TERRAIN_STEP_SECONDS = 0.05  # terrain shape: one (averaged, softened) row per 50 ms; colors stay full-rate
# waveform history: shades of blue, deep navy -> icy near-white
WAVE_TERRAIN_RGB = np.array([[4, 6, 20], [12, 28, 84], [28, 68, 158], [78, 138, 218], [188, 212, 244]])
TERRAIN_BINS = 24          # discrete height colors
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


def terrain_rows(values, first_index, step):
    """Coarse terrain rows from a full-rate history: the frames averaged in
    groups of `step`, the groups aligned to the absolute frame index (so a
    group keeps its frames as the history scrolls and the hills don't jitter),
    then softened along time with two [1, 2, 1] passes. -> (frame positions
    (r,), heights (r, n)); the first and last frame are always rows, so the
    surface spans the whole history."""
    k = len(values)
    starts = np.flatnonzero((first_index + np.arange(k)) % step == 0)
    if not len(starts) or starts[0] != 0:
        starts = np.concatenate([[0], starts])
    counts = np.diff(np.append(starts, k))
    means = np.add.reduceat(values, starts, axis=0) / counts[:, None]
    for _ in range(2):
        padded = np.concatenate([means[:1], means, means[-1:]])
        means = (padded[:-2] + 2 * padded[1:-1] + padded[2:]) / 4
    pos = starts + (counts - 1) / 2.0
    pos = np.concatenate([[0.0], pos, [k - 1.0]])
    means = np.concatenate([means[:1], means, means[-1:]])
    return pos.astype(np.float32), means.astype(np.float32)


_TERRAIN_VS = """
#version 330
layout(location = 0) in float shape;  // height value (time-averaged): the slow topography
layout(location = 1) in float fpos;   // this row's position in history frames (0 = oldest)
uniform vec2 yaw;      // cos, sin
uniform vec2 pitch;    // cos, sin
uniform vec3 cam;      // scale, cx, cy (logical px)
uniform vec2 size;     // widget size (logical px)
uniform float camera;  // SCOPE_CAMERA
uniform int cols;      // values per frame (along Y)
uniform vec3 grid;     // x of frame 0, x step per frame, Y half-length
uniform int mode;      // 0 waveform peak, 1 spectrum dB
uniform vec4 zmap;     // mode 0: (z scale); mode 1: (dB floor, dB ceil, z scale)
out vec2 v_cell;       // (history frame, Y column) for the fragment's color lookup
float to_z(float v) {
    return mode == 0 ? clamp(v, -1.0, 1.0) * zmap.x
                     : pow(clamp((v - zmap.x) / (zmap.y - zmap.x), 0.0, 1.0), 1.6) * zmap.z;
}
void main() {
    int col = gl_VertexID % cols;
    vec3 pos = vec3(grid.x + grid.y * fpos, grid.z * (2.0 * float(col) / float(cols - 1) - 1.0), to_z(shape));
    // exactly Scope3D._project, so the QPainter overlays line up
    float depth = pos.x * yaw.x - pos.y * yaw.y;
    float right = pos.x * yaw.y + pos.y * yaw.x;
    float up = pos.z * pitch.x + depth * pitch.y;
    float toward = -depth * pitch.x + pos.z * pitch.y;
    float f = camera / (camera - toward);
    float sx = cam.y + cam.x * right * f;
    float sy = cam.z - cam.x * up * f;
    gl_Position = vec4(2.0 * sx / size.x - 1.0, 1.0 - 2.0 * sy / size.y, -toward / camera, 1.0);
    v_cell = vec2(fpos, float(col));
}
"""

_TERRAIN_FS = """
#version 330
in vec2 v_cell;
uniform sampler2D values;  // full-rate raw history: width = Y columns, height = frames
uniform int mode;
uniform vec4 zmap;
uniform vec2 zrange;       // z bottom, top -> palette
uniform vec3 palette[5];
uniform float bins;
uniform float gamma;
out vec4 color;
float to_z(float v) {
    return mode == 0 ? clamp(v, -1.0, 1.0) * zmap.x
                     : pow(clamp((v - zmap.x) / (zmap.y - zmap.x), 0.0, 1.0), 1.6) * zmap.z;
}
void main() {
    // real-time coloring: every pixel reads its own frame, however coarse the shape is
    ivec2 size = textureSize(values, 0);
    ivec2 texel = clamp(ivec2(int(v_cell.y + 0.5), int(v_cell.x + 0.5)), ivec2(0), size - 1);
    float height = (to_z(texelFetch(values, texel, 0).r) - zrange.x) / (zrange.y - zrange.x);
    // discrete height bands along the palette, never its pure ends
    float h = pow(clamp(height, 0.0, 0.9999), gamma);
    float q = (floor(h * bins) + 0.5) / bins * 4.0;
    int i = int(min(floor(q), 3.0));
    color = vec4(mix(palette[i], palette[i + 1], q - float(i)), 1.0);
}
"""


class TerrainGL:
    """GPU half of Scope3D. Two rates on one surface: the shape is a coarse
    mesh (one row per TERRAIN_STEP_SECONDS of history, averaged and softened,
    so it rolls into hills and valleys); the color comes from the full-rate
    raw history, uploaded as a texture and read per pixel. Depth-tested, so
    nearer terrain hides farther terrain per pixel. Lives in the widget's GL
    context; call only from paintGL (inside native painting)."""

    def __init__(self):
        self.vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.vao)
        self.program = gl_shaders.compileProgram(
            gl_shaders.compileShader(_TERRAIN_VS, GL.GL_VERTEX_SHADER),
            gl_shaders.compileShader(_TERRAIN_FS, GL.GL_FRAGMENT_SHADER), validate=False)
        self.vbo, self.ibo = GL.glGenBuffers(2)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.ibo)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 1, GL.GL_FLOAT, False, 8, ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 1, GL.GL_FLOAT, False, 8, ctypes.c_void_p(4))
        GL.glBindVertexArray(0)
        self.tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.tex)
        for p in (GL.GL_TEXTURE_MIN_FILTER, GL.GL_TEXTURE_MAG_FILTER):
            GL.glTexParameteri(GL.GL_TEXTURE_2D, p, GL.GL_NEAREST)
        for p in (GL.GL_TEXTURE_WRAP_S, GL.GL_TEXTURE_WRAP_T):
            GL.glTexParameteri(GL.GL_TEXTURE_2D, p, GL.GL_CLAMP_TO_EDGE)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        self.loc = {name: GL.glGetUniformLocation(self.program, name)
                    for name in ("yaw", "pitch", "cam", "size", "camera", "palette", "bins", "gamma",
                                 "cols", "grid", "mode", "zmap", "zrange", "values")}
        self._grid = None  # (rows, n) the index buffer was built for

    def begin(self, view, scissor, viewport):
        """view: yaw deg, pitch deg, (scale, cx, cy), (w, h) logical px.
        scissor / viewport: (x, y, w, h) in framebuffer pixels."""
        yaw, pitch, cam, size = view
        GL.glViewport(*viewport)
        GL.glEnable(GL.GL_SCISSOR_TEST)
        GL.glScissor(*scissor)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LESS)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glDisable(GL.GL_BLEND)
        GL.glDisable(GL.GL_CULL_FACE)
        GL.glClear(GL.GL_DEPTH_BUFFER_BIT)
        GL.glUseProgram(self.program)
        GL.glUniform2f(self.loc["yaw"], np.cos(np.radians(yaw)), np.sin(np.radians(yaw)))
        GL.glUniform2f(self.loc["pitch"], np.cos(np.radians(pitch)), np.sin(np.radians(pitch)))
        GL.glUniform3f(self.loc["cam"], *cam)
        GL.glUniform2f(self.loc["size"], *size)
        GL.glUniform1f(self.loc["camera"], SCOPE_CAMERA)
        GL.glUniform1f(self.loc["bins"], TERRAIN_BINS)
        GL.glUniform1f(self.loc["gamma"], TERRAIN_GAMMA)
        GL.glUniform1i(self.loc["values"], 0)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.tex)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 4)
        GL.glBindVertexArray(self.vao)

    def draw(self, values, first_index, step, x0, dx, mode, zmap, z_range, palette):
        """values: (k, n) raw history, frames oldest -> newest and evenly
        spaced, frame r at x0 + r * dx; first_index: absolute index of frame 0;
        step: frames per terrain row. mode 0: waveform peaks, z = value * zmap[0];
        mode 1: spectrum dB, zmap = (floor, ceil, z scale). z_range (bottom,
        top) maps onto palette ((5, 3) RGB 0..255)."""
        k, n = values.shape
        if k < 2:
            return
        values = np.ascontiguousarray(values, dtype=np.float32)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_R32F, n, k, 0, GL.GL_RED, GL.GL_FLOAT, values)
        pos, shapes = terrain_rows(values, first_index, step)
        r = len(pos)
        GL.glUniform3fv(self.loc["palette"], 5, (palette / 255.0).astype(np.float32))
        GL.glUniform1i(self.loc["cols"], n)
        GL.glUniform3f(self.loc["grid"], x0, dx, SCOPE_Y)
        GL.glUniform1i(self.loc["mode"], mode)
        GL.glUniform4f(self.loc["zmap"], *(list(zmap) + [0.0] * (4 - len(zmap))))
        GL.glUniform2f(self.loc["zrange"], *z_range)
        data = np.empty((r, n, 2), dtype=np.float32)  # interleaved (shape, row position)
        data[..., 0] = shapes
        data[..., 1] = pos[:, None]
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, data.nbytes, data, GL.GL_STREAM_DRAW)
        rows, cols = self._grid or (0, 0)
        if cols != n or rows < r:
            # two triangles per cell between rows j and j + 1. Built for the most rows
            # seen: fewer rows use a prefix of it, so it isn't rebuilt every paint.
            rows = max(r, rows if cols == n else 0)
            v = np.arange(rows * n, dtype=np.uint32).reshape(rows, n)
            a, b, c, d = v[:-1, :-1], v[:-1, 1:], v[1:, 1:], v[1:, :-1]
            idx = np.stack([a, b, c, a, c, d], axis=-1).ravel()
            GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, idx.nbytes, idx, GL.GL_STATIC_DRAW)
            self._grid = (rows, n)
        GL.glDrawElements(GL.GL_TRIANGLES, (r - 1) * (n - 1) * 6, GL.GL_UNSIGNED_INT, ctypes.c_void_p(0))

    @staticmethod
    def end():
        GL.glBindVertexArray(0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glUseProgram(0)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glDisable(GL.GL_SCISSOR_TEST)


class Scope3D(QOpenGLWidget):
    """3D audio scope. Y axis: the live waveform. X+: the last 10 s of waveform
    (max envelope) receding behind it. X-: the matching spectrum (log Hz along
    Y) coming forward. Both are opaque terrains drawn on the GPU with a depth
    buffer, colored in height bands: blues for the waveform, near-black to
    near-white heat for the spectrum. Axes, live trace and EQ are QPainter
    overlays on top.

    MIX/CH toggle: one scene for the mix, or L | R side by side. The rate
    label next to it cycles the history frame hop (1 ms .. 43 ms). Channel mute
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
        self.wave, self.spec = {}, {}  # channel -> (F, SCOPE_POINTS) max envelope / (F, SPEC_ROWS) dB
        self.ages = np.zeros(0)
        self.first_index = 0  # absolute index of the first history frame
        self.yaw, self.pitch, self.zoom = SCOPE_DEFAULT_VIEW
        self._orbit = None
        self._gl = None  # TerrainGL, created with the GL context
        fmt = QSurfaceFormat()
        fmt.setDepthBufferSize(24)
        fmt.setSamples(4)  # MSAA: smooth terrain edges on the GPU
        self.setFormat(fmt)
        self.setMinimumHeight(220)

    def initializeGL(self):
        self._gl = TerrainGL()

    def update_data(self, live, wave, spec, ages, mono, channels, first_index):
        self.live, self.ages, self.mono, self.first_index = live, ages, mono, first_index
        self.wave = {c: wave[:, i] for i, c in enumerate(channels)}
        self.spec = {c: spec[:, i] for i, c in enumerate(channels)}
        self.update()

    def channels_needed(self):
        """Data channels the current scenes draw (0 L, 1 R, 2 mix)."""
        return tuple(idx for _, idx, _ in self._layout()[1]) or (2,)

    # ---- layout -----------------------------------------------------------------
    @staticmethod
    def _toggle_rects():
        return QRectF(6, 4, 34, 15), QRectF(40, 4, 34, 15)

    @staticmethod
    def _rate_rect():
        return QRectF(80, 4, 44, 15)

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
    def paintGL(self):
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

        # X+ waveform / X- spectrum history as opaque GPU terrains (depth-tested, so no draw order)
        valid = self.ages < TIMELINE_SECONDS
        if self._gl is not None and valid.sum() > 1 and idx in self.wave:
            p.beginNativePainting()
            self._gl.begin((self.yaw, self.pitch, cam, (self.width(), self.height())),
                           self._device_rect(rect), self._device_rect(QRectF(self.rect())))
            self._gl.draw(*self._wave_terrain(idx, valid))
            self._gl.draw(*self._spec_terrain(idx, valid))
            self._gl.end()
            p.endNativePainting()

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

        hw, hh = EQ_NODE_HALF_W, EQ_NODE_HALF_H
        for i, band in enumerate(self.engine.eq.bands):
            y, z = self._band_yz(band)
            x, yy = (float(v) for v in proj(0.0, y, z))
            color = QColor(NODE_COLORS[i % len(NODE_COLORS)])
            off = band.is_off() and band.kind != "peak"
            if off:
                color.setAlpha(110)
            # the node is a little two-line tag: gain (or HP/LP) on top, Q below
            box = QRectF(x - hw, yy - hh, hw * 2, hh * 2)
            p.setPen(QPen(QColor(10, 0, 0), 1.2))
            p.setBrush(color)
            p.drawRect(box)
            p.setPen(QColor(10, 0, 0))
            p.setFont(QFont("Consolas", 7, QFont.Bold))
            top = f"{band.gain_db:+.0f}dB" if band.kind == "peak" else band.name
            p.drawText(QRectF(box.left(), box.top() + 1, box.width(), hh - 1), Qt.AlignCenter, top)
            p.drawText(QRectF(box.left(), yy, box.width(), hh - 1), Qt.AlignCenter,
                       "off" if off else f"Q{band.q:.2f}")
            p.setFont(QFont("Consolas", 8, QFont.Bold))
            label = "OFF" if off else f"{fmt_freq(band.freq)}Hz"
            label_rect = QRectF(x - 30, yy - hh - 15, 60, 13)
            p.setPen(QColor(0, 0, 0, 210))
            p.drawText(label_rect.translated(0, 1), Qt.AlignCenter, label)
            p.setPen(TEXT_BRIGHT)
            p.drawText(label_rect, Qt.AlignCenter, label)

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

    def _device_rect(self, rect):
        """Logical widget rect -> GL (x, y, w, h) in framebuffer pixels, y up."""
        dpr = self.devicePixelRatioF()
        return (int(rect.left() * dpr), int((self.height() - rect.bottom()) * dpr),
                int(rect.width() * dpr), int(rect.height() * dpr))

    def _terrain_layout(self, valid):
        """(absolute index of the first drawn frame, frames per terrain row):
        the shape is drawn at TERRAIN_STEP_SECONDS whatever the history rate."""
        step = max(1, int(round(TERRAIN_STEP_SECONDS * SAMPLE_RATE / self.engine.scope_hop)))
        return self.first_index + int(np.argmax(valid)), step

    def _frame_x(self, valid, sign, offset):
        """First frame's X and the step between frames: frames are evenly
        spaced (one per hop), so X is linear in the frame index."""
        x = sign * (self.ages[valid] / TIMELINE_SECONDS * SCOPE_DEPTH + offset)
        return float(x[0]), float((x[-1] - x[0]) / (len(x) - 1))

    def _wave_terrain(self, idx, valid):
        """X+: the waveform's max envelope receding behind the live trace. Only
        the top surface: whatever lies under it is hidden anyway."""
        x0, dx = self._frame_x(valid, 1, 0.02)
        return (self.wave[idx][valid], *self._terrain_layout(valid), x0, dx, 0, (SCOPE_WAVE_Z,),
                (0.0, SCOPE_WAVE_Z), WAVE_TERRAIN_RGB)

    def _spec_terrain(self, idx, valid):
        """X-: the spectrum (log Hz along Y) coming forward."""
        x0, dx = self._frame_x(valid, -1, 0.04)
        # clamped at the display floor first, so silence (-200 dB) doesn't drag the averaged heights down
        values = np.maximum(self.spec[idx][valid], SCOPE_DB_FLOOR)
        return (values, *self._terrain_layout(valid), x0, dx, 1, (SCOPE_DB_FLOOR, SCOPE_DB_CEIL, SCOPE_SPEC_Z),
                (0.0, SCOPE_SPEC_Z), TERRAIN_RGB)

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
        rect = self._rate_rect()  # history frame hop: click cycles
        p.setPen(QPen(QColor(70, 60, 90), 1))
        p.setBrush(QColor(12, 10, 20))
        p.drawRect(rect)
        p.setPen(QColor(235, 235, 255))
        p.drawText(rect, Qt.AlignCenter, f"{self.engine.scope_hop * 1000 / SAMPLE_RATE:.0f} ms")

    # ---- mouse ------------------------------------------------------------------
    def mousePressEvent(self, event):
        pos = event.position()
        if self._rate_rect().contains(pos):
            hops = list(SCOPE_HOPS)
            cur = self.engine.scope_hop
            self.engine.set_scope_hop(hops[(hops.index(cur) + 1) % len(hops)] if cur in hops else SCOPE_HOP_DEFAULT)
            if self.on_mode_change:
                self.on_mode_change()
            self.update()
            return
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
                                       on_change=self._save_settings)
        self.band_list.refresh(self.engine.eq.bands)
        self.engine.set_scope_hop(self.settings.get("scope_hop", SCOPE_HOP_DEFAULT))
        self.scope = Scope3D(self.engine, mode=self.settings.get("scope_mode", "mix"),
                             on_mute_change=self._on_mute_change, on_mode_change=self._save_settings,
                             on_eq_change=lambda: self.band_list.refresh(self.engine.eq.bands),
                             on_eq_commit=self._save_settings)

        eq_row.addWidget(self.scope, stretch=1)  # the 3D scope owns the whole monitor area

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
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(OUTPUT_DIR)))  # Explorer / the Linux file manager

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
        self.settings["scope_hop"] = self.engine.scope_hop
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
        db_l, db_r = self.engine.snapshot()[:2]
        self.vu.set_levels(db_l, db_r)
        self.scope.update_data(*self.engine.scope_snapshot(self.scope.channels_needed()))
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
