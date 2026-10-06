# Air-Check

A desktop "air-check" recorder: it taps whatever your computer is playing
(a loopback of any output device), runs it through a DeHummer and a 5-band
parametric EQ, lets you listen to the processed signal live on another output,
and records it to WAV. A GPU-drawn 3D scope shows the last 10 seconds of
waveform and spectrum as two terrains, with the EQ curve standing on the plane
between them.

Built with Python, PySide6 (Qt) and OpenGL. Developed on Windows 11; the
Arch Linux instructions below are written from the libraries' Linux support
and have not been tested on Arch yet.

## Features

- **Loopback capture** of any playback device (WASAPI loopback on Windows,
  PulseAudio/PipeWire monitor sources on Linux), 48 kHz stereo.
- **Live monitor** of the processed signal on any output device, with a
  master fader (monitor only: it never changes the recording).
- **Recording** to `recordings/recording_YYYYMMDD_HHMMSS.wav`
  (48 kHz, stereo, 32-bit float), with pause/resume.
- **DeHummer**: notch filters on a hum fundamental and its harmonics, with
  50 Hz / 60 Hz presets and user-saved presets.
- **5-band EQ** (high-pass, three peaking bands, low-pass) with per-band Q,
  editable directly on the scope.
- **Per-channel mute** (applies to monitor and recording); mono input is
  detected and shown as one scene.
- **3D scope** (OpenGL, depth-tested):
  - Y axis: the live waveform trace.
  - X+: the last 10 s of waveform (max envelope) as a terrain receding back,
    in shades of blue.
  - X-: the last 10 s of spectrum (log frequency along Y) coming forward,
    in dark heat bands (near-black, purple, red, amber, pale cream).
  - Two rates on one surface: the *shape* is drawn at 50 ms (averaged and
    softened, so it rolls into hills and valleys), while the *color* is the
    full-rate history read per pixel, so short clicks still show.
  - History rate selectable live: 1, 5, 10, 21 (one audio block) or 43 ms
    per frame.
  - Opaque, depth-tested surfaces; nothing below the top surface is drawn.
  - MIX view or L | R side by side.
- **System tray** app: closing the window keeps it running (and recording);
  quit from the tray menu. `--start-hidden` starts it in the tray.
- Settings (devices, EQ, DeHummer, mutes, view, volume) persist in
  `settings.json` next to the app.

### Scope controls

| Where            | Action        | Effect                    |
|------------------|---------------|---------------------------|
| EQ node          | drag          | move frequency / gain     |
| EQ node          | mouse wheel   | change Q                  |
| EQ node          | double-click  | reset that band           |
| elsewhere        | drag          | orbit the view            |
| elsewhere        | mouse wheel   | zoom                      |
| elsewhere        | double-click  | reset the view            |
| MIX / CH toggle  | click         | one mix scene or L \| R   |
| rate label (ms)  | click         | cycle the history rate    |
| L / R buttons    | click         | mute a channel            |

## Requirements

- Python 3.11+ (developed on 3.13)
- A GPU / driver with OpenGL 3.3
- Python packages (`requirements.txt`):

| Package     | Used for                                         | Developed with |
|-------------|--------------------------------------------------|----------------|
| PySide6     | the UI (Qt widgets, tray, OpenGL widget)         | 6.11.1         |
| PyOpenGL    | drawing the scope terrains on the GPU            | 3.1.10         |
| numpy       | all signal and geometry math                     | 2.3.3          |
| scipy       | IIR filtering for the EQ and DeHummer (`lfilter`) | 1.16.2         |
| SoundCard   | loopback capture of an output device             | 0.4.6          |
| sounddevice | monitor output (PortAudio)                       | 0.5.2          |
| soundfile   | writing the WAV recordings (libsndfile)          | 0.13.1         |

## Running on Windows

1. Install Python 3.11+ from python.org (tick "Add python.exe to PATH").
2. In the project folder:

   ```bat
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   python main.py
   ```

3. To run it without a console window, in the tray:

   ```bat
   .venv\Scripts\pythonw.exe main.py --start-hidden
   ```

   For autostart, put a shortcut to that command in
   `shell:startup` (Win+R, `shell:startup`).

The SoundCard, sounddevice and soundfile wheels bundle their native
libraries on Windows; nothing else is needed. **SOURCE** lists every
playback device as a loopback source. To capture a single app, route it to
a virtual cable (e.g. VB-Audio Cable) and pick "CABLE Output" as the source;
it's preferred automatically when present.

## Running on Arch Linux

1. System packages (Qt, the OpenGL bindings, numpy/scipy and the native
   audio libraries):

   ```sh
   sudo pacman -S --needed python python-pip pyside6 python-opengl \
       python-numpy python-scipy portaudio libsndfile pipewire-pulse
   ```

   `pipewire-pulse` (or `pulseaudio`) provides the PulseAudio API that
   SoundCard uses for loopback capture. You also need a working OpenGL
   driver (`mesa`, or `nvidia-utils` on NVIDIA).

2. A virtual environment that reuses those system packages, plus the audio
   bindings that are easiest from pip:

   ```sh
   python -m venv --system-site-packages .venv
   source .venv/bin/activate
   pip install SoundCard sounddevice soundfile
   python main.py
   ```

   (Or skip the system Python packages and `pip install -r requirements.txt`
   inside a plain venv; you still need `portaudio`, `libsndfile` and a
   PulseAudio-compatible server from pacman.)

3. In **SOURCE**, the loopback sources are the "Monitor of ..." devices of
   your outputs. The tray icon needs a desktop with a system tray
   (KDE, XFCE, or GNOME with the AppIndicator extension).

## Project layout

| File          | What it holds                                                     |
|---------------|-------------------------------------------------------------------|
| `main.py`     | the Qt app: window, tray, side panels, 3D scope (QPainter + OpenGL) |
| `engine.py`   | capture thread, DSP chain, monitor output, WAV writer, scope history |
| `eq.py`       | 5-band EQ and DeHummer filters                                    |
| `settings.py` | load/save of `settings.json`                                      |

## Development history

Reconstructed from the local prototype scripts and the git log.

**July 20, 2026: capture prototypes.** Probe scripts tried per-process audio
capture (the `process_audio_capture` library: API survey, a self-capture
isolation test) and a VB-Audio virtual cable. They then tested classic
device-level WASAPI loopback through SoundCard while music played, which
became the capture method. The first engine (loopback capture -> monitor
output) and monitor-device checks followed the same day.

**July 21, 2026: first UI passes.** Screenshot-driven visual checks across
window widths, a paint-cost and audio-dropout check, and a fix for the then
graphic EQ's band ordering (node drags jumping to unrelated bands).

**October 5, 2026** (git):
- DeHummer, 5-band Q EQ, scrolling waveform and spectral monitors.
- A 3D waveform/spectral scope replaces the 2D timeline lanes.
- A tilted floor, a denser scope, EQ nodes on the YZ plane, and an FX sidebar toggle.
- The scope takes over the whole monitor area; spectrum terrain as a heat map.

**October 6, 2026** (git):
- The history is drawn as filled solids instead of a wire mesh. History frames
  were averaged into 0.3 s slices to keep it fast.
- A GPU rewrite: the terrains became depth-tested OpenGL meshes, i.e. one
  continuous surface instead of slices drawn back to front, at about 3 ms of
  CPU per frame. Frames come every ~43 ms with no averaging, so clicks show.
  The open-folder button became cross-platform, and this README was added.
- The history rate became a live setting (1 to 43 ms). The engine computes
  several frames per audio block, and the GPU works out geometry and color
  from the raw values. The waveform got its own blue palette, and both
  palettes got darker.
- Two-rate terrain: a coarse 50 ms mesh carries the shape (rolling hills),
  and a full-rate texture carries the color. A 1 ms history draws in about
  13 ms.
