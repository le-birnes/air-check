# -*- coding: utf-8 -*-
"""Small persisted settings: last device choices, EQ curve, auto-power-on."""
import json
from pathlib import Path

SETTINGS_PATH = Path(__file__).parent / "settings.json"

DEFAULTS = {
    "capture_device": None,
    "monitor_device": None,
    "eq_bands": None,  # None -> eq.BAND_DEFS defaults
    "dehummer": None,  # None -> DeHummer defaults (off)
    "dehummer_presets": {},  # user-saved DeHummer presets, name -> params
    "mute": [False, False],
    "auto_power_on": True,
    "master_volume": 100,
    "scope_hop": 2048,  # scope history frame hop in samples (engine.SCOPE_HOPS)
}


def load():
    if SETTINGS_PATH.exists():
        try:
            data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            merged = dict(DEFAULTS)
            merged.update(data)
            return merged
        except Exception:
            pass
    return dict(DEFAULTS)


def save(data):
    try:
        SETTINGS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass
