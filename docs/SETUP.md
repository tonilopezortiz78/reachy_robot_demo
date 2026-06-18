# Setup — everything I installed and why

## 1. System packages (apt)

| Package | Why |
|---|---|
| `espeak-ng` | (was tried as TTS, replaced by Piper — kept as fallback) |
| `libcairo2-dev`, `libgirepository1.0-dev` | Build deps for `pycairo` (pulled in by the SDK via `PyGObject`) |
| `pkg-config`, `libusb-1.0-0-dev` | Build deps for SDK + motors |
| `libgstreamer1.0-dev`, `libgstreamer-plugins-base1.0-dev`, `libgstreamer-plugins-bad1.0-dev` | Headers for the GStreamer Python bindings used by the SDK's media pipeline |
| `gstreamer1.0-plugins-{base,good,bad,ugly,libav}` | Runtime GStreamer plugins for video/audio processing |
| `gstreamer1.0-tools` | `gst-launch-1.0`, `gst-inspect-1.0` |
| `python3-gst-1.0` | GObject Introspection bindings for GStreamer |
| `gir1.2-gst-plugins-base-1.0`, `gir1.2-gstreamer-1.0` | GIR typelibs needed for `gi.require_version('Gst', '1.0')` etc. |
| `ffmpeg` | Audio playback + direct camera capture (`-f v4l2 -i /dev/video2`) |

## 2. Python venv (`.venv`)

Created with `python3 -m venv .venv` to bypass PEP 668 (externally-managed system Python).

| Package | Why |
|---|---|
| `reachy-mini` (1.8.3) | Official Pollen Robotics Python SDK |
| `piper-tts` (1.4.2) | Local neural TTS (offline, sounds natural) |
| `piper` voice: `en_US-amy-medium.onnx` | One US English female voice, medium quality |
| `onnxruntime` | Runtime for Piper's neural net |
| `huggingface-hub` | Downloads the official emotion/dance move datasets |
| `bark` (install was attempted, timed out) | Expressive TTS alternative — not currently used; can be re-tried later |

## 3. Models downloaded

- `voices/en_US-amy-medium.onnx` (+ `.onnx.json`) — Piper TTS voice model
- HuggingFace dataset `pollen-robotics/reachy-mini-emotions-library` — 84 cute movement presets
- HuggingFace dataset `pollen-robotics/reachy-mini-dances-library` — 19 dance presets

## 4. Known gaps (things I did NOT install)

| Plugin | What it would unlock | Cost |
|---|---|---|
| `gst-plugins-rs` webrtcsink (Rust) | The SDK's own `media.get_frame()` and `media.play_sound()` | ~15-20 min cargo build |
| `pycairo` system headers (done) | Enables the PyGObject GIR chain to import | done |
| `Bark` larger models | "Cooler" voice with laughter, emotion, accents | ~2 GB download |

## 5. Why I used a venv

Pop!_OS / modern Debian is PEP 668 managed. `pip install reachy-mini` fails with:
```
error: externally-managed-environment
```

Workaround: create a venv (`.venv/`) and `pip install` inside it. All scripts in this repo assume `.venv/bin/python`.

## 6. Why `run.sh` is needed

The SDK's `spawn_daemon=True` does `subprocess.Popen(["reachy-mini-daemon", ...])`. That binary lives in `.venv/bin/`. To make it findable, every script either:

- is invoked through `./run.sh` (which prepends `.venv/bin` to `PATH`), or
- sets `PATH` manually before running python

If you see `FileNotFoundError: ... 'reachy-mini-daemon'`, your `PATH` doesn't include `.venv/bin/`.
