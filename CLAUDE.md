# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Control software for a **Reachy Mini Lite** (USB variant) robot from Pollen Robotics. The robot is a USB peripheral — four logical devices over one cable: motors (`/dev/ttyACM0`), speaker (ALSA card 2, `plughw:CARD=Audio,DEV=0`), camera+mic (ALSA card 1, `plughw:CARD=Camera,DEV=0` for capture; `/dev/video2` for video), and camera-audio combined as ALSA card 1. **The laptop is the computer.** The Raspberry Pi inside the robot is only a USB-to-serial bridge for the Feetech motors.

**Audio devices:**
- Speaker: `plughw:CARD=Audio,DEV=0` (card 2, Pollen Robotics Reachy Mini Audio) — playback
- **Microphone: the Pollen "Reachy Mini Audio" device input** — PipeWire source
  `alsa_input.usb-Pollen_Robotics_Reachy_Mini_Audio_<serial>-00.analog-stereo`
  (native 16 kHz, voice-optimised). This is the **only working robot mic** on this unit.

⚠️ **MIC GOTCHA (verified the hard way):** This machine has several microphones and
picking the wrong one breaks everything. Measured signal levels:
- Pollen **Audio** device input → **RMS ~880–2700, the real working voice mic. USE THIS.**
- SunplusIT **Camera** mic (`plughw:CARD=Camera,DEV=0`) → flatlined, RMS ~2 (silent on this unit,
  despite the camera nominally having a mic).
- **Laptop** built-in mic (`alsa_input.pci-...analog-stereo`) → captures ROOM NOISE, not the
  visitor. Using it made Whisper hallucinate ("con Echigua") and mis-detect the language —
  the root cause of the "spoke Japanese, replied Spanish" failure.

`reachy_demo/audio.py` auto-detects the right one at import via `_detect_robot_mic()`
(prefers `Reachy_Mini_Audio`, then `Reachy_Mini_Camera`, then laptop). To inspect the
candidates yourself: `pactl list short sources`. Capture is done with
`pacat --record --device=<source> --rate=16000 --channels=1 --format=s16le` (NOT `arecord` —
PipeWire holds the device, so direct ALSA gives "Device or resource busy").

## Running scripts

Always use `run.sh` — it prepends `.venv/bin` to `PATH`, which is required for `spawn_daemon=True` to find `reachy-mini-daemon`:

```bash
./run.sh demos/demo1_moves.py         # run a specific demo
./menu.sh                              # interactive demo picker
```

Never invoke `.venv/bin/python` directly unless you also prepend `.venv/bin` to `PATH`. If you see `FileNotFoundError: 'reachy-mini-daemon'`, the PATH is wrong.

## SDK boilerplate — required constructor arguments

Do **not** use `spawn_daemon=True` — the SDK spawns the daemon without `--no-media` (which crashes it on this machine) and doesn't wait for it to be ready before connecting.

The correct pattern is to start the daemon manually, poll until port 8000 is listening, then connect with `spawn_daemon=False`:

```python
import socket, subprocess, time
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

def start_daemon():
    proc = subprocess.Popen(["reachy-mini-daemon", "--no-media"], start_new_session=True)
    for _ in range(30):
        time.sleep(0.5)
        try:
            with socket.create_connection(("127.0.0.1", 8000), timeout=0.3):
                return proc
        except OSError:
            pass
    raise RuntimeError("Daemon did not start within 15 s")

daemon_proc = start_daemon()
try:
    with ReachyMini(connection_mode="localhost_only",
                    media_backend="no_media",
                    spawn_daemon=False) as mini:
        mini.wake_up()
        try:
            # ... moves here ...
        finally:
            mini.goto_sleep()   # ALWAYS — disables motors to prevent overheating
finally:
    daemon_proc.terminate()
    try:
        daemon_proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        daemon_proc.kill()
        daemon_proc.wait()
```

- `connection_mode="localhost_only"` — Lite has no mDNS; without this the SDK tries the network and fails.
- `media_backend="no_media"` — The GStreamer `webrtcsink` Rust plugin is not installed. Any mode that enables media will crash the daemon.
- `--no-media` flag on the daemon CLI — same reason; without it the daemon crashes on startup.
- `goto_sleep()` in `finally` — mandatory. Leaving motors energised causes them to overheat (there was a burning smell incident from an infinite loop without this).

## Motion API

Always use `create_head_pose` from `reachy_mini.utils` to build head pose matrices. The older
`set_target_head_pose(pitch=..., yaw=...)` keyword-arg style does **not** work in SDK 1.8.3 —
the method takes a numpy array, not keyword arguments.

```python
from reachy_mini.utils import create_head_pose

# Smooth interpolated move (blocks until done):
mini.goto_target(
    head=create_head_pose(pitch=0.4, degrees=False),
    antennas=[0.5, -0.5],   # [left, right] in radians
    duration=0.5,
    body_yaw=0.0,            # body rotation in radians (optional)
)

# Non-blocking instant command (10 Hz+ control loops only):
mini.set_target(
    head=create_head_pose(yaw=0.3, degrees=False),
    antennas=[0.0, 0.0],
)

# Antennas only (non-blocking):
mini.set_target_antenna_joint_positions([left_rad, right_rad])  # list, not kwargs

# Body rotation only:
mini.set_target_body_yaw(0.5)  # radians
```

`create_head_pose(pitch, yaw, roll, degrees=False)` returns a 4×4 numpy matrix. All angles
are in radians by default. Safe ranges: pitch/roll ±40°, yaw ±180°, body yaw ±160°.

| Method | When to use |
|---|---|
| `goto_target(head, antennas, duration, body_yaw)` | Short gestures; blocks until done; min-jerk interpolation |
| `set_target(head, antennas, body_yaw)` | High-frequency control loops (10 Hz+); non-blocking, no interpolation |
| `set_target_antenna_joint_positions([l, r])` | Antennas only, non-blocking |
| `set_target_body_yaw(rad)` | Body rotation only |
| `play_move(move, play_frequency=80.0, sound=False)` | Play a `RecordedMoves` preset (emotions/dances from HuggingFace) |
| `enable_gravity_compensation()` | Leave head powered but floppy — minimal current |
| `goto_sleep()` | Safe end state — relaxed pose + motors off |

## Audio — speak on the robot speaker

```python
import subprocess
subprocess.Popen(["aplay", "-D", "plughw:CARD=Audio,DEV=0", "-q", wav_path])
```

`plughw:CARD=Audio,DEV=0` goes direct to the robot's USB speaker, bypassing PipeWire/PulseAudio. Never route through PulseAudio sinks for the robot — the routing is fragile and goes to whichever device PipeWire chooses as default.

## Demo overview

| Menu | File | Voice | Features |
|------|------|-------|----------|
| 1 | demo_welcome.py | edge-tts (AvaMultilingual) | Greeting + sine-wave animation |
| 2 | demo_dance.py | edge-tts (AvaMultilingual) | Macarena show, beat-synced |
| 3 | demo_face_recognition.py | edge-tts (AvaMultilingual) | Greets visitors by name |
| 4 | demo_tools7.py | edge-tts (AvaMultilingual) | Parallel AI gesture picker, barge-in, any language |
| 5 | demo_deepseek.py | edge-tts (AvaMultilingual) | Like #4 but uses `opencode run` (DeepSeek V4 Flash) as LLM harness; STT still via Groq; ~8s latency |
| 6 | demo_instant.py | edge-tts (AvaMultilingual, streaming) | Streaming TTS — starts talking ~0.4s after LLM produces a sentence |
| 7 | demo_converse.py | edge-tts (AvaMultilingual) | Unified: instant talk + face ID + web dashboard |

**Character rules shared by all talking demos (4, 5, 6, 7):**
- 1 sentence, 10 words max — enforced in system prompt AND via `max_tokens=45`
- CRITICAL LANGUAGE RULE at top of every system prompt — robot matches user's language and switches mid-conversation
- No arms, no legs yet — Reachy acknowledges this with self-deprecating humour if asked

## demo_converse.py (menu 7 — unified)

The unified demo: instant talk + face ID + web dashboard in one process.

- **LLM:** uses Cerebras if `CEREBRAS_API_KEY` is set in `.env` (OpenAI-compatible,
  same Llama-4-scout, ~2× faster), otherwise falls back to Groq. STT is always Groq.
- **Face ID:** YuNet (detect) + SFace (recognise), Apache-2.0. Weights auto-download
  to `cache/models/` on first run. Falls back to dlib if OpenCV face modules are
  unavailable.
- **Web dashboard:** FastAPI on `http://localhost:8080` — MJPEG `/video` (live camera
  with face boxes), `/status` JSON, and `/api/wake|sleep|say|mute` controls. Frontend
  auto-reconnects.
- **Name onboarding:** when an unknown face is detected, Reachy asks the visitor's
  name, captures a few frames, and adds them to the roster live (no restart needed).
- **Speaker-lock gaze:** the head turns to track the face of whoever just spoke.

## Speech models (Groq)

All talking demos use Groq for both STT and the LLM, configured in `reachy_demo/groq_client.py`
and a `MODEL` constant in each demo:

- **STT**: `whisper-large-v3` (the full model, NOT `-turbo`). Turbo is faster but worse at
  non-English; the full model is far more accurate for the multilingual visitors Reachy meets.
  Language is auto-detected (`language=None`) so it transcribes any language spoken.
- **LLM**: `meta-llama/llama-4-scout-17b-16e-instruct` — natively multilingual and the fastest
  strong model on Groq. Replies are capped at `max_tokens=45` (one short sentence), so a bigger
  model would only add latency you'd feel as lag. Keep Scout unless replies need to get smarter
  at the cost of speed.

## TTS — edge-tts (online demos)

All online demos (5, 6, 7 and demo_dance.py) use edge-tts, configured in `reachy_demo/tts_edge.py`:

```python
VOICE = "en-US-AvaMultilingualNeural"   # single voice for ALL languages (70+ supported)
RATE, PITCH, VOL = "+30%", "+16Hz", "2.5"
```

**PITCH is `+16Hz`** — AvaMultilingual is an adult voice at 0Hz; the pitch lift is what makes
Reachy sound cute and young. Never set it to `0`. Raise toward `+24Hz` for even cuter; past
`~+28Hz` it starts sounding chipmunky. (The old setup used `en-US-AnaNeural`, a naturally cute
English-only child voice — we traded it for AvaMultilingual + pitch to gain 70+ languages.)
The multilingual voice auto-speaks in whatever language the user uses (Spanish, French,
Japanese, Arabic, etc.) without switching voice models.

TTS for offline demo (demo_talk_ns.py) uses Piper:

```python
from piper import PiperVoice
voice = PiperVoice.load("voices/en_US-amy-medium.onnx")
```

## Camera — direct UVC capture

The SDK's `media.get_frame()` requires the missing GStreamer plugin. Use direct V4L2:

```bash
ffmpeg -f v4l2 -framerate 30 -video_size 1280x720 -i /dev/video2 -frames:v 1 out.jpg
ffplay -f v4l2 -framerate 30 -video_size 1280x720 /dev/video2   # live preview
```

Or OpenCV: `cv2.VideoCapture('/dev/video2', cv2.CAP_V4L2)`.

## Cleaning up orphaned daemons

If a script is killed with `kill -9`, the daemon stays running. Before starting a new script:

```bash
pkill -9 -f "reachy-mini-daemon"
```

## What needs the missing GStreamer plugin

`gst-plugins-rs` (`webrtcsink`) is not installed (~15 min `cargo build` to add it). Without it, these don't work:
- `mini.media.get_frame()` — use direct V4L2 instead
- `mini.media.play_sound()` — use `aplay` instead
- The `official_*.py` demos in `demos/` that use media

## Preloaded move libraries (HuggingFace)

```python
from reachy_mini.motion.recorded_move import RecordedMoves
emotions = RecordedMoves("pollen-robotics/reachy-mini-emotions-library")  # 84 presets
dances   = RecordedMoves("pollen-robotics/reachy-mini-dances-library")    # 19 presets
move = emotions.get("welcoming1")
mini.play_move(move, play_frequency=80.0, sound=False)
```

## Hardware notes

- **Back switch**: must be in **Robot/Developer** position (not "Computer"). In Computer mode the SDK cannot drive the motors.
- **Green LED**: solid or slow blink = Pi is on and control stack is up.
- **Motor safety**: gestures should be ≤2 s. Continuous holding of a position causes the Feetech STS3215 servos to overheat. Always end with `goto_sleep()`.
