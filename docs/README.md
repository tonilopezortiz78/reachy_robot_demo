# Reachy Mini — Project Notes

Everything I learned while setting up and using a **Reachy Mini Lite** (USB variant) from a Pop!_OS / Ubuntu laptop.

## TL;DR

- The robot you have is the **Lite** (USB). The Pi inside it is **not** the computer — it is a USB-to-serial bridge for the motors. **Your laptop is the computer** that runs everything.
- The robot is connected to your laptop over a single USB cable that exposes **three independent devices**:
  - motors (serial at `/dev/ttyACM0`)
  - speaker + mic (USB audio, card 2)
  - camera (UVC, `/dev/video2`)
- All motors, audio, and camera work **without** any extra plugin installs, using the Pollen Robotics SDK + plain `aplay` + `ffmpeg -f v4l2`.
- The "back switch" is a physical toggle between **Robot mode** (the SDK can drive the motors) and **Computer mode** (the Pi outputs to a display, used for first-time setup). Keep it on Robot for everything in this repo.
- The green LED = Pi is powered. Solid or slow blink = good. Off = no power.

## Repo layout

```
reachy/
├── .venv/                       Python venv (reachy-mini, piper-tts, etc.)
├── voices/                      Piper TTS voice models
├── demos/                       Runnable demos (see docs/RUN_DEMOS.md)
├── docs/                        This documentation
├── run.sh                       One-line wrapper: sets PATH and runs python
├── menu.sh                      Interactive menu to pick a demo
├── AGENTS.md                    Notes for AI agents
└── *.py                         Standalone scripts
```

## Quick start

```bash
cd /home/tony/software/robots/reachy
./menu.sh
# or run a specific demo:
./run.sh demos/demo1_moves.py
```

## Documents in this folder

| File | What's in it |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | What the robot actually is, what each USB endpoint does, who runs what |
| [SETUP.md](SETUP.md) | Every system package + Python package I installed and why |
| [HARDWARE_DIAGNOSIS.md](HARDWARE_DIAGNOSIS.md) | How I figured out the camera is direct USB, the back switch, the green light |
| [SDK_NOTES.md](SDK_NOTES.md) | SDK gotchas: spawn_daemon, no_media, why some examples fail on the laptop |
| [AUDIO_PIPELINE.md](AUDIO_PIPELINE.md) | Why my first attempt played through the laptop speakers, the ALSA fix |
| [CAMERA_PIPELINE.md](CAMERA_PIPELINE.md) | Direct UVC capture without the GStreamer webrtcsink plugin |
| [RUN_DEMOS.md](RUN_DEMOS.md) | Each demo, what it does, the expected runtime |
| [SAFETY.md](SAFETY.md) | Motor heat, why the smell happened, safe movement patterns |
| [MOVE_PLAN.md](MOVE_PLAN.md) | The one tiny move I want to run next, for your approval |
