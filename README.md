# Reachy Mini — Network School

**AI robot demos for [Network School](https://ns.com) built on the Reachy Mini Lite by Pollen Robotics.**

Created by **Antonio Lopez Ortiz**

---

## What is this?

This repo contains two polished, ready-to-run demos for the Reachy Mini Lite (USB variant). The robot greets visitors, speaks with a robot-processed voice, animates with natural layered motion across all axes, and dances to music.

The robot connects over a single USB-C cable — no WiFi, no network, no extra setup beyond the one-time steps in [`docs/SETUP.md`](docs/SETUP.md).

---

## Quick start

```bash
# Interactive menu (recommended)
./menu.sh

# Or run directly
./run.sh demos/demo_welcome.py    # greeting demo
./run.sh demos/demo_dance.py      # full show with music
```

> **Before running:** make sure the back switch on the robot is in the **Robot / Developer** position (not "Computer"). The green LED should be solid or slow-blinking.

---

## Demo 1 — Welcome to Network School (`demo_welcome.py`)

**What it does:**
1. Three rising **record-cue beeps** — your signal to hit record
2. R2-D2-style **boot beeps** as the robot wakes up
3. Robot speaks immediately on wake-up — *"Welcome to Network School! What would you like to talk about? Robotics, Artificial Intelligence, Crypto, or Network States?"*
4. All axes animate simultaneously during speech: head pitch/yaw/roll, antennas, body rotation — layered sine waves at incommensurable frequencies so motion never looks mechanical
5. Closes with an **attentive listening pose**

**Runtime:** ~25 seconds

---

## Demo 2 — Full Dance Show (`demo_dance.py`)

**What it does:**
1. Record-cue beeps + boot sequence
2. **Greeting speech** with full-body animation (body rotates ±51° during speech)
3. *"And now… watch this!"* transition line
4. **Music starts** (Blipotron — Kevin MacLeod, CC-BY 3.0)
5. Curated **dance sequence** (~60 s):
   - Warm-up → groovy sway section (body spun to +69°)
   - Robot / mechanical section (body snapped to -69°, then +80°)
   - Spiral / complex section (full ±80° sweeps)
   - Climax: `dance3` preset (18 s energetic full-body dance)
   - Victory celebration
6. **Bow out** (`loving1` emotion preset)
7. Robot goes to sleep

**Runtime:** ~90 seconds

### Swapping the music

Drop any MP3 or WAV into `music/` and edit the one line at the top of `demo_dance.py`:

```python
MUSIC = str(ROOT / "music" / "your_track.mp3")
```

Two tracks are included:
- `music/blipotron.mp3` — driving hard electronic (default)
- `music/kick_shock.mp3` — harder alternative

Both by Kevin MacLeod (incompetech.com) — CC-BY 3.0.

---

## Hardware

| Component | Detail |
|---|---|
| Robot | Reachy Mini **Lite** (USB variant) |
| Connection | Single USB-C cable — motors + speaker + camera |
| Motors | `/dev/ttyACM0` (Feetech STS3215 servos) |
| Speaker | ALSA card 2 — `plughw:2,0` |
| Camera | `/dev/video2` (UVC, 1080p) |
| Computer | Your laptop (the Pi inside the robot is only a USB bridge) |

---

## Project layout

```
reachy/
├── run.sh              entry point — always use this to run demos
├── menu.sh             interactive demo picker
├── demos/
│   ├── demo_welcome.py greeting + speech demo
│   └── demo_dance.py   full show with Macarena choreography
├── music/              background tracks (CC-BY included; drop your own MP3 here)
├── voices/             Piper TTS config (model downloaded separately — see below)
└── docs/               technical documentation
    ├── ARCHITECTURE.md hardware & connection model
    ├── SDK_NOTES.md    SDK gotchas and correct API usage
    ├── AUDIO_PIPELINE.md speaker routing details
    ├── CAMERA_PIPELINE.md direct V4L2 camera access
    ├── SAFETY.md       motor heat limits and safe patterns
    └── SETUP.md        full install log
```

---

## Requirements

**System packages:**
```bash
sudo apt install ffmpeg alsa-utils python3-venv
```

**Python venv:**
```bash
python3 -m venv .venv
.venv/bin/pip install reachy-mini piper-tts onnxruntime huggingface-hub librosa
```

**Piper voice model** (not in repo — 61 MB):
```bash
mkdir -p voices
cd voices
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx
# The .onnx.json config file is already included in the repo
```

See [`docs/SETUP.md`](docs/SETUP.md) for the full install log.

---

## Music attribution

*Blipotron* and *Kick Shock* by **Kevin MacLeod** (incompetech.com)
Licensed under Creative Commons Attribution 3.0 — [CC-BY 3.0](https://creativecommons.org/licenses/by/3.0/)
