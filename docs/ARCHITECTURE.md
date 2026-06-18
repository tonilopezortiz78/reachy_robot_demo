# Architecture — what's actually inside Reachy Mini Lite

## High-level mental model (wrong vs right)

**Wrong:** "The Raspberry Pi inside the robot runs the programs, and my laptop talks to it over the network."

**Right:** "The robot is a USB peripheral. The Raspberry Pi inside is a tiny USB hub + serial bridge for the motors. The laptop is the computer that runs every program. There is no network between them."

## Physical connections (one USB cable, three logical devices)

```
Laptop USB-C port
        │
        ▼
┌────────────────────────────────────────┐
│  Reachy Mini (single USB-C cable)      │
│                                        │
│   ┌──────────────┐                     │
│   │ Raspberry Pi │── internal serial ──▶ Motor controller (Feetech)
│   │  CM4         │      /dev/ttyACM0   ▶ 8 servos (head + antennas)
│   └──────────────┘                     │
│                                        │
│   ┌──────────────┐                     │
│   │ USB Audio    │────────────────────▶ Host USB audio (card 2)
│   │ (codec)      │  speaker + 4-mic     "Reachy Mini Audio"
│   └──────────────┘                     │
│                                        │
│   ┌──────────────┐                     │
│   │ USB Camera   │────────────────────▶ Host UVC (video2)
│   │ (SunplusIT)  │  1920×1080@30        "Reachy Mini Camera"
│   └──────────────┘                     │
│                                        │
└────────────────────────────────────────┘
```

## How the SDK finds each thing

| Function | Path | Notes |
|---|---|---|
| Motors | `/dev/ttyACM0` (CH340-compatible bridge) | Daemon talks Feetech protocol |
| Speaker | `plughw:2,0` (ALSA direct) | Bypasses PulseAudio routing |
| Mic | `plughw:2,0` (capture direction) | ReSpeaker-style 4-mic array |
| Camera | `/dev/video2` (V4L2/UVC) | SunplusIT sensor, 1080p |

The Pi is essentially a "USB sound card + webcam + motor hat" all rolled into one. Your laptop treats it like three separate USB devices.

## Wireless version vs Lite (for clarity)

| Feature | Lite (you have this) | Wireless |
|---|---|---|
| Computer | Your laptop | CM4 inside the robot |
| Power | Through your laptop's USB port | Battery + USB-C PD |
| Connection | USB | WiFi (mDNS `reachy-mini.local`) |
| Camera | Direct USB UVC | Direct on-board CSI/USB |
| Compute for AI | Full laptop CPU/GPU | Limited CM4 |
| Recommended for | Development, dev loops | Deployment, autonomy |

You can tell a Lite from a Wireless by:
- No `reachy-mini.local` mDNS resolution when you `ping reachy-mini.local`
- No `usb0` / `eth1` RNDIS interface from the robot
- The robot only works when its USB-C is plugged into a computer

## The back switch — what it really does

The robot has a 2-position slide switch on the back. Despite the labelling, it does **not** turn the robot on/off. It selects which "persona" the Pi presents over USB:

- **Robot / Developer position**: the Pi runs the robot's control stack. Motors, audio, camera are exposed to the laptop. The SDK can drive the motors.
- **Computer position**: the Pi boots into a desktop environment and outputs video to its internal display. Used for first-time setup (WiFi, password, app install). In this mode the SDK **cannot** drive the motors.

**Always leave the switch in Robot/Developer position for any program in this repo.** The green LED should be solid or slow-blinking (the Pi is on and the control stack is up).

## Why the Pi needs to be there at all

The motor controller speaks a synchronous serial protocol that the laptop cannot do directly. The Pi converts USB packets from the laptop into Feetech motor commands. That's its only job in Lite mode.

## Daemon, SDK, and where they run

- **Daemon** (`reachy-mini-daemon`): runs on the **laptop** (when `spawn_daemon=True` it auto-starts, otherwise you start it manually). Holds the persistent motor connection, exposes a WebSocket on `localhost:8000` and a REST API.
- **SDK** (`reachy_mini` Python package): runs on the **laptop**. Your script opens `ReachyMini()` and the SDK talks WebSocket to the daemon.
- **Browser apps** (JS / HuggingFace Spaces): also on the laptop (or any laptop, anywhere) — they talk to the same daemon via WebRTC through a central signalling server.

The Pi itself runs **no application code** in Lite mode. It's a peripheral.
