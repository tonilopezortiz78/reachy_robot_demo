#!/usr/bin/env python3
"""Play a beep on every plausible robot speaker device. Listen for which one makes sound."""
import subprocess, time

DEVICES = [
    ("plughw:CARD=Audio,DEV=0",   "Reachy Mini Audio (card 2) — expected robot speaker"),
    ("plughw:CARD=Camera,DEV=0",  "Reachy Mini Camera (card 1)"),
    ("default",                    "PipeWire default output"),
]

BEEP = [
    "ffmpeg", "-hide_banner", "-loglevel", "error",
    "-f", "lavfi", "-i", "aevalsrc=0.6*sin(880*2*PI*t):c=mono:s=22050",
    "-t", "0.8",
    "-f", "alsa",
]

for dev, label in DEVICES:
    print(f"\nPlaying beep on: {dev}")
    print(f"  ({label})")
    ret = subprocess.run(BEEP + [dev], capture_output=True)
    if ret.returncode == 0:
        print("  -> OK")
    else:
        print(f"  -> FAILED: {ret.stderr.decode().strip()[:120]}")
    time.sleep(0.5)

print("\nDone. Which device did you hear the beep from?")
