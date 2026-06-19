#!/usr/bin/env python3
"""
Record 3 seconds from every capture device. Shows volume and plays back on robot speaker.
Run this, speak clearly, then listen which playback has your voice.
"""
import subprocess, tempfile, os, time
from pathlib import Path

SPEAKER = "plughw:CARD=Audio,DEV=0"
DURATION = 3

DEVICES = [
    ("plughw:CARD=Audio,DEV=0",   "Reachy Mini Audio  (card 2 — expected robot mic)"),
    ("plughw:CARD=Camera,DEV=0",  "Reachy Mini Camera (card 1 — camera mic)"),
    ("plughw:0,0",                "Laptop internal mic (card 0)"),
    ("default",                    "PipeWire default source"),
    ("pulse",                      "PulseAudio / PipeWire-pulse bridge"),
]

for dev, label in DEVICES:
    wav = tempfile.mktemp(suffix=".wav")
    print(f"\n{'='*60}")
    print(f"Device : {dev}")
    print(f"Label  : {label}")
    print(f"SPEAK NOW — recording {DURATION}s ...")

    ret = subprocess.run(
        ["arecord", "-D", dev, "-f", "S16_LE", "-r", "16000", "-c", "1",
         "-d", str(DURATION), wav],
        capture_output=True, text=True,
    )

    if ret.returncode != 0:
        print(f"  arecord FAILED: {ret.stderr.strip()[:120]}")
        time.sleep(0.5)
        continue

    size = Path(wav).stat().st_size
    # Volume check
    vol = subprocess.run(
        ["ffmpeg", "-i", wav, "-af", "volumedetect", "-f", "null", "/dev/null"],
        capture_output=True, text=True,
    )
    max_vol = "[no data]"
    for line in vol.stderr.splitlines():
        if "max_volume" in line:
            max_vol = line.strip()
    print(f"  Size   : {size} bytes")
    print(f"  Volume : {max_vol}")

    print("  Playing back on robot speaker (plughw:CARD=Audio,DEV=0) ...")
    subprocess.run(
        ["aplay", "-D", SPEAKER, "-q", wav],
        capture_output=True,
    )
    print("  Playback done.")
    Path(wav).unlink(missing_ok=True)
    time.sleep(1)

print("\n\nDone. Tell me which device played back your voice.")
