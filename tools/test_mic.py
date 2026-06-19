#!/usr/bin/env python3
"""
Test every capture device via PipeWire (parecord).
For each device: records 3s, shows volume, plays back on robot speaker.
SPEAK CLEARLY during each 3-second window.
"""
import subprocess, tempfile, time
from pathlib import Path

SPEAKER = "plughw:CARD=Audio,DEV=0"
DURATION = 3

# PipeWire source names (use parecord, not arecord+plughw)
DEVICES = [
    ("alsa_input.usb-Pollen_Robotics_Reachy_Mini_Audio_100025004254700094-00.analog-stereo",
     "Reachy Mini Audio mic (robot body)"),
    ("alsa_input.usb-SunplusIT_Inc_Reachy_Mini_Camera_J20251031V0-02.analog-stereo",
     "Reachy Mini Camera mic (robot head)"),
    ("alsa_input.pci-0000_00_1f.3.analog-stereo",
     "Laptop built-in mic"),
]

for dev, label in DEVICES:
    wav = tempfile.mktemp(suffix=".wav")
    print(f"\n{'='*60}")
    print(f"Label  : {label}")
    print(f"Device : {dev[:60]}")
    print(f"SPEAK NOW — recording {DURATION}s ...")

    proc = subprocess.Popen(
        ["parecord", f"--device={dev}",
         "--channels=1", "--rate=16000", "--format=s16le", wav],
        stderr=subprocess.DEVNULL,
    )
    time.sleep(DURATION)
    proc.terminate()
    proc.wait()

    if not Path(wav).exists() or Path(wav).stat().st_size < 100:
        print("  FAILED — no audio captured")
        time.sleep(0.5)
        continue

    size = Path(wav).stat().st_size
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
    print(f"  Playing back on robot speaker ...")
    subprocess.run(["aplay", "-D", SPEAKER, "-q", wav], capture_output=True)
    print("  Done.")
    Path(wav).unlink(missing_ok=True)
    time.sleep(1)

print("\n\nDone. Tell me which playback had your voice.")
