#!/usr/bin/env python3
"""
Test capture devices one at a time. Beeps before each recording.
SPEAK after the beep. Plays back on robot speaker.
"""
import subprocess, tempfile, time, sys
from pathlib import Path

SPEAKER = "plughw:CARD=Audio,DEV=0"

DEVICES = [
    ("alsa_input.usb-Pollen_Robotics_Reachy_Mini_Audio_100025004254700094-00.analog-stereo",
     "1 — Reachy body mic"),
    ("alsa_input.usb-SunplusIT_Inc_Reachy_Mini_Camera_J20251031V0-02.analog-stereo",
     "2 — Reachy camera mic"),
    ("alsa_input.pci-0000_00_1f.3.analog-stereo",
     "3 — Laptop built-in mic"),
]

def beep():
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "aevalsrc=0.5*sin(1000*2*PI*t):c=mono:s=22050",
         "-t", "0.3", "-f", "alsa", SPEAKER],
        check=False,
    )

for dev, label in DEVICES:
    print(f"\n{'='*50}")
    print(f"Testing: {label}")
    print("Beep → SPEAK for 5 seconds → playback")
    time.sleep(1)

    beep()
    time.sleep(0.2)

    wav = tempfile.mktemp(suffix=".wav")
    print("Recording... speak now!")
    proc = subprocess.Popen(
        ["parecord", f"--device={dev}",
         "--channels=1", "--rate=16000", "--format=s16le", wav],
        stderr=subprocess.DEVNULL,
    )
    time.sleep(5)
    proc.terminate()
    proc.wait()

    if not Path(wav).exists() or Path(wav).stat().st_size < 1000:
        print("  FAILED — no audio")
        time.sleep(0.5)
        continue

    vol = subprocess.run(
        ["ffmpeg", "-i", wav, "-af", "volumedetect", "-f", "null", "/dev/null"],
        capture_output=True, text=True,
    )
    for line in vol.stderr.splitlines():
        if "max_volume" in line:
            print(f"  {line.strip()}")

    print("  Playing back on robot speaker ...")
    beep()
    time.sleep(0.2)
    subprocess.run(["aplay", "-D", SPEAKER, "-q", wav], capture_output=True)
    print("  Done. Did you hear your voice? (check terminal)")
    Path(wav).unlink(missing_ok=True)

    cont = input("  Press Enter for next device (or q to quit): ").strip()
    if cont.lower() == "q":
        break

print("\nDone.")
