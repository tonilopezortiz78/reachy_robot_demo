#!/usr/bin/env python3
"""Record 5 seconds from robot mic, transcribe with Groq Whisper, print result."""
import os, subprocess, tempfile
from pathlib import Path
from groq import Groq

def _load_key():
    env = Path(__file__).parent.parent / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "GROQ_API_KEY" in line:
                return line.replace("GROQ_API_KEY:", "").replace("GROQ_API_KEY=", "").strip()
    return os.environ.get("GROQ_API_KEY")

key = _load_key()
if not key:
    raise SystemExit("GROQ_API_KEY not found in .env")
client = Groq(api_key=key)

MIC = "default"   # PipeWire routes to Pollen Robotics Audio input (the robot mic)

SPEAKER = "plughw:CARD=Audio,DEV=0"

print("Recording 5 seconds — speak into the robot camera mic now...")
wav = tempfile.mktemp(suffix=".wav")
subprocess.run(
    ["arecord", "-D", MIC, "-f", "S16_LE", "-r", "16000", "-c", "1", "-d", "5", wav],
    check=True,
)

size = Path(wav).stat().st_size
print(f"Captured {size} bytes.")

# Show volume level
result = subprocess.run(
    ["ffmpeg", "-i", wav, "-af", "volumedetect", "-f", "null", "/dev/null"],
    capture_output=True, text=True,
)
for line in result.stderr.splitlines():
    if "max_volume" in line or "mean_volume" in line:
        print(" ", line.strip())

# Play back so you can hear what the mic recorded
print("Playing back on robot speaker...")
subprocess.run(["aplay", "-D", SPEAKER, "-q", wav], check=False)

# Transcribe
print("Sending to Whisper...")
with open(wav, "rb") as f:
    trans = client.audio.transcriptions.create(
        model="whisper-large-v3-turbo",
        file=("audio.wav", f),
        language="en",
    )

Path(wav).unlink(missing_ok=True)
print(f"\nWhisper heard: {trans.text!r}")
