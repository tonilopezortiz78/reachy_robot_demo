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

MIC = "plughw:CARD=Camera,DEV=0"

print("Recording 5 seconds — speak now...")
wav = tempfile.mktemp(suffix=".wav")
subprocess.run(
    ["arecord", "-D", MIC, "-f", "S16_LE", "-r", "16000", "-c", "1", "-d", "5", wav],
    check=True,
)

size = Path(wav).stat().st_size
print(f"Captured {size} bytes. Sending to Whisper...")

with open(wav, "rb") as f:
    result = client.audio.transcriptions.create(
        model="whisper-large-v3-turbo",
        file=("audio.wav", f),
        language="en",
    )

Path(wav).unlink(missing_ok=True)
print(f"\nWhisper heard: {result.text!r}")
