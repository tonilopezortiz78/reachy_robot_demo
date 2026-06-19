#!/usr/bin/env python3
"""Live VAD diagnostic — prints Silero speech probability for each 32ms chunk."""
import subprocess, sys, numpy as np, torch
sys.path.insert(0, ".")

MIC = "plughw:CARD=Camera,DEV=0"
RATE = 16000
CHUNK = 512  # 32ms

print("Loading Silero VAD model...")
vad, _ = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
vad.eval()
print("Ready. Speak into the robot mic. Ctrl+C to stop.\n")

arecord = subprocess.Popen(
    ["arecord", "-D", MIC, "-f", "S16_LE", "-r", str(RATE), "-c", "1", "-q"],
    stdout=subprocess.PIPE, stderr=None,
)

try:
    while True:
        raw = arecord.stdout.read(CHUNK * 2)
        if not raw or len(raw) < CHUNK * 2:
            print("arecord closed.")
            break
        f32 = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        prob = vad(torch.from_numpy(f32), RATE).item()
        db = 20 * np.log10(np.sqrt(np.mean(f32 ** 2)) + 1e-10)
        bar = "#" * int(prob * 40)
        tag = "SPEECH" if prob > 0.25 else "      "
        print(f"  {tag}  prob={prob:.2f}  dBFS={db:+.1f}  |{bar:<40}|")
except KeyboardInterrupt:
    pass
finally:
    arecord.terminate()
    arecord.wait()
