#!/usr/bin/env python3
"""Live VAD diagnostic — prints Silero speech probability for each 32ms chunk.
Uses the SAME auto-detected robot mic as the talking demos (reachy_demo.audio.MIC),
so a healthy run here means the demos will hear you too. If you see "arecord
closed" immediately, the mic is held by an orphan — run:
    pkill -9 -f 'pacat --record'   (or use reachy_demo.audio.cleanup_orphan_capture())
"""
import subprocess, sys, numpy as np, torch
sys.path.insert(0, ".")

from reachy_demo.audio import MIC, MIC_RATE, VAD_CHUNK, cleanup_orphan_capture

print(f"Using mic: {MIC}")
print("Cleaning any orphan capture processes first...")
n = cleanup_orphan_capture()
if n:
    print(f"  Killed {n} orphan(s).")
print("Loading Silero VAD model...")
vad, _ = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
vad.eval()
print("Ready. Speak into the robot mic. Ctrl+C to stop.\n")

arecord = subprocess.Popen(
    ["pacat", "--record", "--raw", f"--device={MIC}",
     f"--rate={MIC_RATE}", "--channels=1", "--format=s16le"],
    stdout=subprocess.PIPE, stderr=None,
)

try:
    while True:
        raw = arecord.stdout.read(VAD_CHUNK * 2)
        if not raw or len(raw) < VAD_CHUNK * 2:
            print("arecord closed — mic delivered no audio. "
                  "Check for orphan processes or replug USB.")
            break
        f32 = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        prob = vad(torch.from_numpy(f32), MIC_RATE).item()
        db = 20 * np.log10(np.sqrt(np.mean(f32 ** 2)) + 1e-10)
        bar = "#" * int(prob * 40)
        tag = "SPEECH" if prob > 0.25 else "      "
        print(f"  {tag}  prob={prob:.2f}  dBFS={db:+.1f}  |{bar:<40}|")
except KeyboardInterrupt:
    pass
finally:
    arecord.terminate()
    arecord.wait()
