"""
tools/debug_one_turn.py — record ONE turn from the robot mic and run the full
language pipeline (Whisper → resolve language → LLM), with no daemon/motors.

This isolates the mic + STT + language-matching path — the part that was failing
("Japanese in, Spanish out") — from the robot control stack (which triggers a
sandbox kill). Records to logs/ like the real demo so the audio is replayable.

Usage:  ./run.sh tools/debug_one_turn.py [seconds]
"""
import subprocess
import sys
import time
from pathlib import Path

from groq import Groq

from reachy_demo.audio import MIC, MIC_RATE
from reachy_demo.groq_client import (
    load_api_key, transcribe_lang, resolve_language, language_directive,
)
from reachy_demo.session_log import SessionLogger
from reachy_demo.text import clean_for_tts

ROOT = Path(__file__).parent.parent
MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
SECONDS = float(sys.argv[1]) if len(sys.argv) > 1 else 8.0

log = SessionLogger(ROOT, "debug_one_turn")
log.event(f"MIC = {MIC}")

# Countdown so the person knows exactly when to start talking
for i in (3, 2, 1):
    print(f"  speak in {i}...", flush=True)
    time.sleep(1)
print(f"  >>> TALK NOW — say a clear sentence (Japanese, English, anything) <<<", flush=True)
log.event(f"Recording {SECONDS:.0f}s...")

# Record straight from the auto-detected robot mic
proc = subprocess.Popen(
    ["pacat", "--record", "--raw", f"--device={MIC}",
     f"--rate={MIC_RATE}", "--channels=1", "--format=s16le"],
    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
)
time.sleep(SECONDS)
proc.terminate()
pcm = proc.stdout.read() if proc.stdout else b""
try:
    proc.wait(timeout=2)
except Exception:
    proc.kill()

# Level check — was the mic actually capturing the user?
import numpy as np
arr = np.frombuffer(pcm, dtype=np.int16) if pcm else np.array([], dtype=np.int16)
rms = float(np.sqrt(np.mean(arr.astype(np.float32) ** 2))) if len(arr) else 0.0
peak = int(np.abs(arr).max()) if len(arr) else 0
audio_path = log.save_audio(pcm, rate=MIC_RATE)
log.event(f"Captured {len(arr)} samples  RMS={rms:.0f} peak={peak}  -> {audio_path}")
if rms < 30:
    log.event("⚠️  Signal very low — mic may not have heard you. Try again louder/closer.")

# Wrap PCM as WAV for Whisper
import io, wave
buf = io.BytesIO()
with wave.open(buf, "wb") as wf:
    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(MIC_RATE)
    wf.writeframes(pcm)
wav_bytes = buf.getvalue()

client = Groq(api_key=load_api_key(ROOT))
text, whisper_lang = transcribe_lang(client, wav_bytes)
final_lang = resolve_language(text, whisper_lang)
directive = language_directive(final_lang)
log.event(f"Whisper heard: {text!r}")
log.event(f"Whisper lang=[{whisper_lang}]  ->  final lang=[{final_lang}]"
          + ("  (script OVERRODE Whisper)" if final_lang != whisper_lang else ""))

reply = ""
if text:
    SYS = ("You are Reachy, a cute little robot. Reply in ONE short sentence, "
           "max 10 words.")
    msgs = [{"role": "system", "content": SYS},
            {"role": "user", "content": text}]
    if directive:
        msgs.append({"role": "system", "content": directive})
    r = client.chat.completions.create(model=MODEL, messages=msgs,
                                       max_tokens=45, temperature=0.8)
    reply = clean_for_tts(r.choices[0].message.content)
    log.event(f"Reachy would say [{final_lang}]: {reply}")

log.turn(kind="debug_turn", mic=MIC, rms=round(rms, 1), peak=peak,
         audio=audio_path, whisper_lang=whisper_lang, final_lang=final_lang,
         transcript=text, directive=directive, reply=reply)
log.event(f"Recorded to: {log.dir}")
print("\nSUMMARY:")
print(f"  mic rms/peak : {rms:.0f}/{peak}")
print(f"  heard        : {text!r}")
print(f"  lang         : whisper=[{whisper_lang}] final=[{final_lang}]")
print(f"  reply        : {reply!r}")
