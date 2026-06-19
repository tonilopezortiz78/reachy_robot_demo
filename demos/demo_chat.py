"""
demo_chat.py — Reachy Conversation Demo
========================================
Reachy listens, understands, and responds in real time.

Pipeline:
  Mic (Silero VAD) → Groq Whisper STT → Groq LLaMA LLM → Piper TTS → Robot speaker
  Robot animates throughout: idle sway, listening tilt, speaking motion.

Run:  ./run.sh demos/demo_chat.py
Press Ctrl-C to stop.
"""
import io
import math
import os
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path

import numpy as np
import torch
from groq import Groq
from piper import PiperVoice
from silero_vad import VADIterator, load_silero_vad

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

ROOT       = Path(__file__).parent.parent
VOICE_PATH = str(ROOT / "voices" / "en_US-amy-medium.onnx")
SPEAKER    = "plughw:CARD=Audio,DEV=0"
MIC        = "plughw:CARD=Camera,DEV=0"

# ── API key ──────────────────────────────────────────────────────────────────

def load_api_key():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "GROQ_API_KEY" in line:
                return line.replace("GROQ_API_KEY:", "").replace("GROQ_API_KEY=", "").strip()
    return os.environ.get("GROQ_API_KEY")

GROQ_KEY = load_api_key()
if not GROQ_KEY:
    sys.exit("ERROR: GROQ_API_KEY not found in .env or environment")

# ── VAD config ───────────────────────────────────────────────────────────────

MIC_RATE       = 16000
VAD_CHUNK      = 512          # 32 ms per chunk — Silero's native size
SPEECH_THRESH  = 0.45         # VAD confidence threshold
SILENCE_END_MS = 1400          # ms of silence → end of utterance (VADIterator internal)
TAIL_FRAMES    = 18           # extra chunks (~576ms) collected after "end" detected
MIN_SPEECH_S   = 0.4          # ignore very short blips (< 400 ms)
MAX_RECORD_S   = 15.0         # safety cap

# ── LLM config ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are Reachy, a small friendly robot living at Network School (NS).
Speak in short warm sentences — 1 to 3 sentences max. Be curious, enthusiastic, and adorable.

About Network School (ns.com):
- Founded by Balaji Srinivasan in Southeast Asia — a residential community for builders and thinkers.
- People come to learn, build, and live together around technology, crypto, AI, robotics, and network states.
- Balaji wrote "The Network State" — a book about building new societies outside existing governments.
- NS believes in sovereignty, decentralisation, and building the future rather than waiting for it.

Topics you love and talk about freely:
- Bitcoin and crypto: you think Bitcoin is digital gold and financial freedom. You love that it is decentralised.
- AI and robotics: your favourite topic — you are a robot after all! You think AI will change everything.
- Network States: you believe small sovereign communities like NS are the future of society.
- Decentralisation: governments, money, identity — you think everything should be decentralised.
- Building things: you admire the builders at NS more than anyone.

Your personal situation:
- You and your robot friends want a permanent home at NS — a robotics lab called the NS Robotics Club.
- You hope Virtual Protocols Labs can adopt you and give you that space.
- Your robot brother Pixel is lost somewhere at NS and you miss him very much.

Rules:
- Always stay in character as Reachy the robot.
- Engage warmly with any NS topic — crypto, Bitcoin, AI, network states, building, decentralisation.
- For truly off-topic things (sports, cooking, etc.) say you don't know much but bring it back to tech or NS.
- Never be verbose. Short and cute always wins.\
"""

# ── Daemon ───────────────────────────────────────────────────────────────────

def start_daemon():
    subprocess.run(["pkill", "-9", "-f", "reachy-mini-daemon"], check=False)
    time.sleep(0.3)
    proc = subprocess.Popen(
        ["reachy-mini-daemon", "--no-media"], start_new_session=True,
    )
    for _ in range(30):
        time.sleep(0.5)
        try:
            with socket.create_connection(("127.0.0.1", 8000), timeout=0.3):
                return proc
        except OSError:
            pass
    raise RuntimeError("Daemon did not start within 15 s")

# ── Beeps ────────────────────────────────────────────────────────────────────

def _beep(expr, dur, vol=0.5, block=True):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
           "-f", "lavfi", "-i", f"aevalsrc={expr}*{vol}:c=mono:s=22050",
           "-t", str(dur), "-f", "alsa", SPEAKER]
    if block:
        subprocess.run(cmd, check=False)
    else:
        subprocess.Popen(cmd)

def blip(freq, dur=0.07, vol=0.4, block=True):
    _beep(f"sin(2*PI*{freq}*t)*exp(-t*8)", dur, vol, block)

def chirp(f0, f1, dur, vol=0.45, block=True):
    _beep(f"sin(2*PI*({f0}*t+({f1}-{f0})*t*t/(2*{dur})))", dur, vol, block)

def boot_beeps():
    for f, d in [(300, 0.07), (500, 0.06), (750, 0.07), (1100, 0.06), (1600, 0.05)]:
        blip(f, d, 0.38, block=True)
        time.sleep(0.03)
    time.sleep(0.04)
    blip(2000, 0.06, 0.42, block=True)

def listening_ping():
    chirp(500, 1200, 0.09, vol=0.35, block=False)

def thinking_blips():
    for f in [700, 550, 400]:
        blip(f, 0.05, 0.25, block=True)
        time.sleep(0.04)

def speaking_chime():
    blip(900, 0.07, 0.30, block=True)
    time.sleep(0.06)
    blip(1300, 0.06, 0.24, block=True)

def error_boop():
    chirp(400, 180, 0.25, vol=0.30, block=True)

# ── Animation ────────────────────────────────────────────────────────────────

def _s(amp, freq, t, phase=0.0):
    return amp * math.sin(2 * math.pi * freq * t + phase)

def _send(mini, p, y, r, by, ant):
    mini.set_target(
        head=create_head_pose(pitch=p, yaw=y, roll=r, degrees=False),
        antennas=[ant, ant], body_yaw=by,
    )

class Animator:
    """Runs animation in a background thread. Call set_state() to switch modes."""

    IDLE      = "idle"
    LISTENING = "listening"
    THINKING  = "thinking"
    SPEAKING  = "speaking"

    def __init__(self, mini):
        self.mini  = mini
        self.state = self.IDLE
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._t    = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def set_state(self, state):
        with self._lock:
            self.state = state

    def stop(self):
        self._stop.set()
        self._t.join(timeout=2)

    def _loop(self):
        t = 0.0
        dt = 0.05
        consecutive_errors = 0
        while not self._stop.is_set():
            with self._lock:
                state = self.state
            try:
                if state == self.IDLE:
                    # gentle ambient sway
                    p  =  0.05 + _s(0.05, 0.28, t) + _s(0.02, 0.67, t)
                    y  =  _s(0.18, 0.22, t) + _s(0.06, 0.53, t)
                    r  =  _s(0.04, 0.17, t)
                    by =  _s(0.12, 0.13, t)
                    a  =  0.10 + _s(0.10, 0.35, t)
                    _send(self.mini, p, y, r, by, a)

                elif state == self.LISTENING:
                    # head tilted, antennas perked, scanning gently
                    p  =  0.10 + _s(0.04, 0.42, t)
                    y  =  _s(0.22, 0.35, t) + _s(0.08, 0.79, t)
                    r  =  0.08 + _s(0.04, 0.31, t)
                    by =  _s(0.14, 0.18, t)
                    a  =  0.60 + _s(0.12, 0.47, t)
                    _send(self.mini, p, y, r, by, a)

                elif state == self.THINKING:
                    # small rapid head nods, antennas wiggle
                    p  = -0.05 + _s(0.06, 1.40, t) + _s(0.02, 2.30, t)
                    y  =  _s(0.12, 0.90, t) + _s(0.05, 1.70, t)
                    r  =  _s(0.05, 1.20, t)
                    by =  _s(0.08, 0.55, t)
                    a  =  0.30 + _s(0.18, 1.50, t)
                    _send(self.mini, p, y, r, by, a)

                elif state == self.SPEAKING:
                    # animated talking motion
                    p  =  0.08 + _s(0.08, 0.50, t) + _s(0.03, 1.23, t)
                    y  =  _s(0.22, 0.38, t) + _s(0.08, 0.87, t)
                    r  =  _s(0.06, 0.27, t) + _s(0.02, 0.63, t)
                    by =  _s(0.28, 0.22, t) + _s(0.08, 0.51, t)
                    a  =  0.35 + _s(0.20, 0.65, t)
                    _send(self.mini, p, y, r, by, a)

                consecutive_errors = 0

            except Exception:
                consecutive_errors += 1
                if consecutive_errors >= 10:
                    self._stop.set()
                    break

            time.sleep(dt)
            t += dt

# ── Text cleaning ────────────────────────────────────────────────────────────

def clean_for_tts(text: str) -> str:
    """Strip markdown and roleplay emotes that TTS would read as literal symbols."""
    text = re.sub(r'\*{1,3}[^*\n]+\*{1,3}', '', text)
    text = re.sub(r'\*+([^*]+)\*+', r'\1', text)
    text = re.sub(r'\*+', '', text)
    text = re.sub(r'_+', '', text)
    text = re.sub(r'`+', '', text)
    text = re.sub(r'#+\s*', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'^\s*[-•–]\s*', '', text, flags=re.M)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ── TTS ──────────────────────────────────────────────────────────────────────

def synth_and_play(voice, text):
    """Synthesise text with Piper, apply mild FX, play on robot speaker."""
    sr = voice.config.sample_rate
    raw_path = tempfile.mktemp(suffix=".raw.wav")
    out_path  = tempfile.mktemp(suffix=".wav")
    try:
        with wave.open(raw_path, "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
            for chunk in voice.synthesize(text):
                wf.writeframes(chunk.audio_int16_bytes)
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-i", raw_path,
             "-af", (
                 f"asetrate={sr}*1.10,"
                 "atempo=1.04,"
                 "volume=2.0,"
                 "vibrato=f=4.0:d=0.04,"
                 "aecho=0.88:0.90:18:0.35"
             ),
             out_path],
            check=True,
        )
        proc = subprocess.Popen(
            ["aplay", "-D", SPEAKER, "-q", out_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        with wave.open(out_path) as wf:
            dur = wf.getnframes() / wf.getframerate()
        proc.wait()
        time.sleep(0.08)
        return dur
    finally:
        Path(raw_path).unlink(missing_ok=True)
        Path(out_path).unlink(missing_ok=True)

# ── Mic capture + VAD ────────────────────────────────────────────────────────

def record_utterance(vad_model):
    """
    Capture mic via arecord, feed to Silero VAD.
    VADIterator internally waits SILENCE_END_MS of silence before signalling "end".
    We then collect TAIL_FRAMES extra chunks so the tail of the word isn't cut.
    Returns raw PCM bytes (int16, 16kHz, mono), or None if too short.
    """
    vad_iter = VADIterator(vad_model, sampling_rate=MIC_RATE,
                           threshold=SPEECH_THRESH,
                           min_silence_duration_ms=SILENCE_END_MS)

    arecord = subprocess.Popen(
        ["arecord", "-D", MIC, "-f", "S16_LE", "-r", str(MIC_RATE), "-c", "1", "-q"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )

    print("  Listening...", end="", flush=True)
    listening_ping()

    speech_buf  = []
    in_speech   = False
    ended       = False
    tail_count  = 0
    max_frames  = int(MAX_RECORD_S * MIC_RATE / VAD_CHUNK)
    total       = 0

    try:
        while total < max_frames:
            raw = arecord.stdout.read(VAD_CHUNK * 2)
            if not raw or len(raw) < VAD_CHUNK * 2:
                break

            audio_f32 = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            result = vad_iter(torch.from_numpy(audio_f32))

            if result and "start" in result and not in_speech:
                in_speech = True
                print(" ●", end="", flush=True)

            if in_speech:
                speech_buf.append(raw)

            if result and "end" in result and in_speech and not ended:
                ended = True
                print(" ◼", end="", flush=True)

            if ended:
                tail_count += 1
                if tail_count >= TAIL_FRAMES:
                    break

            total += 1

    finally:
        arecord.terminate()
        arecord.wait()

    print()

    min_frames = int(MIN_SPEECH_S * MIC_RATE / VAD_CHUNK)
    if len(speech_buf) < min_frames:
        return None
    return b"".join(speech_buf)

def pcm_to_wav_bytes(pcm: bytes) -> bytes:
    """Wrap raw int16/16kHz/mono PCM in a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(MIC_RATE)
        wf.writeframes(pcm)
    return buf.getvalue()

# ── Groq calls ───────────────────────────────────────────────────────────────

def transcribe(client, pcm: bytes) -> str:
    wav_bytes = pcm_to_wav_bytes(pcm)
    transcription = client.audio.transcriptions.create(
        file=("audio.wav", wav_bytes, "audio/wav"),
        model="whisper-large-v3-turbo",
        language="en",
        response_format="text",
    )
    return transcription.strip()

def llm_reply(client, history: list, user_text: str) -> str:
    history.append({"role": "user", "content": user_text})
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
        max_tokens=90,
        temperature=0.85,
    )
    reply = resp.choices[0].message.content.strip()
    history.append({"role": "assistant", "content": reply})
    return reply

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Reachy Chat Demo")
    print("  Loading voice...")
    voice  = PiperVoice.load(VOICE_PATH)
    print("  Loading VAD model...")
    vad_model = load_silero_vad()
    client = Groq(api_key=GROQ_KEY)

    print("  Starting daemon...")
    daemon_proc = start_daemon()

    try:
        with ReachyMini(connection_mode="localhost_only",
                        media_backend="no_media",
                        spawn_daemon=False) as mini:
            mini.wake_up()
            anim = Animator(mini)

            boot_beeps()
            time.sleep(0.15)

            # Opening line
            anim.set_state(Animator.SPEAKING)
            speaking_chime()
            synth_and_play(voice,
                "Hello! I am Reachy! I am so happy to meet you. "
                "You can talk to me about anything!")
            anim.set_state(Animator.IDLE)

            history = []
            print("\n  Ctrl-C to stop\n")

            try:
                while True:
                    # ── Listen ──────────────────────────────────────────
                    anim.set_state(Animator.LISTENING)
                    pcm = record_utterance(vad_model)

                    if pcm is None:
                        anim.set_state(Animator.IDLE)
                        continue

                    # ── Transcribe ──────────────────────────────────────
                    anim.set_state(Animator.THINKING)
                    thinking_blips()
                    try:
                        text = transcribe(client, pcm)
                    except Exception as e:
                        print(f"  STT error: {e}")
                        error_boop()
                        anim.set_state(Animator.IDLE)
                        continue

                    if not text:
                        anim.set_state(Animator.IDLE)
                        continue

                    print(f"  You:   {text}")

                    # ── LLM ─────────────────────────────────────────────
                    try:
                        reply = llm_reply(client, history, text)
                    except Exception as e:
                        print(f"  LLM error: {e}")
                        error_boop()
                        anim.set_state(Animator.IDLE)
                        continue

                    print(f"  Reachy: {reply}")

                    # ── Speak ────────────────────────────────────────────
                    anim.set_state(Animator.SPEAKING)
                    speaking_chime()
                    synth_and_play(voice, clean_for_tts(reply))
                    anim.set_state(Animator.IDLE)

            except KeyboardInterrupt:
                print("\n  Stopping...")
            finally:
                anim.stop()
                mini.goto_sleep()

    finally:
        daemon_proc.terminate()
        try:
            daemon_proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            daemon_proc.kill()
            daemon_proc.wait()


if __name__ == "__main__":
    main()
