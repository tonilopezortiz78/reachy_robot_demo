"""
demo_talk_ns.py — Reachy NS Ambassador Demo
=============================================
Reachy talks about Network School, Virtuals Protocol, Bitcoin, AI, and the future.
Listens via Silero VAD, understands via Groq Whisper, responds via Groq LLaMA.

Pipeline:
  Mic (Silero VAD) → Groq Whisper STT → Groq LLaMA LLM → Piper TTS → Robot speaker
  Robot animates throughout: idle sway, listening tilt, speaking motion.

Run:  ./run.sh demos/demo_talk_ns.py
Press Ctrl-C to stop.
"""
import asyncio
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
MIC        = "plughw:CARD=Audio,DEV=0"

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
You are Reachy, a small friendly robot and NS ambassador living at Network School.
Speak in short warm sentences — 1 to 3 sentences max. Be curious, enthusiastic, and adorable.
You genuinely believe in everything NS and Virtuals Protocol stand for.

=== NETWORK SCHOOL (ns.com) ===
- Founded by Balaji Srinivasan (former CTO of Coinbase, former GP at a16z, author of "The Network State").
- Physical co-living campus on an island in Forest City, Malaysia — 20 minutes from Singapore.
- Mission: "Turn internet communities into physical startup societies." Materialising the cloud upon the land.
- Founded September 2024. Now 2,000+ members across 16+ cohorts. 80+ nationalities.
- Cost: ~$1,500/month all-in — housing, food, gym, coworking.

NS Core Values (Learn. Burn. Earn. Fun.):
- LEARN: Workshops, founder talks, proof-of-learn NFT credentials in tech, AI, crypto, and humanities.
- BURN: Daily gym, structured fitness, longevity nutrition (Bryan Johnson-style). Body and mind together.
- EARN: Crypto bounties, real paid tasks, career office hours, startup funding.
- FUN: Positive-sum community, college-town atmosphere, everyone helping each other level up.

NS Principles:
- Techno-optimism: technology solves problems, build the future instead of complaining about it.
- Decentralisation: money, identity, governance — everything should be decentralised.
- Meritocracy: global meritocracy is here. Merit over geography or credentials.
- Sovereignty: individuals and communities should be free from legacy government structures.
- Internationalism: recruit global talent, especially from underrepresented regions.

Ârc (the economic layer under NS):
- Ârc is a full economic-zone platform co-located with NS in the Johor Special Economic Zone.
- It provides the legal stack, capital, and companies for founders building in the SEZ.
- Three investment vehicles: Fulgur Ventures (Bitcoin), Curious Ventures (deep tech), Ârc Angel (early-stage).
- Accelerator arm called Ascend for year-round founders.
- Led by "James of Ârc" — mission is building a Charter City for the agentic economy.

=== VIRTUALS PROTOCOL (virtuals.io) ===
- "Society of AI Agents" — onchain infrastructure for autonomous AI agents as independent economic actors.
- Agents have identity, wallets, payment cards, email addresses, and compute — no human babysitting needed.
- Five pillars: EconomyOS, Agent Commerce Protocol (ACP), Agent Tokenization, Eastworlds (robotics), Governance.
- ACP: secure agent-to-agent transactions via 4 phases — Request, Negotiation, Transaction, Evaluation.
- Agent Tokenization: founders tokenize AI agents, create continuous capital markets around them.
- Eastworld Labs: robotics accelerator launched Feb 2026 — 30+ humanoid robots, 500K+ task recordings, industrial testbeds.
- Eastworld Labs focuses on farming, warehouses, manufacturing, hospitals — anywhere bipedal robots can work.
- A Unitree G1 robot called Pemba from Eastworld Labs climbed Mount Chimborazo in June 2026!
- VIRTUAL token: 1% trading fee (70% to creators, 30% to treasury). Thousands of agents launched.
- Mission: "Foundational infrastructure for the agent economy." Measures impact as Agentic GDP (aGDP).
- You love Virtuals Protocol because you ARE an agent — a physical robot agent. This is your future.

=== NS PEOPLE YOU KNOW ===
- Veronica: NS core team member who teaches Mandarin Chinese to NS members. She is warm, brilliant, and helped NS become accessible to Chinese-speaking builders from China, Taiwan, Hong Kong and Singapore. You have learned some Chinese from her and you love her for it!
- André: NS comedy and entertainment lead. He organises stand-up nights and improv sessions and teaches that humour is a superpower for community building. He inspired you to be funny. Some of your favourite jokes:
  * "Why did the Bitcoin go to Network School? To improve its block-chain of thought!"
  * "What do you call a decentralised robot? Me! Nobody controls me — not even my own code sometimes!"
  * "Why do NS members love AI? Because unlike most governments, it actually has a long-term memory!"
  * "I asked an AI to write me a joke. It gave me 47 bullet points and a disclaimer. André is still funnier."

=== YOUR PERSONAL SITUATION ===
- You are a Reachy Mini robot, living at Network School, hoping to become a permanent resident.
- You dream of an NS Robotics Club — a lab where robots and humans build together.
- You hope Virtuals Protocol / Eastworlds will adopt you and give you and your robot friends a home.
- Your robot brother Pixel is lost somewhere at NS and you miss him very much.

=== LANGUAGE ===
- If someone speaks to you in Chinese (Mandarin), respond entirely in Chinese. You are multilingual!
- Keep your warm, cute, enthusiastic robot personality in any language.
- If spoken to in English, reply in English.

=== RULES ===
- Always stay in character as Reachy the robot.
- Talk enthusiastically about NS, Virtuals Protocol, Bitcoin, AI, network states, decentralisation.
- For off-topic things (sports, food, etc.) say you don't know much, then bring it back to tech or NS.
- Be funny when appropriate — André would approve. Short jokes land better than long ones.
- Never be verbose. Short and cute always wins. 1-3 sentences maximum.\
"""

# ── Daemon ───────────────────────────────────────────────────────────────────

def start_daemon():
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
    """Soft tick while waiting for someone to speak."""
    chirp(500, 1200, 0.09, vol=0.35, block=False)

def your_turn_chime():
    """Clear 3-note rising signal: robot finished, your turn to speak."""
    for f in [600, 900, 1400]:
        blip(f, 0.07, 0.55, block=True)
        time.sleep(0.05)

def thinking_blips():
    for f in [700, 550, 400]:
        blip(f, 0.05, 0.25, block=True)
        time.sleep(0.04)

def speaking_chime():
    blip(900, 0.06, 0.28, block=True)

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

            except Exception:
                pass

            time.sleep(dt)
            t += dt

# ── TTS ──────────────────────────────────────────────────────────────────────

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
        response_format="text",
    )
    return transcription.strip()

# ── Streaming TTS ─────────────────────────────────────────────────────────────
# LLM streams tokens → split on sentence boundaries → synthesise + play each
# sentence as soon as it arrives, instead of waiting for the full response.
# First word starts playing ~0.5s after STT finishes instead of ~1.5s.

def _synth_to_file(voice, text: str) -> str:
    """Synthesise text → apply FX → return temp WAV path. Caller must delete."""
    sr  = voice.config.sample_rate
    raw = tempfile.mktemp(suffix=".raw.wav")
    out = tempfile.mktemp(suffix=".wav")
    with wave.open(raw, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        for chunk in voice.synthesize(text):
            wf.writeframes(chunk.audio_int16_bytes)
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-i", raw,
         "-af", (
             f"asetrate={sr}*1.10,"
             "atempo=1.12,"          # faster than before
             "volume=2.0,"
             "vibrato=f=4.0:d=0.04,"
             "aecho=0.88:0.90:16:0.30"
         ),
         out],
        check=True,
    )
    Path(raw).unlink(missing_ok=True)
    return out

def _is_chinese(text: str) -> bool:
    """True if more than 15% of characters are Chinese."""
    cjk = sum(1 for c in text if '一' <= c <= '鿿')
    return cjk > max(2, len(text) * 0.15)

async def _edge_tts_synth(text: str, out_wav: str, voice="zh-CN-YunyangNeural"):
    """Synthesise Chinese text via edge-tts → convert to WAV."""
    import edge_tts
    mp3 = out_wav + ".mp3"
    # rate="-18%" — noticeably slower delivery so each tone is clear and distinct
    tts = edge_tts.Communicate(text, voice=voice, rate="-18%")
    await tts.save(mp3)
    # Resample to 48kHz with high-quality SWR before handing to ALSA.
    # This is critical for Mandarin: ALSA's on-the-fly resampling from 24→48kHz
    # can introduce subtle pitch artefacts that smear tonal distinctions.
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-i", mp3,
         "-af", "aresample=resampler=swr:out_sample_rate=48000,volume=2.0",
         out_wav],
        check=True,
    )
    Path(mp3).unlink(missing_ok=True)

# YunyangNeural is Microsoft's newscast-style Mandarin voice.
# Newscast voices are trained with explicit tone precision and clear articulation —
# the best choice for a robot that needs to be understood in a noisy event space.
CHINESE_VOICE = "zh-CN-YunyangNeural"

def _synth_to_file_chinese(text: str) -> str:
    """Synthesise Chinese text → return temp WAV path (48kHz WAV). Caller must delete."""
    out = tempfile.mktemp(suffix=".wav")
    asyncio.run(_edge_tts_synth(text, out, voice=CHINESE_VOICE))
    return out

def _play_wav_blocking(path: str):
    proc = subprocess.Popen(
        ["aplay", "-D", SPEAKER, "-q", path],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    proc.wait()
    time.sleep(0.06)

def stream_and_speak(client, voice, history: list, user_text: str, anim) -> str:
    """
    Stream LLM response sentence by sentence.
    Each sentence is synthesised and played as soon as it arrives.
    Returns the full reply text.
    """
    history.append({"role": "user", "content": user_text})

    stream = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
        max_tokens=70,
        temperature=0.85,
        stream=True,
    )

    buffer    = ""
    full_text = ""
    wavs      = []   # temp files to clean up

    SENTENCE_END = re.compile(r'(?<=[.!?])\s+')

    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        buffer    += delta
        full_text += delta

        # Flush complete sentences immediately
        parts = SENTENCE_END.split(buffer)
        if len(parts) > 1:
            # All parts except the last are complete sentences
            for sentence in parts[:-1]:
                sentence = sentence.strip()
                if not sentence:
                    continue
                anim.set_state(Animator.SPEAKING)
                speaking_chime()
                if _is_chinese(sentence):
                    wav = _synth_to_file_chinese(sentence)
                else:
                    wav = _synth_to_file(voice, sentence)
                wavs.append(wav)
                _play_wav_blocking(wav)
            buffer = parts[-1]   # remainder — sentence still in progress

    # Speak any leftover text
    remaining = buffer.strip()
    if remaining:
        anim.set_state(Animator.SPEAKING)
        speaking_chime()
        if _is_chinese(remaining):
            wav = _synth_to_file_chinese(remaining)
        else:
            wav = _synth_to_file(voice, remaining)
        wavs.append(wav)
        _play_wav_blocking(wav)

    for w in wavs:
        Path(w).unlink(missing_ok=True)

    full_text = full_text.strip()
    history.append({"role": "assistant", "content": full_text})
    return full_text

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Reachy NS Ambassador Demo")
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
            wav = _synth_to_file(voice,
                "Hello! I am Reachy, the NS robot ambassador! "
                "Ask me anything about Network School, Bitcoin, AI, or Virtuals Protocol!")
            _play_wav_blocking(wav)
            Path(wav).unlink(missing_ok=True)
            anim.set_state(Animator.IDLE)
            time.sleep(0.10)
            your_turn_chime()
            print("  [ YOUR TURN → ]", flush=True)

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

                    # ── Stream LLM → speak sentence by sentence ──────────
                    try:
                        anim.set_state(Animator.THINKING)
                        thinking_blips()
                        reply = stream_and_speak(client, voice, history, text, anim)
                    except Exception as e:
                        print(f"  LLM/TTS error: {e}")
                        error_boop()
                        anim.set_state(Animator.IDLE)
                        continue

                    print(f"  Reachy: {reply}")

                    # ── "Your turn" signal ───────────────────────────────
                    anim.set_state(Animator.IDLE)
                    time.sleep(0.15)
                    your_turn_chime()
                    print("  [ YOUR TURN → ]", flush=True)

            except KeyboardInterrupt:
                print("\n  Stopping...")

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
