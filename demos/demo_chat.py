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
import sys
import time
from pathlib import Path

from groq import Groq
from silero_vad import load_silero_vad

from reachy_mini import ReachyMini

from reachy_demo.animator import Animator
from reachy_demo.audio import (
    blip, boot_beeps, error_chime, pcm_to_wav_bytes, record_utterance,
    thinking_blips,
)
from reachy_demo.daemon import start_daemon, stop_daemon
from reachy_demo.groq_client import load_api_key, transcribe
from reachy_demo.text import clean_for_tts
from reachy_demo.tts_piper import load_voice, synth_and_play

ROOT       = Path(__file__).parent.parent
VOICE_PATH = str(ROOT / "voices" / "en_US-amy-medium.onnx")

GROQ_KEY = load_api_key(ROOT)
if not GROQ_KEY:
    sys.exit("ERROR: GROQ_API_KEY not found in .env or environment")

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

# ── Demo-specific sound effects ───────────────────────────────────────────────
# speaking_chime differs from the common version (two blips instead of one)

def speaking_chime():
    blip(900, 0.07, 0.30, block=True)
    time.sleep(0.06)
    blip(1300, 0.06, 0.24, block=True)

# ── LLM (non-streaming) ───────────────────────────────────────────────────────

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
    voice  = load_voice(VOICE_PATH)
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
                        text = transcribe(client, pcm_to_wav_bytes(pcm), language="en")
                    except Exception as e:
                        print(f"  STT error: {e}")
                        error_chime()
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
                        error_chime()
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
        stop_daemon(daemon_proc)


if __name__ == "__main__":
    main()
