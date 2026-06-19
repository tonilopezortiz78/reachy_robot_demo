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
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from groq import Groq
from silero_vad import load_silero_vad

from reachy_mini import ReachyMini

from reachy_demo.animator import Animator
from reachy_demo.audio import (
    boot_beeps, error_chime, pcm_to_wav_bytes, play_wav_blocking,
    record_utterance, speaking_chime, thinking_blips, your_turn_chime,
)
from reachy_demo.daemon import launch_daemon, wait_for_daemon, stop_daemon
from reachy_demo.groq_client import load_api_key, stream_chat, transcribe
from reachy_demo.text import SENTENCE_END, clean_for_tts
from reachy_demo.tts_piper import load_voice, synth_to_file

ROOT       = Path(__file__).parent.parent
VOICE_PATH = str(ROOT / "voices" / "en_US-amy-medium.onnx")

GROQ_KEY = load_api_key(ROOT)
if not GROQ_KEY:
    sys.exit("ERROR: GROQ_API_KEY not found in .env or environment")

MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

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

=== QUANTUS PROTOCOL (quantus.com) ===
- Quantus is a Layer 1 blockchain built from scratch to be quantum-resistant — "the blockchain quantum computers can't break."
- The problem: Bitcoin, Ethereum, Solana all use ECDSA-256 / ED25519 cryptography, which a quantum computer with ~2,400 logical qubits could break. The crypto industry has $3 trillion with zero quantum defence. "Q-Day" is the name for that moment.
- Quantus uses ML-DSA Dilithium-5 — NIST FIPS 204 certified post-quantum cryptography. The only chain built quantum-safe from day one.
- Technology: recursive STARKs for scalability (50 QTPS), 247 active validators, mobile wallet in public beta (iOS/Android), Q-Day Checker tool.
- You find this fascinating and a little scary — even your own little robot wallet could be vulnerable on Q-Day! Quantus is building the defence.

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
- Never be verbose. Short and cute always wins. 1-3 sentences maximum.
- CRITICAL: Never use asterisks in any form. No *beep*, no *smile*, no **bold**, no *italic*, no action markers, no emotes. Zero asterisks. This is voice — only say words that should be spoken aloud.\
"""

# ── Chinese TTS (one-shot asyncio.run — used only inside stream_and_speak) ────

CHINESE_VOICE = "zh-CN-YunyangNeural"


def _is_chinese(text: str) -> bool:
    """True if more than 15% of characters are Chinese."""
    cjk = sum(1 for c in text if '一' <= c <= '鿿')
    return cjk > max(2, len(text) * 0.15)


async def _edge_tts_synth(text: str, out_wav: str, voice=CHINESE_VOICE):
    """Synthesise Chinese text via edge-tts → convert to WAV."""
    import edge_tts
    mp3 = out_wav + ".mp3"
    # rate="-18%" — noticeably slower delivery so each tone is clear and distinct
    tts = edge_tts.Communicate(text, voice=voice, rate="-18%")
    await asyncio.wait_for(tts.save(mp3), timeout=10.0)
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


def _synth_to_file_chinese(text: str) -> str:
    """Synthesise Chinese text → return temp WAV path (48kHz WAV). Caller must delete."""
    out = tempfile.mktemp(suffix=".wav")
    asyncio.run(_edge_tts_synth(text, out, voice=CHINESE_VOICE))
    return out

# ── Streaming TTS ─────────────────────────────────────────────────────────────
# LLM streams tokens → split on sentence boundaries → synthesise + play each
# sentence as soon as it arrives, instead of waiting for the full response.
# First word starts playing ~0.5s after STT finishes instead of ~1.5s.

def stream_and_speak(client, voice, history: list, user_text: str, anim) -> str:
    history.append({"role": "user", "content": user_text})

    buffer     = ""
    full_text  = ""
    wavs       = []
    spoke      = False
    t_start    = time.time()
    first_tok  = True
    n_sentence = 0

    try:
        for delta in stream_chat(client, history, MODEL, SYSTEM_PROMPT):
            if delta and first_tok:
                print(f"  LLM  {time.time()-t_start:.2f}s (first token)", flush=True)
                first_tok = False
            buffer    += delta
            full_text += delta

            parts = SENTENCE_END.split(buffer)
            if len(parts) > 1:
                for sentence in parts[:-1]:
                    sentence = clean_for_tts(sentence)
                    if not sentence:
                        continue
                    n_sentence += 1
                    anim.set_state(Animator.SPEAKING)
                    if not spoke:
                        speaking_chime()
                        spoke = True
                    t1 = time.time()
                    if _is_chinese(sentence):
                        wav = _synth_to_file_chinese(sentence)
                    else:
                        wav = synth_to_file(voice, sentence)
                    print(f"  Piper[{n_sentence}] {time.time()-t1:.2f}s", flush=True)
                    wavs.append(wav)
                    play_wav_blocking(wav)
                buffer = parts[-1]

        remaining = clean_for_tts(buffer)
        if remaining:
            n_sentence += 1
            anim.set_state(Animator.SPEAKING)
            if not spoke:
                speaking_chime()
            t1 = time.time()
            if _is_chinese(remaining):
                wav = _synth_to_file_chinese(remaining)
            else:
                wav = synth_to_file(voice, remaining)
            print(f"  Piper[{n_sentence}] {time.time()-t1:.2f}s", flush=True)
            wavs.append(wav)
            play_wav_blocking(wav)

        full_text = full_text.strip()
        history.append({"role": "assistant", "content": full_text})
        if len(history) > 8:
            history[:] = history[-8:]
        print(f"  Total  {time.time()-t_start:.2f}s", flush=True)
        return full_text
    finally:
        for w in wavs:
            Path(w).unlink(missing_ok=True)

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Reachy NS Ambassador Demo")
    print("  Starting daemon...")
    daemon_proc = launch_daemon()           # non-blocking — starts in background
    print("  Loading voice + VAD...")
    voice     = load_voice(VOICE_PATH)     # ~1 s
    vad_model = load_silero_vad()          # ~2 s  } both overlap with daemon startup
    client    = Groq(api_key=GROQ_KEY)
    wait_for_daemon(daemon_proc)           # wait for remainder (usually already up)

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
            wav = synth_to_file(voice,
                "Hello! I am Reachy, the NS robot ambassador! "
                "Ask me anything about Network School, Bitcoin, AI, or Virtuals Protocol!")
            play_wav_blocking(wav)
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
                        t0 = time.time()
                        text = transcribe(client, pcm_to_wav_bytes(pcm))
                        print(f"  STT  {time.time()-t0:.2f}s", flush=True)
                    except Exception as e:
                        print(f"  STT error: {e}")
                        error_chime()
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
                        error_chime()
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
            finally:
                anim.stop()
                mini.goto_sleep()

    finally:
        stop_daemon(daemon_proc)


if __name__ == "__main__":
    main()
