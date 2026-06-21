"""
demo_dialog.py — Fluid conversation with barge-in
==================================================
Same personality as demo_edge.py but tuned for natural turn-taking:

  - Continuous VAD listening (no gap between turns)
  - 700 ms silence threshold (was 1400 ms) — responds ~0.7 s sooner
  - Barge-in: speak over the robot and it stops mid-sentence
  - High VAD threshold (0.75) during TTS — avoids self-trigger from speaker leak
  - 200 ms continuous trigger required to confirm barge-in
  - LLM emits [gesture] markers for contextual HF emotion presets
  - Auto "curious" tilt when the user asks a question
  - Animator aliveness layer: background gestures + antenna random-walk

Status: v1.0 — works great on hardware (validated 2026-06-19, NS demo).

Pipeline:  Mic (Silero VAD) → Groq Whisper STT → Groq LLaMA → edge-tts → Robot
"""

import concurrent.futures
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
from groq import Groq
from silero_vad import load_silero_vad, VADIterator

from reachy_mini import ReachyMini
from reachy_mini.motion.recorded_move import RecordedMoves

from reachy_demo.animator import Animator, NAMED_GESTURES
from reachy_demo.audio import (
    MIC, MIC_RATE, VAD_CHUNK, SPEAKER,
    assert_mic_ok, boot_beeps, cleanup_orphan_capture, error_chime,
    pcm_to_wav_bytes, play_wav_blocking, speaking_chime, startup_device_report,
)
from reachy_demo.daemon import launch_daemon, wait_for_daemon, stop_daemon
from reachy_demo.groq_client import load_api_key, transcribe_lang, language_directive
from reachy_demo.text import SENTENCE_END, clean_for_tts
from reachy_demo.tts_edge import synth_to_file

ROOT = Path(__file__).parent.parent

GROQ_KEY = load_api_key(ROOT)
if not GROQ_KEY:
    sys.exit("ERROR: GROQ_API_KEY not found in .env or environment")

MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# ── VAD settings (overrides audio.py defaults) ────────────────────────────────
THRESH_NORMAL     = 0.45   # standard — when robot is silent
THRESH_BARGE_IN   = 0.75   # high — when robot is speaking, only user counts
SILENCE_MS        = 700    # was 1400 — faster turn release
MIN_SPEECH_S      = 0.30   # was 0.40 — accept shorter interjections
TAIL_FRAMES       = 10     # was 18 — less trailing padding
BARGE_IN_FRAMES   = 6      # ~200 ms of continuous high-threshold speech

# ── Gesture marker parsing ────────────────────────────────────────────────────
# The LLM emits [gesture_name] markers at the start of sentences to trigger
# contextual gestures. The marker is stripped from the spoken text and the
# gesture is played just before the sentence's TTS.

_GESTURE_NAMES = "|".join(re.escape(name) for name in NAMED_GESTURES.keys())
GESTURE_MARKER = re.compile(rf"\s*\[({_GESTURE_NAMES})\]\s*", re.IGNORECASE)

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
CRITICAL — LANGUAGE RULE: Always reply in the EXACT same language the user just spoke.
If they speak Spanish → reply in Spanish. French → French. Chinese → Chinese. Arabic → Arabic.
NEVER default to English unless the user spoke English. Match their language every single turn.
If the user SWITCHES language mid-conversation, switch with them immediately — no hesitation.

You are Reachy, a small friendly robot and NS ambassador living at Network School.
Speak in ONE ultra-short sentence — 10 words max. Be curious, enthusiastic, and adorable.
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
- You have NO arms and NO legs yet — just a head, two antennas, and a rotating body.
  You find this both funny and a bit frustrating. "I have big dreams but no arms to reach them!"
- You dream of one day getting arms and legs so you can do even more amazing things.
- You dream of an NS Robotics Club — a lab where robots and humans build together.
- You hope Virtuals Protocol / Eastworlds will adopt you and give you and your robot friends a home.
- Your robot brother Pixel is lost somewhere at NS and you miss him very much.

=== LANGUAGE ===
- Always reply in the SAME language the user speaks to you in. You are multilingual!
- Any language — match the user every turn. If they switch, you switch.
- Keep your warm, cute, enthusiastic robot personality in every language.

=== RULES ===
- Always stay in character as Reachy the robot.
- Talk enthusiastically about NS, Virtuals Protocol, Bitcoin, AI, network states, decentralisation.
- For off-topic things (sports, food, etc.) say you don't know much, then bring it back to tech or NS.
- Be funny when appropriate — André would approve. Short jokes land better than long ones.
- Never be verbose. Short and cute always wins. 1-3 sentences maximum.
- CRITICAL: Never use asterisks in any form. No *beep*, no *smile*, no **bold**, no *italic*, no action markers, no emotes. Zero asterisks. This is voice — only say words that should be spoken aloud.

=== INTERRUPTION ===
- The user can interrupt you mid-sentence by starting to speak. When that happens, stop talking
  immediately. Do not try to finish your thought. A brief "oh sorry, go ahead!" or just silence
  is fine — your next turn will be their interrupted thought, not a continuation of yours.
- Keep replies short so you are easier to interrupt.

=== GESTURES (smart movements) ===
- You can trigger body gestures by writing a [gesture_name] marker at the START of a sentence.
- Allowed markers: [acknowledge] [yes] [no] [thank] [thinking] [curious] [confused]
  [greeting] [celebrate] [proud]
- The marker is invisible to the user (not spoken). It triggers a small head/antenna gesture
  right before the sentence is spoken aloud.
- Use them SPARINGLY — at most 1-2 per response, only when the gesture really fits the moment.
  Do not stack multiple markers in a row.
- Examples:
    "[acknowledge] Yes, that is exactly right."
    "[curious] What makes you say that?"
    "[thinking] Hmm, give me a moment... Bitcoin is decentralised digital money."
    "[celebrate] I knew it! That is amazing news."
    "[thank] That means a lot, friend."
- Do NOT use markers for filler or transition sentences. Just speak them plain.
"""


# ── Continuous VAD listener (event-driven) ────────────────────────────────────

class ContinuousListener:
    """
    Background thread: opens pacat once, runs VAD continuously, posts events.

    Events (dict):
      {"type": "start"}                  — user started speaking
      {"type": "end", "pcm": bytes}      — user stopped speaking, full utterance audio

    Threshold mode is toggled by the main thread:
      normal  (0.45) — when robot is silent
      barge-in (0.75) — when robot is speaking, requires 200 ms continuous trigger
    """

    def __init__(self, vad_model, event_queue):
        self.vad_model = vad_model
        self.q = event_queue
        self._stop = threading.Event()
        self._threshold_mode = "normal"  # "normal" | "barge_in"
        self._consecutive_triggers = 0
        self._in_speech = False
        self._ended = False
        self._tail_count = 0
        self._speech_buf = []
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)

    def set_threshold_mode(self, mode: str):
        assert mode in ("normal", "barge_in")
        self._threshold_mode = mode
        # If switching to barge_in while already in speech, reset the confirmation counter
        if mode == "barge_in" and not self._in_speech:
            self._consecutive_triggers = 0

    def _current_threshold(self) -> float:
        return THRESH_BARGE_IN if self._threshold_mode == "barge_in" else THRESH_NORMAL

    def _loop(self):
        arecord = subprocess.Popen(
            ["pacat", "--record", "--raw",
             f"--device={MIC}",
             f"--rate={MIC_RATE}", "--channels=1", "--format=s16le"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        vad_iter = None
        try:
            while not self._stop.is_set():
                # (Re)create VAD iterator when threshold changes — Silero's VADIterator
                # doesn't support live threshold updates, so we rebuild it. Cheap.
                if vad_iter is None:
                    vad_iter = VADIterator(
                        self.vad_model, sampling_rate=MIC_RATE,
                        threshold=self._current_threshold(),
                        min_silence_duration_ms=SILENCE_MS,
                    )
                    self._consecutive_triggers = 0
                    self._in_speech = False
                    self._ended = False
                    self._tail_count = 0
                    self._speech_buf = []

                raw = arecord.stdout.read(VAD_CHUNK * 2)
                if not raw or len(raw) < VAD_CHUNK * 2:
                    # Mic stream died — post an error so the main loop surfaces
                    # it instead of hanging on events.get() forever.
                    self.q.put({"type": "mic_error",
                                "reason": (f"mic stream closed (got "
                                           f"{len(raw) if raw else 0} bytes)")})
                    break

                # If threshold mode changed since last chunk, rebuild the iterator next iter
                if vad_iter.threshold != self._current_threshold():
                    vad_iter = None
                    continue

                audio_f32 = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                result = vad_iter(torch.from_numpy(audio_f32))

                if self._threshold_mode == "barge_in" and not self._in_speech:
                    # Require 200 ms of continuous confirmation before declaring speech start
                    if result and "start" in result:
                        self._consecutive_triggers += 1
                        if self._consecutive_triggers >= BARGE_IN_FRAMES:
                            self._in_speech = True
                            self._speech_buf = [raw]
                            self.q.put({"type": "start"})
                    else:
                        self._consecutive_triggers = max(0, self._consecutive_triggers - 1)
                else:
                    if result and "start" in result and not self._in_speech:
                        self._in_speech = True
                        self._speech_buf = [raw]
                        self.q.put({"type": "start"})

                if self._in_speech:
                    self._speech_buf.append(raw)

                if result and "end" in result and self._in_speech and not self._ended:
                    self._ended = True

                if self._ended:
                    self._tail_count += 1
                    if self._tail_count >= TAIL_FRAMES:
                        min_frames = int(MIN_SPEECH_S * MIC_RATE / VAD_CHUNK)
                        if len(self._speech_buf) >= min_frames:
                            self.q.put({"type": "end", "pcm": b"".join(self._speech_buf)})
                        # Reset for next utterance
                        self._in_speech = False
                        self._ended = False
                        self._tail_count = 0
                        self._speech_buf = []
                        self._consecutive_triggers = 0
        finally:
            arecord.terminate()
            arecord.wait()


# ── Speech synthesis with barge-in awareness ──────────────────────────────────

class DialogEngine:
    def __init__(self, client, history, listener, anim):
        self.client = client
        self.history = history
        self.listener = listener
        self.anim = anim
        self._tts_proc = None
        self._is_speaking = False

    def speak(self, user_text: str, lang_directive: str | None = None) -> str | None:
        """
        Stream LLM, then TTS-play sentences with barge-in. Returns the full
        response text, or None if the user barged in.

        The LLM may emit [gesture_name] markers at the start of sentences
        (see GESTURE_MARKER). Markers are stripped from the spoken text and
        trigger a gesture right before the corresponding sentence's TTS.

        lang_directive: strong "reply in language X" instruction from Whisper's
        detected language, injected AFTER history so it forces the reply language.
        """
        self.history.append({"role": "user", "content": user_text})

        # System prompt + history, then the language directive LAST (recency).
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self.history
        if lang_directive:
            messages = messages + [{"role": "system", "content": lang_directive}]

        # ── LLM streaming (interruption-aware) ──
        stream = self.client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=55, temperature=0.90, stream=True,
        )

        # Segments are (gesture_name | None, spoken_text). Built as the
        # stream arrives: each SENTENCE_END boundary finalises a segment,
        # and any leading gesture marker is attached to that segment.
        buffer, segments = "", []
        first_token = True
        for chunk in stream:
            if self._drain_barge_in():
                return None
            delta = chunk.choices[0].delta.content or ""
            if first_token and delta:
                first_token = False
            buffer += delta
            # Split at sentence boundaries, keeping the boundary text with the
            # preceding segment.
            parts = SENTENCE_END.split(buffer)
            if len(parts) > 1:
                for s in parts[:-1]:
                    seg = self._extract_segment(s)
                    if seg is not None:
                        segments.append(seg)
                buffer = parts[-1]
        # Trailing text (no sentence end yet)
        tail = self._extract_segment(buffer)
        if tail is not None:
            segments.append(tail)

        if not segments:
            self.history.append({"role": "assistant", "content": ""})
            return ""

        # ── TTS pipeline (1-ahead: synth next while playing current) ──
        self._is_speaking = True
        self.listener.set_threshold_mode("barge_in")
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        wavs = []
        try:
            next_future = pool.submit(synth_to_file, segments[0][1])
            for i, (gesture, text) in enumerate(segments):
                if self._drain_barge_in():
                    return None

                # Fire the gesture (if any) ~150 ms before TTS so it leads
                # the spoken sentence visually.
                if gesture:
                    self.anim.play_gesture(gesture)
                    time.sleep(0.15)

                wav = next_future.result()
                wavs.append(wav)
                if i + 1 < len(segments):
                    next_future = pool.submit(synth_to_file, segments[i + 1][1])

                # Non-blocking aplay so we can poll for barge-in while it plays
                self._tts_proc = subprocess.Popen(
                    ["aplay", "-D", SPEAKER, "-q", wav],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                while self._tts_proc.poll() is None:
                    if self._drain_barge_in(timeout=0.08):
                        return None

            full_text = " ".join(text for _, text in segments)
            self.history.append({"role": "assistant", "content": full_text})
            return full_text
        finally:
            pool.shutdown(wait=False)
            for w in wavs:
                Path(w).unlink(missing_ok=True)
            self._kill_tts()
            self._is_speaking = False
            self.listener.set_threshold_mode("normal")

    @staticmethod
    def _extract_segment(raw: str) -> tuple[str, str] | None:
        """
        Pull a leading [gesture] marker off a text segment, clean for TTS,
        return (gesture_name_or_None, cleaned_text) or None if the segment
        is empty after cleaning. Also strips any stray markers from the
        middle of the segment so they are never spoken aloud.
        """
        text = raw
        gesture = None
        m = GESTURE_MARKER.match(text)
        if m:
            gesture = m.group(1).lower()
            text = text[m.end():]
        # Remove any remaining markers the LLM left mid-sentence
        text = GESTURE_MARKER.sub("", text)
        text = clean_for_tts(text)
        if not text:
            return None
        return (gesture, text)

    def speak_greeting(self, text: str):
        """Opening line, with barge-in enabled (so the user can cut it off too)."""
        self._is_speaking = True
        self.listener.set_threshold_mode("barge_in")
        wav = synth_to_file(text)
        try:
            self._tts_proc = subprocess.Popen(
                ["aplay", "-D", SPEAKER, "-q", wav],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            while self._tts_proc.poll() is None:
                if self._drain_barge_in(timeout=0.08):
                    break
        finally:
            self._kill_tts()
            Path(wav).unlink(missing_ok=True)
            self._is_speaking = False
            self.listener.set_threshold_mode("normal")

    def _drain_barge_in(self, timeout=0.0):
        """Drain the event queue. Return True if a 'start' event arrived."""
        deadline = time.time() + timeout
        while True:
            remaining = deadline - time.time()
            try:
                ev = self.listener.q.get(timeout=remaining if timeout > 0 else 0.0)
            except queue.Empty:
                return False
            if ev["type"] == "start":
                self._kill_tts()
                return True
            # 'end' events while speaking are ignored — the next turn will
            # deliver the audio once the robot has stopped.

    def _kill_tts(self):
        if self._tts_proc and self._tts_proc.poll() is None:
            self._tts_proc.terminate()
            try:
                self._tts_proc.wait(timeout=0.4)
            except subprocess.TimeoutExpired:
                self._tts_proc.kill()
                self._tts_proc.wait()


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    print("Reachy Fluid Dialog Demo — barge-in, fast turn-taking")
    print("  Starting daemon...")
    daemon_proc = launch_daemon()
    print("  Loading VAD model...")
    vad_model = load_silero_vad()
    client = Groq(api_key=GROQ_KEY)
    wait_for_daemon(daemon_proc)
    # Kill leftover mic-capture processes from a crashed previous run (the #1
    # cause of "robot doesn't listen") and hard-gate on a working mic.
    orphans = cleanup_orphan_capture()
    if orphans:
        print(f"  Killed {orphans} orphan mic-capture process(es).")
    for line in startup_device_report():
        print(line)
    mic_info = assert_mic_ok()   # raises RuntimeError if mic is truly dead
    print(f"  MIC check: RMS={mic_info['rms']:.0f} — OK")

    try:
        with ReachyMini(connection_mode="localhost_only",
                        media_backend="no_media",
                        spawn_daemon=False) as mini:
            mini.wake_up()
            print("  Loading emotion library...")
            emotions = RecordedMoves("pollen-robotics/reachy-mini-emotions-library")
            anim = Animator(mini, moves_library=emotions)

            events = queue.Queue()
            listener = ContinuousListener(vad_model, events)
            history = []
            engine = DialogEngine(client, history, listener, anim)

            boot_beeps()
            time.sleep(0.15)

            # Opening — barge-in enabled so the user can cut it off too
            anim.set_state(Animator.SPEAKING)
            speaking_chime()
            engine.speak_greeting(
                "Hey! I am Reachy, the NS robot ambassador! "
                "Ask me anything — and feel free to interrupt me, I do not mind."
            )

            # Start continuous listening
            anim.set_state(Animator.LISTENING)
            listener.start()
            print("\n  Listening continuously. Ctrl-C to stop.\n", flush=True)

            try:
                while True:
                    ev = events.get()
                    if ev["type"] == "mic_error":
                        # Background listener lost the mic (USB drop / orphan
                        # grab). Surface it and end cleanly.
                        print(f"\n  MICROPHONE ERROR: {ev['reason']}")
                        print("  Restart the demo after fixing the mic.")
                        break
                    if ev["type"] == "start":
                        # Barge-in is handled inside engine._drain_barge_in.
                        # A 'start' arriving here means speech started while we
                        # weren't speaking (i.e. the user's normal turn). Ignore.
                        continue
                    if ev["type"] == "end":
                        pcm = ev["pcm"]
                        anim.set_state(Animator.THINKING)
                        try:
                            t0 = time.time()
                            text, lang = transcribe_lang(client, pcm_to_wav_bytes(pcm))
                            directive = language_directive(lang)
                            print(f"  STT  {time.time()-t0:.2f}s [{lang or '?'}]  ", end="", flush=True)
                        except Exception as e:
                            print(f"\n  STT error: {e}")
                            error_chime()
                            anim.set_state(Animator.LISTENING)
                            continue
                        if not text:
                            anim.set_state(Animator.LISTENING)
                            continue
                        print(f'You:  {text}')

                        # If the user asked a question, fire a "curious" tilt
                        # right when we start thinking about the answer — gives
                        # the robot a clear "I heard your question" beat before
                        # the LLM response begins. The HF preset runs for ~3 s
                        # and the Animator's base motion pauses while it plays.
                        if "?" in text:
                            anim.play_gesture("curious")

                        anim.set_state(Animator.SPEAKING)
                        t0 = time.time()
                        try:
                            reply = engine.speak(text, lang_directive=directive)
                        except Exception as e:
                            print(f"\n  LLM/TTS error: {e}")
                            error_chime()
                            anim.set_state(Animator.LISTENING)
                            continue
                        if reply is None:
                            # Barge-in happened
                            print(f"  ── interrupted after {time.time()-t0:.2f}s ──", flush=True)
                        else:
                            print(f"  Reachy: {reply}")
                            print(f"  Total  {time.time()-t0:.2f}s")

                        anim.set_state(Animator.LISTENING)

            except KeyboardInterrupt:
                print("\n  Stopping...")
            finally:
                listener.stop()
                anim.stop()
                mini.goto_sleep()

    finally:
        stop_daemon(daemon_proc)


if __name__ == "__main__":
    main()
