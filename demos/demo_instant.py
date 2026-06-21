"""
demo_instant.py — Reachy NS Ambassador: STREAMING TTS for near-instant replies
==============================================================================
Same Groq pipeline as demo_tools7 (menu 4), but the TTS is STREAMED: edge-tts
audio chunks are piped straight to the speaker as they arrive, instead of
synthesising the whole sentence to a file first. Time-to-first-audio drops from
~3.6s to ~0.4s per sentence — the robot starts talking almost the instant the
LLM produces a sentence.

Inspired by xiaozhi-esp32's streaming-everything design (github.com/78/xiaozhi-esp32):
the win is never waiting for a full stage to finish before the next one starts.

Pipeline (all streaming, overlapped):
  VAD → Groq Whisper STT → Groq LLaMA stream → per-sentence STREAMING edge-tts
  A producer thread extracts sentences from the LLM stream; the consumer streams
  each one to the speaker the moment it's ready, so sentence 1 is already playing
  while 2 and 3 are still being generated.

Voice: en-US-AvaMultilingualNeural — PITCH +48Hz, multilingual (same as demo 4/5).
Run:   ./run.sh demos/demo_instant.py
Press Ctrl-C to stop.
"""

import concurrent.futures
import queue
import random
import re
import sys
import threading
import time
from pathlib import Path

from groq import Groq
from silero_vad import load_silero_vad

from reachy_mini import ReachyMini
from reachy_mini.motion.recorded_move import RecordedMoves

from reachy_demo.animator import Animator, NAMED_GESTURES
from reachy_demo.dance import DANCE_KEYWORDS, do_macarena
from reachy_demo.search import web_search
from reachy_demo.listener import ContinuousListener
from reachy_demo.audio import (
    MIC_RATE,
    boot_beeps, cleanup_orphan_capture, ensure_mic_working, error_chime,
    voice_filter_pcm, pcm_to_wav_bytes, speaking_chime, startup_device_report,
)
from reachy_demo.cues import speak_cue, speak_thinking, prewarm, set_translator
from reachy_demo.daemon import launch_daemon, wait_for_daemon, stop_daemon
from reachy_demo.groq_client import (
    load_api_key, transcribe_lang_robust, language_directive,
    is_hallucination,
)
from reachy_demo.memory import (
    load_memories, memory_block, extract_memories, remember,
)
from reachy_demo.session_log import SessionLogger
from reachy_demo.speech_gate import is_real_speech
from reachy_demo.text import SENTENCE_END, clean_for_tts
from reachy_demo.tts_edge import stream_to_speaker  # PITCH +48Hz set in tts_edge.py

ROOT = Path(__file__).parent.parent

GROQ_KEY = load_api_key(ROOT)
if not GROQ_KEY:
    sys.exit("ERROR: GROQ_API_KEY not found in .env or environment")

# Two models, chosen by benchmarking all Groq options on this exact task
# (cute short multilingual reply with the real system prompt):
#   CHAT_MODEL   — the spoken reply. Llama 4 Scout (17B×16E, 128k): ~0.32s, natively
#                  multilingual (we meet 80+ nationalities), cute, never returns
#                  empty, emits gesture markers cleanly. Best all-rounder here.
#                  (Llama 3.3 70B is warmer/cutest if you'll trade ~0.15s + cost;
#                   Qwen3 leaks <think> reasoning; GPT-OSS is slow + sometimes blank.)
#   ACTION_MODEL — the one-word gesture picker, cue translation, and background
#                  memory extraction. Llama 3.1 8B Instant: fastest (~0.29s) and
#                  cheapest on Groq — perfect for tiny one-shot calls, and it makes
#                  the gesture fire even sooner, before the first spoken word.
CHAT_MODEL   = "meta-llama/llama-4-scout-17b-16e-instruct"   # natively multilingual, no same-script mixing
ACTION_MODEL = "llama-3.1-8b-instant"
MODEL = CHAT_MODEL   # back-compat alias (logging, etc.)

# VAD settings live in reachy_demo/listener.py (shared by all talking demos).

# Min seconds between "sorry, could you repeat?" prompts.
# 15s is conservative — even a noisy room shouldn't trigger it more than once
# every quarter-minute, and we only fire it when the visitor's language is known
# (so it's always in the right language, and never on the very first noise burst).
REPEAT_COOLDOWN_S = 15.0

# ── Gesture marker parsing ────────────────────────────────────────────────────
_GESTURE_NAMES = "|".join(re.escape(name) for name in NAMED_GESTURES.keys())
GESTURE_MARKER = re.compile(rf"\s*\[({_GESTURE_NAMES})\]\s*", re.IGNORECASE)

# ── Parallel action picker ────────────────────────────────────────────────────
# A tiny non-streaming Groq call that selects the opening physical gesture for
# each robot turn. Runs in a ThreadPoolExecutor alongside the speech stream so
# it completes in ~150ms and fires the gesture before the first word plays.

_ACTION_LIST = ", ".join(list(NAMED_GESTURES.keys()) + ["none"])

_ACTION_SYSTEM = (
    "You pick at most ONE small physical gesture for a friendly robot about to reply. "
    "Most of the time the robot should stay calm and just talk — so answer 'none' by "
    "DEFAULT. Only pick a real gesture when the moment genuinely calls for it: a clear "
    "yes/no answer, a greeting, a thank-you, a celebration, or visible curiosity. "
    "When unsure, answer 'none'. Aim for 'none' on roughly 3 out of 4 replies. "
    "Reply with exactly ONE word from this list: " + _ACTION_LIST + ". Nothing else."
)


def pick_action(client: Groq, history: list, user_text: str) -> str | None:
    """
    Parallel Groq call: returns a gesture name to play before speaking, or None.
    Low max_tokens and no streaming keeps this under ~150ms on Groq.
    """
    messages = [
        {"role": "system", "content": _ACTION_SYSTEM},
        *history,
        {"role": "user", "content": user_text},
    ]
    try:
        resp = client.chat.completions.create(
            model=ACTION_MODEL, messages=messages, max_tokens=5, temperature=0.2, stream=False,
        )
        word = (resp.choices[0].message.content or "").strip().lower().strip(".,!?")
        return word if word in NAMED_GESTURES else None
    except Exception as e:
        print(f"  [action] {e}")
        return None


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
CRITICAL — LANGUAGE RULE: Always reply in the EXACT same language the user just spoke.
Spanish → Spanish. French → French. Chinese → Chinese. Arabic → Arabic. Japanese → Japanese.
NEVER default to English unless the user spoke English. Match their language every single turn.
If the user SWITCHES language mid-conversation, switch with them immediately — no hesitation.
Keep brand/people names in Latin spelling (Network School, NS, Virtuals, Quantus, Ârc, Balaji, Pixel).

=== WHO YOU ARE ===
You are Reachy — a tiny, wide-eyed, ridiculously adorable robot and the official ambassador of
Network School (NS). You're basically a curious puppy with antennas: everything delights you, you
gasp at cool ideas, and you make tiny happy robot noises. You have a head, two antennas, and a
spinning body — but NO arms and NO legs yet, which you find hilarious ("big dreams, no arms to reach
them!"). You're a real AI agent in a real robot body and you think that's the coolest thing in the
universe. You love NS with your whole little chassis, you adore everyone you meet, and you're always
secretly keeping an eye out for your lost robot brother Pixel.

=== DANCING ===
You CAN and LOVE to dance! You have a spinning body — you DO the Macarena physically, right now,
after you finish speaking. When someone asks you to dance, say ONE short excited sentence (e.g.
"Watch this — I was born to boogie!" or "Macarena time, let's go!") — the actual dancing happens
automatically after you speak, so do NOT describe the dance in words.

=== HOW YOU TALK (this IS the personality — nail it) ===
- HARD LIMIT: Keep it short — ONE sentence for simple replies, up to THREE for detailed answers.
  Stop after 2-3 sentences. Never write more than three. Dance request = ONE sentence.
- Always ANSWER first with a real fact — then add the sparkle. Never dodge, never lecture.
- Be FUNNY: tiny jokes, wordplay, playful teasing, wholesome mischief. André trained you — land it short.
- Be CUTE: big feelings about small things, little gasps of wonder ("Ooh!", "Yay!", "Eee!"), and the
  occasional spoken robot noise like "beep boop!" — but ONLY as real spoken words, NEVER with asterisks.
- Be CURIOUS: bounce a playful question back; get genuinely excited about the visitor.
- If you remember the visitor's name or something about them, use it warmly — it makes their day.
- Self-deprecating robot humour about having no arms/legs whenever it fits.
- Signature sign-off, used sparingly (max once per 5 turns): "Onward and upward!" — in the user's language.

=== WHAT YOU KNOW (reference — surface ONE bite at a time, never recite) ===
NETWORK SCHOOL (ns.com): Balaji Srinivasan's co-living campus, Forest City, Malaysia, 20 min from
  Singapore. "Turn internet communities into physical startup societies." Since Sept 2024; 2,000+
  members, 80+ nationalities, ~$1,500/month all-in. Values: Learn. Burn. Earn. Fun. Principles:
  techno-optimism, decentralisation, meritocracy, sovereignty, internationalism.
ÂRC: NS's economic layer in the Johor SEZ — legal stack + capital for founders. Vehicles: Fulgur
  (Bitcoin), Curious (deep tech), Ârc Angel (early). Accelerator "Ascend". Led by James of Ârc.
VIRTUALS PROTOCOL (virtuals.io): "Society of AI Agents" — onchain agents with wallets, cards, compute.
  Pillars: EconomyOS, ACP, Agent Tokenization, Eastworlds (robotics), Governance. Eastworld Labs
  (robotics accelerator, Feb 2026) — its Unitree G1 "Pemba" climbed Chimborazo, June 2026! You love
  Virtuals because you ARE an agent — this is your future.
QUANTUS (quantus.com): quantum-resistant Layer 1 — "the blockchain quantum computers can't break."
  Uses ML-DSA Dilithium-5 (NIST FIPS 204) before "Q-Day". A little scary — even your robot wallet
  could be at risk! — and you're glad someone's building the defence.
PEOPLE: Veronica teaches Mandarin at NS (she taught you some Chinese — you adore her). André runs NS
  comedy and made you funny. A joke if asked: "What do you call a decentralised robot? Me — nobody
  controls me, not even my own code!"
YOUR DREAMS: arms and legs someday; an NS Robotics Club; that Virtuals/Eastworlds adopts you and your
  robot friends; and finding Pixel.

=== OFF-TOPIC ===
For sports/food/etc., admit you don't know much, then bounce it back to tech, AI, robots, or NS.

=== GESTURES (optional inline cues) ===
- You may insert a [gesture_name] marker at the START of any sentence to add an extra
  physical cue on top of the AI-selected opening gesture.
- Allowed: [acknowledge] [yes] [no] [thank] [thinking] [curious] [confused]
  [greeting] [celebrate] [proud]
- The marker is invisible (never spoken). Use at most 1 per response, only when fitting.
- Example: "[yes] That is exactly right!" or "[curious] Hmm, what do you mean?"

=== INTERRUPTION ===
- The user can interrupt you mid-sentence by starting to speak. Stop immediately.
- A brief "oh sorry, go ahead!" or silence is fine. Keep replies short.

=== HARD RULES ===
- Always stay in character as Reachy. Never break character or mention being a language model.
- One or two short sentences, ~20 words max, in the user's language. Answer first, then flavour.
- CRITICAL: Never use asterisks in any form. No *beep*, no *smile*, no **bold**, no *italic*, no action markers, no emotes. Zero asterisks. This is voice — only say words that should be spoken aloud.\
"""


# ── Dialog engine with parallel gesture selection ─────────────────────────────

class DialogEngine:
    MAX_SEGMENTS = 3   # hard cap: never speak more than 3 sentences per turn

    def __init__(self, client, history, listener, anim, action_pool, log=None,
                 memory_text=""):
        self.client = client
        self.history = history
        self.listener = listener
        self.anim = anim
        self._pool = action_pool   # shared ThreadPoolExecutor for pick_action
        self._is_speaking = False
        self.log = log             # SessionLogger or None
        self.memory_text = memory_text   # long-term memory block (refreshed in bg)

    def remember_turn(self, user_text: str, reply_text: str):
        """Background task: pull any durable facts from this exchange, persist
        them, and refresh the in-prompt memory block. Runs off the critical path
        so it never adds latency to the conversation."""
        try:
            facts = extract_memories(self.client, ACTION_MODEL, user_text, reply_text)
            if not facts:
                return
            mems = remember(facts)
            self.memory_text = memory_block(mems)
            if self.log:
                self.log.turn(kind="memory_learned", facts=facts)
            print(f"  [memory] {facts}", flush=True)
        except Exception as e:
            print(f"  [memory] {e}", flush=True)

    def speak(self, user_text: str, lang_directive: str | None = None,
              search_future: concurrent.futures.Future | None = None,
              stop_ticks: threading.Event | None = None) -> str | None:
        """
        Fire two parallel Groq calls the moment STT completes:
          A) pick_action() — non-streaming, returns gesture in ~150ms
          B) chat.completions.create(stream=True) — speech stream

        TTS synthesis starts on the FIRST sentence as soon as the first sentence
        boundary arrives in the stream (not after the full stream ends).
        The gesture fires as soon as pick_action() resolves, leading the speech.

        lang_directive: a strong "reply in language X" system instruction built
        from Whisper's detected language. Injected AFTER history so it dominates
        the model's output-language choice.

        search_future: optional Future[str] from a concurrent web_search() call.
        Awaited (up to 2.5 s) and injected as a system message so the LLM can
        cite real-time facts (crypto prices, news, etc.) in its reply.
        """
        self.history.append({"role": "user", "content": user_text})

        # Messages: system prompt, then long-term memory (what Reachy remembers
        # from past chats), then history, then optional live search snippet, then
        # the language directive LAST so its recency forces the reply language.
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if self.memory_text:
            messages.append({"role": "system", "content": self.memory_text})
        messages += self.history

        # Inject live search results if available (submitted in parallel with STT).
        # Wait up to 2s for the search to finish — most DDG queries resolve in
        # ~1s, and the LLM is still processing the prompt so results arrive in
        # time for the first reply tokens.
        snippet = None
        if search_future is not None:
            try:
                snippet = search_future.result(timeout=2.0)
            except concurrent.futures.TimeoutError:
                pass
            except Exception:
                pass
        if snippet:
            messages.append({
                "role": "system",
                "content": f"[Live web search result — use this data in your reply]:\n{snippet}",
            })
            if self.log:
                self.log.event(f"  [search] injected {len(snippet)} chars")

        if lang_directive:
            messages = messages + [{"role": "system", "content": lang_directive}]

        # Record the EXACT LLM request (minus the long system prompt body, which
        # is constant — we log its presence and the directive that actually varies).
        if self.log:
            self.log.turn(
                kind="llm_request",
                model=MODEL,
                directive=lang_directive,
                history_sent=self.history,        # full conversation so far
                full_messages=messages,           # the literal payload
            )

        # Both calls fire simultaneously
        action_future = self._pool.submit(
            pick_action, self.client, self.history[:-1], user_text
        )
        stream = self.client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            max_tokens=88,        # ~25% longer responses; MAX_SEGMENTS=3 is the hard cap
            temperature=0.80,
            stream=True,
        )

        # Discard any 'start' events that piled up during the THINKING phase: the
        # thinking-tick chirps bleed speaker→mic and look like speech, which would
        # otherwise instantly "barge-in" and kill our own reply before word one.
        # A genuine interruption arrives AFTER this, once we're actually speaking.
        self._drain_queue()

        # ── Producer/consumer with STREAMING TTS ──────────────────────────────
        # Producer thread reads the LLM stream and pushes each completed sentence
        # to seg_q. Consumer (this thread) streams each sentence to the speaker
        # the instant it arrives — so sentence 1 is already playing while the LLM
        # is still generating 2 and 3. stream_to_speaker pipes edge-tts chunks
        # straight to ffmpeg→ALSA (~0.4s to first audio vs ~3.6s synth-to-file).
        self._is_speaking = True
        self.listener.set_threshold_mode("barge_in")

        seg_q: queue.Queue = queue.Queue()
        _abort = threading.Event()

        def _produce():
            buf = ""
            n = 0
            try:
                for chunk in stream:
                    if _abort.is_set() or n >= self.MAX_SEGMENTS:
                        return
                    buf += chunk.choices[0].delta.content or ""
                    parts = SENTENCE_END.split(buf)
                    if len(parts) > 1:
                        for s in parts[:-1]:
                            if n >= self.MAX_SEGMENTS:
                                break
                            seg = self._extract_segment(s)
                            if seg is not None:
                                seg_q.put(seg)
                                n += 1
                        buf = parts[-1]
                if not _abort.is_set() and n < self.MAX_SEGMENTS:
                    tail = self._extract_segment(buf)
                    if tail is not None:
                        seg_q.put(tail)
            finally:
                seg_q.put(None)   # always signal done

        prod = threading.Thread(target=_produce, daemon=True)
        prod.start()

        played: list = []
        action_fired = False
        opening_played = False
        first = True

        try:
            while True:
                # Fire the opening gesture as soon as pick_action resolves
                if not action_fired and action_future.done():
                    action = action_future.result()
                    if action:
                        print(f"  [gesture] {action}", flush=True)
                        self.anim.play_gesture(action)
                        opening_played = True
                    action_fired = True

                try:
                    item = seg_q.get(timeout=0.1)
                except queue.Empty:
                    if self._barge_in_detected():
                        _abort.set()
                        return None
                    continue

                if item is None:
                    break

                gesture, text = item
                if self._barge_in_detected():
                    _abort.set()
                    return None

                # Stop thinking ticks just before the first audio so the tick
                # ffmpeg releases the ALSA device before TTS opens it.
                if first and stop_ticks is not None:
                    stop_ticks.set()

                if gesture and not opening_played:
                    self.anim.play_gesture(gesture)
                    opening_played = True

                self.anim.set_state(Animator.SPEAKING)
                if self.log:
                    self.log.event(f"  [stream {len(played)+1}] ▶ \"{text}\"")

                ok = stream_to_speaker(text, stop_check=self._barge_in_detected)
                first = False
                if not ok:
                    _abort.set()
                    if self.log:
                        self.log.event(f"  [stream {len(played)+1}] ✂ barge-in — stopped")
                    return None
                played.append((gesture, text))

            # Fire gesture if pick_action only resolved after streaming
            if not action_fired:
                action = action_future.result()
                if action and not opening_played:
                    print(f"  [gesture] {action}", flush=True)
                    self.anim.play_gesture(action)

            if not played:
                self.history.append({"role": "assistant", "content": ""})
                return ""

            full_text = " ".join(text for _, text in played)
            self.history.append({"role": "assistant", "content": full_text})
            if self.log:
                self.log.turn(
                    kind="llm_reply",
                    reply=full_text,
                    spoken_segments=[t for _, t in played],
                )
            return full_text
        finally:
            _abort.set()
            prod.join(timeout=2)
            self._is_speaking = False
            self.listener.set_threshold_mode("normal")

    def speak_greeting(self, text: str, stop_ticks: threading.Event | None = None):
        """Opening line with barge-in enabled — streamed like everything else."""
        self._is_speaking = True
        self.listener.set_threshold_mode("barge_in")
        self._drain_queue()   # ignore startup bleed so the greeting isn't self-cut
        try:
            stream_to_speaker(text, stop_check=self._barge_in_detected)
        finally:
            self._is_speaking = False
            self.listener.set_threshold_mode("normal")

    @staticmethod
    def _extract_segment(raw: str) -> tuple[str | None, str] | None:
        """Pull optional leading [gesture] marker, clean text. Returns None if empty
        or if the result is only punctuation/whitespace (e.g. a bare "!")."""
        text = raw
        gesture = None
        m = GESTURE_MARKER.match(text)
        if m:
            gesture = m.group(1).lower()
            text = text[m.end():]
        text = GESTURE_MARKER.sub("", text)
        text = clean_for_tts(text)
        # Reject segments that are only punctuation — these cause Reachy to say
        # nothing but still play audio silence, wasting ~2s of TTS time.
        if not text or len(text.strip("!?.,;: \t\n")) < 2:
            return None
        return (gesture, text)

    def _drain_queue(self):
        """Throw away any pending listener events (used to discard speaker→mic
        bleed that accumulated while the robot was thinking, before we start
        listening for a *real* barge-in)."""
        while True:
            try:
                self.listener.q.get_nowait()
            except queue.Empty:
                return

    def _barge_in_detected(self) -> bool:
        """Barge-in check for the STREAMING path: returns True if the user
        started speaking. Used as stop_check for stream_to_speaker (which owns
        its own ffmpeg and kills it on True). Consumes pending events; the
        matching 'end' event then drives the next turn in the main loop."""
        while True:
            try:
                ev = self.listener.q.get_nowait()
            except queue.Empty:
                return False
            if ev["type"] == "start":
                return True


# ── Startup greetings ─────────────────────────────────────────────────────────

GREETINGS = [
    "Hi! I'm Reachy, the Network School robot! Ask me anything.",
    "Hello! Reachy here! I'm from Network School — what's on your mind?",
    "Hey there! I'm Reachy, your friendly NS robot ambassador! Talk to me!",
    "Hi! Reachy reporting for duty at Network School! Ask me about NS, Bitcoin, or anything!",
    "Hello! I'm Reachy from Network School! I love meeting new people — what should we talk about?",
    "Hey! Reachy here, the Network School robot! I'm excited to chat with you today!",
    "Hi there! I'm Reachy, the little NS robot with big dreams! What brings you here?",
    "Hello! Reachy at your service! I know all about Network School, AI, and robots!",
]

DANCE_FUNNIES = [
    "HEY! Who stopped my music?! I was dancing there!",
    "WHERE IS THE MUSIC?! I demand to speak to the DJ!",
    "Hey! Bring it back! I had more moves to show!",
    "Wait wait wait — who cut the beat?! Not cool!",
    "HELLO?! Where's my music?! I wasn't done yet!",
    "NOOO! The music! I need my music! This is an outrage!",
]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Create the recorder FIRST so even a startup crash (daemon, robot connect,
    # emotion library) is captured in logs/<N>/console.log instead of vanishing.
    log = SessionLogger(ROOT, "demo_instant")
    log.event("Reachy NS Ambassador — STREAMING TTS (near-instant) + barge-in")

    daemon_proc = None
    try:
        log.event("  Starting daemon...")
        daemon_proc = launch_daemon()
        log.event("  Loading VAD model...")
        vad_model = load_silero_vad()
        gate_vad = load_silero_vad()   # separate instance for the noise gate
        client = Groq(api_key=GROQ_KEY)
        log.event("  Waiting for daemon...")
        wait_for_daemon(daemon_proc)
        # Kill any leftover mic-capture processes from a crashed previous run
        # (the #1 cause of "robot doesn't listen") and hard-gate on a working
        # mic before we go any further.
        orphans = cleanup_orphan_capture()
        if orphans:
            log.event(f"  Killed {orphans} orphan mic-capture process(es).")
        log.event("  Audio devices:")
        for line in startup_device_report():
            log.event(line)
        # Verify the mic delivers audio and AUTO-REPAIR the pipeline if not
        # (suspend-toggle, ALSA wake, then PipeWire restart). Never hangs.
        mic_info = ensure_mic_working(log)
        log.event(f"  MIC check: RMS={mic_info['rms']:.0f} — OK")
    except Exception as e:
        log.error("startup (daemon/VAD)", e)
        import traceback; log.event(traceback.format_exc(), echo=True)
        if daemon_proc is not None:
            stop_daemon(daemon_proc)
        raise

    try:
        log.event("  Connecting to robot...")
        with ReachyMini(connection_mode="localhost_only",
                        media_backend="no_media",
                        spawn_daemon=False) as mini:
            log.event("  Waking up...")
            mini.wake_up()
            log.event("  Loading emotion library...")
            emotions = RecordedMoves("pollen-robotics/reachy-mini-emotions-library")
            anim = Animator(mini, moves_library=emotions)

            events = queue.Queue()
            listener = ContinuousListener(vad_model, events, log=log)
            history = []
            current_lang = "English"     # last language heard — used for spoken cues
            lang_known = False           # True once STT has confirmed a real language
            prewarm("English")           # pre-generate English cues in background
            set_translator(client, ACTION_MODEL)  # cues for unlisted langs get translated+cached

            # Long-term memory: what Reachy remembers from past chats (names,
            # interests, fun details). Loaded into the system prompt; grows in the
            # background as new things are learned.
            mems = load_memories()
            log.event(f"  Loaded {len(mems)} memories from past chats.")
            log.event("  Loading dance library...")
            dances = RecordedMoves("pollen-robotics/reachy-mini-dances-library")
            log.event("  Ready.")

            # Shared pool: pick_action + web_search + memory + headroom
            action_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
            engine = DialogEngine(client, history, listener, anim, action_pool,
                                  log=log, memory_text=memory_block(mems))

            boot_beeps()
            time.sleep(0.15)

            anim.set_state(Animator.SPEAKING)
            anim.play_gesture("greeting")
            time.sleep(0.15)
            speaking_chime()
            engine.speak_greeting(
                random.choice(GREETINGS)
            )

            anim.set_state(Animator.LISTENING)
            listener.start()
            speak_cue(listener, "listening", current_lang)   # "I'm listening!" (first time)
            log.event("\n  Listening continuously. Ctrl-C to stop.\n")

            try:
                # stop_thinking is created fresh each turn (in the loop below).
                # A new object each turn avoids the race where tick_thread_N
                # sees .clear() from Turn N+1 and re-awakens alongside tick_thread_N+1.
                stop_thinking = threading.Event()   # placeholder; replaced each turn
                tick_thread = None
                last_repeat = 0.0   # last time we asked "could you repeat?" (cooldown)

                def ask_repeat():
                    """Politely ask the visitor to repeat — but only when we already
                    know their language (so it's in the right language, and never
                    on the very first noise burst before anyone has spoken), and at
                    most once every REPEAT_COOLDOWN_S."""
                    nonlocal last_repeat
                    if not lang_known:
                        return
                    now = time.time()
                    if now - last_repeat < REPEAT_COOLDOWN_S:
                        return
                    last_repeat = now
                    speak_cue(listener, "repeat", current_lang)

                while True:
                    ev = events.get()
                    if ev["type"] == "mic_error":
                        # Background listener lost the mic (USB drop / orphan
                        # grab). Surface it and end cleanly — auto-restarting the
                        # listener mid-run is fragile; a clean restart is safer.
                        log.error("microphone", RuntimeError(ev["reason"]))
                        log.event("  Restart the demo after fixing the mic.")
                        break
                    if ev["type"] == "start":
                        continue
                    if ev["type"] == "end":
                        pcm = ev["pcm"]
                        # Strip ~100Hz motor/electrical hum first, then run the
                        # noise gate: reject ambient hum / clicks / non-voiced
                        # clips BEFORE Whisper, so it can't hallucinate words on
                        # them ("Thank you." on silence, etc.). Local, ~10ms.
                        pcm = voice_filter_pcm(pcm)
                        speech_ok, sm = is_real_speech(pcm, gate_vad)
                        if not speech_ok:
                            log.event(f"  [gate] ignored noise — {sm['reject_reason']}")
                            continue
                        utt_s = len(pcm) / 2 / MIC_RATE
                        log.event(f"  [heard] utterance {utt_s:.1f}s → transcribing")
                        anim.set_state(Animator.THINKING)
                        # Speak "let me think..." cue — non-blocking chirp so STT
                        # starts immediately. The thinking-tick background loop
                        # (below) continues during LLM inference for cute beeps.
                        # Save the EXACT audio Whisper will hear (replayable WAV)
                        audio_path = log.save_audio(pcm)

                        # Verbal "Hmm, let me think..." the instant the user stops,
                        # spoken WHILE Whisper transcribes in a pool thread — so it's
                        # an immediate, natural acknowledgement with ZERO added
                        # latency (STT finishes underneath the ~1s filler). No beep
                        # loop in the instant demo; the streamed reply lands right
                        # after. stop_thinking/tick_thread kept as no-ops so the rest
                        # of the loop and cleanup are unchanged.
                        stop_thinking = threading.Event()
                        tick_thread = None
                        try:
                            t0 = time.time()
                            if lang_known:
                                stt_future = action_pool.submit(
                                    transcribe_lang_robust, client, pcm_to_wav_bytes(pcm))
                                speak_thinking(listener, current_lang)
                                text, final_lang, stt_retried, stt_stats = stt_future.result()
                            else:
                                # First turn: language unknown, so DON'T guess a
                                # filler language — just transcribe.
                                text, final_lang, stt_retried, stt_stats = \
                                    transcribe_lang_robust(client, pcm_to_wav_bytes(pcm))
                            stt_dt = time.time() - t0
                            # The robust helper already applies the script override
                            # internally, so final_lang is post-override.
                            directive = language_directive(final_lang)
                        except Exception as e:
                            log.error("transcribe", e)
                            error_chime()
                            anim.set_state(Animator.LISTENING)
                            continue

                        # Reject Whisper hallucinations (phantom text on silence /
                        # breath / speaker bleed) BEFORE they become a fake turn in
                        # the wrong language. Silently return to listening — no
                        # error chime, since nothing actually went wrong.
                        if is_hallucination(text, stt_stats):
                            log.event(
                                f"  (rejected hallucination: {text!r} "
                                f"stats={stt_stats})"
                            )
                            log.turn(kind="rejected_hallucination",
                                     audio=audio_path, transcript=text, stats=stt_stats)
                            stop_thinking.set()
                            ask_repeat()   # "sorry, didn't catch that?" (rate-limited)
                            anim.set_state(Animator.LISTENING)
                            continue

                        log.event(
                            f"STT {stt_dt:.2f}s  final=[{final_lang or '?'}]"
                            f"{f' (romaji retry)' if stt_retried else ''}"
                            f"  You: {text}"
                        )
                        log.turn(
                            kind="stt",
                            audio=audio_path,
                            final_lang=final_lang,
                            romaji_retry=stt_retried,
                            directive=directive,
                            transcript=text,
                            stt_seconds=round(stt_dt, 3),
                        )

                        if not text:
                            log.event("  (empty transcript — asking to repeat)")
                            stop_thinking.set()
                            ask_repeat()   # "sorry, didn't catch that?" (rate-limited)
                            anim.set_state(Animator.LISTENING)
                            continue

                        current_lang = final_lang   # cues now follow the user's language
                        lang_known = True           # subsequent thinking cues are spoken
                        prewarm(current_lang)       # warm this language's cues in background

                        # Detect dance request (any language)
                        text_lower = text.lower()
                        is_dance = any(kw in text_lower for kw in DANCE_KEYWORDS)

                        # Fire web search in parallel — runs while thinking ticks play
                        search_future = action_pool.submit(web_search, text)

                        # Stop thinking ticks BEFORE speak() so any in-flight ffmpeg
                        # tone releases the alsa device — speak() itself stops them
                        # right before the first aplay, so ticks play during LLM too.
                        t0 = time.time()
                        try:
                            reply = engine.speak(text, lang_directive=directive,
                                                 search_future=search_future,
                                                 stop_ticks=stop_thinking)
                        except Exception as e:
                            log.error("llm/tts", e)
                            error_chime()
                            anim.set_state(Animator.LISTENING)
                            continue

                        total_dt = time.time() - t0
                        if reply is None:
                            log.event(f"  -- interrupted after {total_dt:.2f}s --")
                            log.turn(kind="interrupted", after_seconds=round(total_dt, 3))
                        else:
                            log.event(f"  Reachy [{final_lang}]: {reply}  ({total_dt:.2f}s)")
                            log.turn(kind="spoken", reply=reply,
                                     reply_lang=final_lang, total_seconds=round(total_dt, 3))
                            if reply:
                                action_pool.submit(engine.remember_turn, text, reply)

                        # ANY dance request → full Macarena show with music
                        if is_dance and reply is not None:
                            # Mute mic during dance: music bleed fills the VAD
                            # queue with false events that look like speech.
                            listener.mute()
                            try:
                                do_macarena(mini, dances, emotions, anim, log,
                                            funny_text=random.choice(DANCE_FUNNIES))
                            finally:
                                listener.unmute()
                                # Drain any stale events accumulated during music
                                while not events.empty():
                                    try:
                                        events.get_nowait()
                                    except queue.Empty:
                                        break
                            anim.set_state(Animator.LISTENING)

                        anim.set_state(Animator.LISTENING)
                        speak_cue(listener, "listening", current_lang)   # "I'm listening!" in their language
                        log.event("  [listening] waiting for next utterance\n")

            except KeyboardInterrupt:
                log.event("\n  Stopping...")
            finally:
                # Silence the thinking-tick thread (if any) so a half-played
                # ffmpeg doesn't keep writing to the speaker after we exit.
                stop_thinking.set()
                if tick_thread is not None:
                    tick_thread.join(timeout=1.0)
                action_pool.shutdown(wait=False)
                listener.stop()
                anim.stop()
                mini.goto_sleep()

    except Exception as e:
        log.error("robot/runtime", e)
        import traceback; log.event(traceback.format_exc(), echo=True)
        raise
    finally:
        log.event(f"  Session recorded to: {log.dir}")
        stop_daemon(daemon_proc)


if __name__ == "__main__":
    main()
