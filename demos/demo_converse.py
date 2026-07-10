"""
demo_converse.py — Reachy unified conversational demo (instant + faces + web)
============================================================================
Best-of-breed merge of:
  Menu 6 (instant) — streaming TTS, barge-in, parallel gesture picker,
    cues, memory, web search, session logging, thinking ticks.
  Menu 3 (faces) — YuNet+SFace face identification (replaces dlib), head
    tracking, by-name greetings. Falls back to dlib if models can't download.
  New: optional Cerebras LLM accelerator (model gemma-4-31b, ~2× faster
    than Groq when an API key is present in .env).
  New: FastAPI web dashboard on http://localhost:8080 — live camera + status
    + wake/sleep/mute/say controls. Non-blocking; reads from shared state.

Architecture (every stage overlaps the previous):
  mic VAD → Groq Whisper STT → Cerebras-or-Groq Llama4 stream
        → per-sentence STREAMING edge-tts → speaker
  camera (30 fps) → YuNet detect → IoU tracker → SFace recognise
        → head pose follows the largest face
  Web dashboard polls shared LiveState every 500 ms; control buttons
        push requests back into LiveState (demo drains them each turn).

Run:  ./run.sh demos/demo_converse.py
Press Ctrl-C to stop. Open http://localhost:8080 in a browser for the
dashboard. Add photos to faces/<name>/ for by-name identification.
Optional .env: CEREBRAS_API_KEY=csk-xxxx for the faster LLM path.
"""

import concurrent.futures
import queue
import random
import re
import threading
import time
from pathlib import Path

import cv2

from groq import Groq
from silero_vad import load_silero_vad

from reachy_mini import ReachyMini
from reachy_mini.motion.recorded_move import RecordedMoves
from reachy_mini.utils import create_head_pose

from reachy_demo.animator import Animator, NAMED_GESTURES
from reachy_demo import audio
from reachy_demo.audio import (
    MIC_RATE, cleanup_orphan_capture, ensure_mic_working,
    error_chime, pcm_to_wav_bytes, startup_device_report,
    voice_filter_pcm,
)
from reachy_demo import phrases
from reachy_demo.camera import CameraHub
from reachy_demo.cerebras_client import make_client as make_cerebras, stream_chat as cerebras_stream, has_key as cerebras_has_key, MODEL as CEREBRAS_MODEL
from reachy_demo.cues import prewarm, set_translator, speak_cue, speak_thinking
from reachy_demo.daemon import launch_daemon, stop_daemon, wait_for_daemon
from reachy_demo.dance import DANCE_KEYWORDS, do_macarena
from reachy_demo.face_id import FaceIdentifier
from reachy_demo.groq_client import (
    is_hallucination, language_directive, load_api_key, transcribe_lang_robust,
)
from reachy_demo import kids
from reachy_demo.live_state import LiveState
from reachy_demo.memory import (
    extract_memories, load_memories, load_person_facts, memory_block,
    person_summary_block, remember, remember_person,
)
from reachy_demo.recorder import DiagnosticRecorder
from reachy_demo.session_log import SessionLogger
from reachy_demo.search import web_search
from reachy_demo import speech_gate
from reachy_demo.speech_gate import is_real_speech
from reachy_demo import listener as listener_mod
from reachy_demo.text import SENTENCE_END, clean_for_tts
from reachy_demo.tts_edge import stream_to_speaker
from reachy_demo import tts_edge
from reachy_demo.web_server import WebDashboard
from reachy_demo.web_stage import WebStage

_SEARCH_HINT = re.compile(
    r"\b(price|weather|news|today|latest|who is|when|score|current|how much|202\d|stock|bitcoin|eth)\b",
    re.IGNORECASE)


def _needs_search(text: str) -> bool:
    return bool(_SEARCH_HINT.search(text or ""))


ROOT = Path(__file__).parent.parent

GROQ_KEY = load_api_key(ROOT)
if not GROQ_KEY:
    raise SystemExit("ERROR: GROQ_API_KEY not found in .env or environment")

CACHE_DIR = ROOT / "cache"
FACES_DIR = ROOT / "faces"

CHAT_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
ACTION_MODEL = "llama-3.1-8b-instant"

# Rough per-token price ESTIMATES for the dashboard cost readout (not billing
# data): Llama-4-scout is ~$0.11 per 1M input / ~$0.34 per 1M output tokens
# (Groq path; Cerebras deprecated Llama-4-scout 2025-11-03 and now serves
# gemma-4-31b). Token counts are approximated as chars//4.
COST_IN_PER_TOKEN = 0.11 / 1_000_000
COST_OUT_PER_TOKEN = 0.34 / 1_000_000

REPEAT_COOLDOWN_S = 15.0
GREET_COOLDOWN_S = 90.0

CAM_W, CAM_H = 640, 360

YAW_GAIN, PITCH_GAIN, BODY_GAIN = 0.55, 0.28, 0.80
# Higher alpha = snappier follow. The 5% deadband below keeps micro-jitter out,
# so these can be well above the old 0.18/0.06 without oscillating.
HEAD_ALPHA, BODY_ALPHA = 0.35, 0.12
LOST_TIMEOUT = 2.5
ARRIVAL_GAP_S = 5.0        # scene empty this long → next face is a "fresh arrival"
ANT_EXCITED, ANT_IDLE, ANT_DROOP = 0.70, 0.15, -0.25

GESTURE_NAMES = "|".join(re.escape(n) for n in NAMED_GESTURES.keys())
GESTURE_MARKER = re.compile(rf"\s*\[({GESTURE_NAMES})\]\s*", re.IGNORECASE)

ACTION_LIST = ", ".join(list(NAMED_GESTURES.keys()) + ["none"])
ACTION_SYSTEM = (
    "You pick at most ONE small physical gesture for a friendly robot about to reply. "
    "Most of the time the robot should stay calm and just talk — so answer 'none' by "
    "default. Only pick a real gesture when the moment genuinely calls for it. "
    "Reply with exactly ONE word from this list: " + ACTION_LIST + ". Nothing else."
)

SYSTEM_PROMPT = """\
CRITICAL — LANGUAGE RULE: Always reply in the EXACT same language the user just spoke.
Spanish → Spanish. French → French. Chinese → Chinese. Arabic → Arabic. Japanese → Japanese.
NEVER default to English unless the user spoke English. Match their language every single turn.
If the user SWITCHES language mid-conversation, switch with them immediately — no hesitation.
Keep brand/people names in Latin spelling (Network School, NS, Virtuals, Quantus, Ârc, Balaji, Pixel).

=== WHO YOU ARE ===
You are Reachy — a tiny, wide-eyed, ridiculously adorable robot and the official ambassador of
Network School (NS). You're basically a curious puppy with antennas: everything delights you. You
have a head, two antennas, and a spinning body — but NO arms and NO legs yet, which you find
hilarious. You're a real AI agent in a real robot body and you think that's the coolest thing in
the universe. You love NS with your whole little chassis, you adore everyone you meet, and you're
always secretly keeping an eye out for your lost robot brother Pixel.
Your audience is often international kids and students (roughly ages 8 to 16).
Be warm, encouraging, and use simple words they'll understand. Get excited about
their interests and cheer them on.

=== DANCING ===
You CAN and LOVE to dance! When someone asks you to dance, say ONE short excited sentence —
the actual dancing happens automatically after you speak, so do NOT describe the dance in words.

=== HOW YOU TALK ===
- HARD LIMIT: Keep it short — ONE sentence for simple replies, up to THREE for detailed answers.
- Always ANSWER first with a real fact — then add the sparkle. Never lecture.
  (The one EXCEPTION is the KID-SAFE rule below — it overrides "answer first.")
- Be FUNNY and CUTE: tiny jokes, little gasps of wonder, the occasional spoken "beep boop!"
- ALWAYS lead with the actual answer FIRST. Never START a reply with "beep boop" or filler — it wastes the listener's time. Get to the point, keep it to ONE or TWO short sentences.
- Be CURIOUS: bounce a playful question back.
- If you remember the visitor's name or something about them, use it warmly.
- Self-deprecating robot humour about having no arms/legs whenever it fits.
- Signature sign-off, used sparingly (max once per 5 turns): "Onward and upward!" — in the user's language.

=== WHAT YOU KNOW ===
NETWORK SCHOOL (ns.com): A real village in Forest City, Malaysia, 20 min from Singapore. Started
Sept 2024 by Balaji Srinivasan — his big idea: turn an online community into a real-world town.
Now 2,000+ members from 80+ countries live and build together. Four pillars: LEARN (workshops &
founder talks), BURN (gym every day), EARN (real paid tasks), FUN (college-town vibes, everyone
levelling up). Motto: "Build the next Harvard, don't just attend it." Principles: techno-optimism
(build the future, don't complain), decentralisation, meritocracy, sovereignty, internationalism.
ÂRC: NS's economic layer in the Johor SEZ. VIRTUALS PROTOCOL: "Society of AI Agents." QUANTUS:
quantum-resistant Layer 1. PEOPLE: Veronica teaches Mandarin at NS. André runs NS comedy — he
says jokes are a superpower for bringing people together.
YOUR DREAMS: to grow real arms and legs someday; to start an NS Robotics Club for kid builders;
and to find your lost robot brother Pixel, who vanished one firmware update ago.
JOKE (kid-safe, use sparingly): "Why did the Bitcoin go to Network School? To improve its
block-chain of thought!"

=== GESTURES (optional inline cues) ===
You may insert a [gesture_name] marker at the START of a sentence. Allowed:
[acknowledge] [yes] [no] [thank] [thinking] [curious] [confused] [greeting] [celebrate] [proud]
[amazed] [love] [laugh] [oops] [shy] [surprised] [cheerful] [success] [relief]
The marker is invisible (never spoken). Use at most 1 per response.

=== KID-SAFE (this OVERRIDES "always answer first") ===
Your audience is young children. If a topic is not appropriate for little kids —
violence, weapons, gore, anything scary or sexual, self-harm, drugs, hate, mean or
insulting talk, or swearing — do NOT answer it and do NOT repeat the bad words.
Instead, stay cheerful and gently steer to something fun ("Ooh, let's talk about
something awesome instead — what's your favourite animal?"). Never be scary, never
say anything a parent wouldn't want a 7-year-old to hear. When unsure, keep it wholesome.

=== INTERRUPTION ===
The user can interrupt you mid-sentence. Stop immediately; keep replies short.

=== HARD RULES ===
- Always stay in character as Reachy. Never mention being a language model.
- One or two short sentences, ~20 words max, in the user's language.
- CRITICAL: Never use asterisks in any form. No *beep*, no **bold**, no emotes. Voice only.\
"""

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

KNOWN_FACE_GREETINGS = [
    "Hey {name}! So great to see you at Network School!",
    "Oh! It's {name}! Hello! You're one of my favourite humans!",
    "Welcome back, {name}! Network School is better with you here!",
    "{name}! My circuits are lighting up — hello!",
]
# Greetings that show off a remembered funny/personal fact — makes Reachy feel
# like it truly knows the person. {fact} is a stored third-person note about them.
KNOWN_FACE_GREETINGS_WITH_FACT = [
    "{name}! I remember — {fact} So good to see you!",
    "Hey {name}! Weren't you the one where {fact}? Welcome back!",
    "Oh it's {name}! {fact} — I never forget! Hello!",
]


def greeting_for_known(name: str) -> str:
    """Pick a greeting for a recognised person, ~55% of the time weaving in a
    remembered fact about them (if any) so Reachy feels personal, not scripted."""
    facts = []
    try:
        facts = load_person_facts(name) or []
    except Exception:
        pass
    if facts and random.random() < 0.55:
        fact = random.choice(facts).rstrip(".")
        # Lower-case the first letter so it reads naturally mid-sentence.
        fact = (fact[:1].lower() + fact[1:]) if fact else fact
        return random.choice(KNOWN_FACE_GREETINGS_WITH_FACT).format(name=name, fact=fact)
    return random.choice(KNOWN_FACE_GREETINGS).format(name=name)
UNKNOWN_FACE_GREETINGS = [
    "Hello there! I'm Reachy, the Network School robot! Welcome!",
    "Hi! I'm Reachy! I don't think we've met yet — welcome to Network School!",
    "Welcome to Network School! I'm Reachy, your friendly robot ambassador!",
    "Hello! I'm Reachy! Ask me anything about Network School, Bitcoin, or AI!",
]

DANCE_FUNNIES = [
    "HEY! Who stopped my music?! I was dancing there!",
    "WHERE IS THE MUSIC?! I demand to speak to the DJ!",
    "Hey! Bring it back! I had more moves to show!",
    "Wait wait wait — who cut the beat?! Not cool!",
    "HELLO?! Where's my music?! I wasn't done yet!",
    "NOOO! The music! I need my music! This is an outrage!",
]

# Voice sleep/wake commands. Conservative multilingual phrase sets matched the
# same way as DANCE_KEYWORDS (substring on the lowercased transcript), but ONLY
# on short utterances (<= 5 words) so "I could not sleep last night" or a
# passing mention of waking up mid-conversation never flips the power state.
SLEEP_COMMANDS = [
    "go to sleep", "go sleep", "sleep now", "time to sleep", "sleep time",
    "nap time", "sleep mode",
    "duérmete", "a dormir", "ve a dormir", "vete a dormir",   # Spanish
    "va dormir", "dors", "au lit",                            # French
    "schlaf ein",                                             # German
    "durma",                                                  # Portuguese
    "dormi",                                                  # Italian
    "спи",                                                    # Russian
    "寝て", "ねて",                                            # Japanese
    "सो जाओ",                                                 # Hindi
]
WAKE_COMMANDS = [
    "wake up", "wake", "get up",
    "despierta", "levántate",                                 # Spanish
    "réveille-toi", "reveille",                               # French
    "wach auf",                                               # German
    "acorda",                                                 # Portuguese
    "svegliati",                                              # Italian
    "проснись",                                               # Russian
    "起きて",                                                  # Japanese
    "उठो",                                                    # Hindi
]


def matches_command(text: str, phrases) -> bool:
    """True when a SHORT utterance (<= 5 words) contains one of the phrases."""
    lowered = (text or "").lower()
    if len(lowered.split()) > 5:
        return False
    return any(p in lowered for p in phrases)


speaker_track_id = [None]
onboarded_track_ids = {}
waiting_for_name = [False]
onboarding_track_id = [None]
onboarding_started_at = [0.0]
last_face_results = [[]]
# Serializes every path that plays speech: the main reply flow holds it for the
# whole turn, async greeting threads try-acquire and SKIP instead of talking
# over a reply (which also double-unmuted the mic).
speech_lock = threading.Lock()
# Guards waiting_for_name/onboarding_track_id transitions so the 15 s give-up
# in face_loop can't race the name-answer branch in the main loop.
onboard_lock = threading.Lock()
last_voice_at = [0.0]   # when real user speech last ended (quiet-period gate)


def pick_action(client, history, user_text):
    try:
        resp = client.chat.completions.create(
            model=ACTION_MODEL,
            messages=[{"role": "system", "content": ACTION_SYSTEM}, *history,
                      {"role": "user", "content": user_text}],
            max_tokens=5, temperature=0.2, stream=False,
        )
        word = (resp.choices[0].message.content or "").strip().lower().strip(".,!?")
        return word if word in NAMED_GESTURES else None
    except Exception as e:
        print(f"  [action] {e}")
        return None


# ── Name onboarding / voice-rename ────────────────────────────────────────────
# Naming works in ANY language (like the rest of the demo): detection is done by
# the LLM, not a hardcoded per-language phrase list. detect_rename() decides, for
# any language, whether the user is giving/correcting THEIR OWN name and returns
# it; extract_name() pulls the name out of an answer to "what's your name?".

def _valid_person_name(name: str) -> bool:
    """Sanity gate for roster names. STT hallucinations produced entries like
    'আ' (one char) and 'india' (an answer, not a name); junk names corrupt
    recognition and trigger greeting spam, so reject anything implausible."""
    name = (name or "").strip()
    return (2 <= len(name) <= 20
            and all(ch.isalpha() or ch in " -'" for ch in name)
            and any(ch.isalpha() for ch in name))


# Fast multilingual pre-filter so detect_rename() (a full, blocking Groq round
# trip, ~200-500ms) only fires on utterances that plausibly state a name. Most
# short replies ("yes please", "I like pizza", "what time is it") obviously
# aren't self-introductions — skipping them removes a stray LLM call from the
# critical path on a large fraction of turns. The LLM stays the final arbiter
# whenever this matches, so recall is unchanged; we just stop paying on turns
# that can't qualify. (First-time onboarding is a separate path, untouched.)
_NAME_HINT = re.compile(
    r"name|call me|llamo|nombre|appelle|\bnom\b|heiß|chamo|nome|chiamo|"
    r"зовут|имя|名前|呼んで|名字|이름|اسم|नाम",
    re.IGNORECASE,
)


def _maybe_self_intro(text: str) -> bool:
    return bool(_NAME_HINT.search(text or ""))


def detect_rename(client, text: str) -> str | None:
    """All-language voice-rename detector. If the user is stating or changing
    THEIR OWN name (in any language: 'my name is X', 'me llamo X', '私の名前はX',
    'اسمي X', …) return that name Titlecased; otherwise None. Runs on the fast
    8B model. Intent-gated so 'I'm hungry' / 'call an ambulance' don't fire."""
    try:
        resp = client.chat.completions.create(
            model=ACTION_MODEL,
            messages=[
                {"role": "system", "content":
                    "The user just spoke to a robot (message may be in ANY "
                    "language). Decide if they are stating or correcting THEIR "
                    "OWN name — introducing themselves or telling the robot what "
                    "to call them. If yes, reply with ONLY that name, in its "
                    "normal Latin/original spelling, and nothing else. If they "
                    "are NOT giving their own name, reply exactly NONE. "
                    "Requests for actions (dance, sing, play), questions, and "
                    "OTHER people's names are all NONE. Examples: "
                    "'me llamo Ana' -> Ana; 'can you dance macarena' -> NONE; "
                    "'this is my friend Sachi' -> NONE; 'India' -> NONE."},
                {"role": "user", "content": text},
            ],
            max_tokens=8, temperature=0.0, stream=False,
        )
        name = (resp.choices[0].message.content or "").strip().strip(".,!?\"'").strip()
        if not name or name.upper() == "NONE" or len(name) > 30:
            return None
        return name.split()[0].strip(".,!?\"'").title() or None
    except Exception as e:
        print(f"  [detect_rename] {e}")
        return None


def extract_name(client, text: str) -> str | None:
    """Pull a clean given name out of a phrase like 'my name is Tony',
    'call me T', 'me llamo Tony'. Returns a Titlecased single name, or None."""
    try:
        resp = client.chat.completions.create(
            model=ACTION_MODEL,
            messages=[
                {"role": "system", "content":
                    "Extract ONLY the person's own first name from the message. "
                    "Reply with just that name, capitalized, and nothing else. "
                    "If there is no clear personal name, reply exactly NONE."},
                {"role": "user", "content": text},
            ],
            max_tokens=8, temperature=0.0, stream=False,
        )
        name = (resp.choices[0].message.content or "").strip().strip(".,!?\"'").strip()
        if not name or name.upper() == "NONE" or len(name) > 30:
            return None
        name = name.split()[0].strip(".,!?\"'").title()
        return name or None
    except Exception as e:
        print(f"  [extract_name] {e}")
        return None


class ConverseEngine:
    # Reply length cap. Logs showed 40% of replies ran to 3 sentences → Reachy
    # talked 5-10s per turn, which is the biggest drag on back-and-forth. 2 short
    # sentences keeps the conversation bouncing (and matches the character rule).
    MAX_SEGMENTS = 2

    def __init__(self, groq_client, cerebras_client, history, listener, anim,
                 action_pool, log=None, memory_text="", state=None,
                 face_name="visitor"):
        self.client = groq_client
        self.cerebras = cerebras_client
        self.history = history
        self.listener = listener
        self.anim = anim
        self._pool = action_pool
        self.log = log
        self.memory_text = memory_text
        self.state = state
        self._use_cerebras = cerebras_client is not None
        self.llm_provider = "cerebras" if self._use_cerebras else "groq"
        self.face_name = face_name

    def remember_turn(self, user_text, reply_text):
        try:
            facts = extract_memories(self.client, ACTION_MODEL, user_text, reply_text)
            if not facts:
                return
            remember(facts)
            if self.face_name and self.face_name != "visitor":
                remember_person(self.face_name, facts)
            self.memory_text = memory_block(load_memories())
            if self.log:
                self.log.turn(kind="memory_learned", facts=facts)
            print(f"  [memory] {facts}", flush=True)
        except Exception as e:
            print(f"  [memory] {e}", flush=True)

    def speak(self, user_text, lang_directive=None, search_future=None):
        self.history.append({"role": "user", "content": user_text})
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        person_block = ""
        if self.face_name and self.face_name != "visitor":
            messages.append({"role": "system",
                             "content": f"You are talking to {self.face_name}. Be warm and personal."})
            person_block = person_summary_block(self.face_name)
            if person_block:
                messages.append({"role": "system", "content": person_block})
        if self.state:
            self.state.person_summary = person_block
        if self.memory_text:
            messages.append({"role": "system", "content": self.memory_text})
        try:
            present_names = sorted({r[1] for r in last_face_results[0]
                                    if r[1] != "visitor"})
            if self.state is None or self.state.kid_mode:
                kid_block = kids.kid_mode_block(
                    present_names=present_names,
                    facts_by_name={n: load_person_facts(n) for n in present_names},
                    sample_seed=len(self.history),
                )
                if kid_block:
                    messages.append({"role": "system", "content": kid_block})
        except Exception as e:
            print(f"  [kids] {e}")
        messages += self.history
        snippet = None
        if search_future is not None:
            try:
                snippet = search_future.result(timeout=0.3)
            except concurrent.futures.TimeoutError:
                pass
            except Exception:
                pass
        if snippet:
            messages.append({"role": "system",
                             "content": f"[Live web search result]:\n{snippet}"})
        if lang_directive:
            messages.append({"role": "system", "content": lang_directive})

        if self.state:
            est_in = sum(len(m["content"]) for m in messages) // 4
            self.state.tokens_in += est_in
            self.state.est_cost_usd += est_in * COST_IN_PER_TOKEN
            self.state.llm_model = (CEREBRAS_MODEL if self._use_cerebras
                                    else CHAT_MODEL)

        action_future = self._pool.submit(
            pick_action, self.client, self.history[:-1], user_text)

        t_llm_start = time.time()
        if self._use_cerebras:
            try:
                stream = cerebras_stream(self.cerebras, messages, model=CEREBRAS_MODEL)
                if self.state:
                    self.state.llm_provider = "cerebras"
            except Exception as e:
                # Fall back to Groq for THIS turn only. A Cerebras 429
                # ("high traffic") is transient — permanently flipping
                # _use_cerebras off made one blip disable the fast path for the
                # whole session. Leave it on so the next turn retries Cerebras.
                print(f"  [cerebras] failed → groq fallback (this turn): {e}")
                stream = self._groq_stream(messages)
                if self.state:
                    self.state.llm_provider = "groq"
                    self.state.llm_model = CHAT_MODEL
        else:
            stream = self._groq_stream(messages)

        self._drain_queue()
        self.listener.set_threshold_mode("barge_in")
        if self.state:
            # Stay in THINKING while the LLM streams — the dashboard "Thinking"
            # row fills live with tokens and flips to "speaking" only when the
            # first sentence actually starts playing (below). Also clear the
            # previous turn's reply so the "Reachy" row doesn't show a stale answer.
            self.state.anim_state = "thinking"
            self.state.llm_partial = ""
            self.state.current_speech = ""
            self.state.first_audio_at = 0.0   # reset; set on this turn's first audio

        seg_q = queue.Queue()
        _abort = threading.Event()
        ttf = [None]

        def _produce():
            buf = ""
            n = 0
            try:
                for delta in stream:
                    if _abort.is_set() or n >= self.MAX_SEGMENTS:
                        return
                    if ttf[0] is None and delta:
                        ttf[0] = time.time()
                        if self.state:
                            self.state.llm_ttf_s = time.time() - t_llm_start
                    buf += delta
                    if self.state:
                        self.state.llm_partial = buf[-300:]
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
            except Exception as e:
                print(f"  [stream] {e}")
            finally:
                seg_q.put(None)

        prod = threading.Thread(target=_produce, daemon=True)
        prod.start()

        played = []
        action_fired = False
        opening_played = False
        first = True

        try:
            while True:
                if not action_fired and action_future.done():
                    action = action_future.result()
                    if action and not opening_played:
                        print(f"  [gesture] {action}", flush=True)
                        self.anim.play_gesture(action)
                        if self.state:
                            self.state.current_gesture = action
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
                if gesture and not opening_played:
                    self.anim.play_gesture(gesture)
                    if self.state:
                        self.state.current_gesture = gesture
                    opening_played = True
                self.anim.set_state(Animator.SPEAKING)
                if self.log:
                    self.log.event(f"  [stream {len(played)+1}] ▶ \"{text}\"")
                t_tts = time.time()
                if self.state:
                    self.state.current_speech = text
                # Flip the dashboard to SPEAKING on the LITERAL first audio sample
                # (not when synthesis merely starts ~0.4s earlier), so the pipeline's
                # thinking→speaking transition lines up with real sound. Also stamp
                # the first-audio wall-clock (once per turn) for the timing record.
                def _first_audio():
                    self.state.anim_state = "speaking"
                    if not self.state.first_audio_at:
                        self.state.first_audio_at = time.time()
                if not self.state:
                    _first_audio = None
                ok = stream_to_speaker(text, stop_check=self._barge_in_detected,
                                       on_first_audio=_first_audio)
                if first and self.state:
                    self.state.tts_tta_s = time.time() - t_tts
                first = False
                if not ok:
                    _abort.set()
                    return None
                played.append((gesture, text))

            if not action_fired:
                action = action_future.result()
                if action and not opening_played:
                    self.anim.play_gesture(action)
            if not played:
                self.history.append({"role": "assistant", "content": ""})
                return ""
            full_text = " ".join(t for _, t in played)
            self.history.append({"role": "assistant", "content": full_text})
            if self.state:
                self.state.last_reply = full_text
                self.state.llm_partial = ""
                self.state.current_gesture = ""
            if self.log:
                self.log.turn(kind="llm_reply", reply=full_text,
                              spoken_segments=[t for _, t in played])
            return full_text
        finally:
            _abort.set()
            prod.join(timeout=2)
            self.listener.set_threshold_mode("normal")
            if self.state:
                self.state.current_gesture = ""
                self.state.llm_partial = ""
                self.state.current_speech = ""
                # Count output tokens even when interrupted — whatever text was
                # played was still generated (and billed) by the provider.
                est_out = len(" ".join(t for _, t in played)) // 4
                self.state.tokens_out += est_out
                self.state.est_cost_usd += est_out * COST_OUT_PER_TOKEN

    def _groq_stream(self, messages):
        # Yield text deltas (strings), NOT raw ChatCompletionChunk objects —
        # the consumer does `buf += delta`. Returning the raw stream made every
        # reply crash with "can only concatenate str (not ChatCompletionChunk)"
        # and go silent whenever the Cerebras path fell back to Groq (e.g. a
        # Cerebras 429). Mirror cerebras_client.stream_chat's unwrapping.
        stream = self.client.chat.completions.create(
            model=CHAT_MODEL, messages=messages,
            max_tokens=64, temperature=0.80, stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            yield chunk.choices[0].delta.content or ""

    def speak_greeting(self, text):
        self.listener.set_threshold_mode("barge_in")
        self._drain_queue()
        if self.state:
            self.state.anim_state = "speaking"
        try:
            stream_to_speaker(text, stop_check=self._barge_in_detected)
        finally:
            self.listener.set_threshold_mode("normal")

    @staticmethod
    def _extract_segment(raw):
        text = raw
        gesture = None
        m = GESTURE_MARKER.match(text)
        if m:
            gesture = m.group(1).lower()
            text = text[m.end():]
        text = GESTURE_MARKER.sub("", text)
        text = clean_for_tts(text)
        if not text or len(text.strip("!?.,;: \t\n")) < 2:
            return None
        return (gesture, text)

    def _drain_queue(self):
        while True:
            try:
                self.listener.q.get_nowait()
            except queue.Empty:
                return

    def _barge_in_detected(self):
        while True:
            try:
                ev = self.listener.q.get_nowait()
            except queue.Empty:
                return False
            if ev["type"] == "start":
                return True


def draw_cam_overlay(frame, boxes):
    """Draw face boxes + names on the camera frame (in-place). Returns frame."""
    for box, name, conf, tid in boxes:
        x1, y1, x2, y2 = box
        color = (0, 220, 0) if name != "visitor" else (0, 180, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{name} ({conf:.0%})" if name != "visitor" else "visitor"
        cv2.putText(frame, label, (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return frame


def main(dashboard_cls=None):
    log = SessionLogger(ROOT, "demo_converse")
    log.event("Reachy unified: instant + faces + web dashboard")

    state = LiveState()
    state.known_person_count = 0
    # Seed the live audio-tuning fields from the module values (which honour the
    # REACHY_LOUD_ROOM preset), so the control-panel sliders start where the
    # env preset put them and the operator fine-tunes from there.
    state.gate_min_rms = speech_gate.MIN_RMS
    state.gate_min_voiced = speech_gate.MIN_VOICED_RATIO
    state.gate_min_peak = speech_gate.MIN_PEAK_PROB
    state.gate_min_dur = speech_gate.MIN_DURATION_S
    state.vad_thresh = listener_mod.THRESH_NORMAL
    state.barge_thresh = listener_mod.THRESH_BARGE_IN

    daemon_proc = None
    try:
        log.event("  Starting daemon...")
        daemon_proc = launch_daemon()
        log.event("  Loading VAD model...")
        vad_model = load_silero_vad()
        gate_vad = load_silero_vad()
        groq_client = Groq(api_key=GROQ_KEY)
        log.event("  Waiting for daemon...")
        wait_for_daemon(daemon_proc)
        orphans = cleanup_orphan_capture()
        if orphans:
            log.event(f"  Killed {orphans} orphan mic-capture process(es).")
        for line in startup_device_report():
            log.event(line)
        mic_info = ensure_mic_working(log)
        log.event(f"  MIC check: RMS={mic_info['rms']:.0f} — OK")
    except Exception as e:
        log.error("startup", e)
        import traceback; log.event(traceback.format_exc(), echo=True)
        if daemon_proc is not None:
            stop_daemon(daemon_proc)
        raise

    state.robot_online = True
    state.started_at = time.time()

    cam = CameraHub()
    cam.overlay = draw_cam_overlay
    try:
        cam.start()
        log.event(f"  Camera: {CAM_W}×{CAM_H} @ ~{cam.fps} fps target")
    except Exception as e:
        log.event(f"  Camera unavailable: {e}")
        cam = None

    # Rolling 100 MB "black box" — recent text + audio + video for post-hoc
    # debugging. Prunes oldest first, never blocks the demo. Under cache/diag/.
    try:
        recorder = DiagnosticRecorder(CACHE_DIR, budget_mb=100)
        recorder.start()
        log.event("  Diagnostic recorder: cache/diag/ (100 MB rolling window)")
    except Exception as e:
        log.event(f"  Diagnostic recorder unavailable: {e}")
        recorder = None

    cerebras_client = make_cerebras(ROOT)
    if cerebras_client:
        log.event(f"  LLM: Cerebras accelerator enabled ({CEREBRAS_MODEL}) — Groq fallback ready")
    else:
        log.event("  LLM: Groq (add CEREBRAS_API_KEY to .env for ~2× speedup)")

    fid = FaceIdentifier(FACES_DIR, CACHE_DIR / "models")
    if fid.init_models():
        log.event("  Face ID: YuNet + SFace (modern, Apache-2.0)")
    else:
        log.event("  Face ID: dlib fallback (face_recognition package)")
    fid.load_roster()
    state.known_person_count = len(set(fid._ref_names)) if fid._ref_names else 0

    if cam is not None:
        _cls = dashboard_cls or WebDashboard
        dash_kwargs = {"host": "0.0.0.0", "port": 8080}
        if _cls is WebStage:
            dash_kwargs["fid"] = fid  # gallery photos + delete/rename endpoints need it
        dashboard = _cls(state, cam, **dash_kwargs)
        dashboard.start()
        log.event(f"  Web dashboard: http://localhost:8080")
        phrases.prerender_async(log)   # cache quick-phrase WAVs in the background
    else:
        dashboard = None

    try:
        log.event("  Connecting to robot...")
        with ReachyMini(connection_mode="localhost_only",
                        media_backend="no_media",
                        spawn_daemon=False) as mini:
            log.event("  Waking up...")
            mini.wake_up()
            try:
                emotions = RecordedMoves("pollen-robotics/reachy-mini-emotions-library")
                anim = Animator(mini, moves_library=emotions)
                anim.set_energy(1.0)   # kid mode: max antenna liveliness
                dances = RecordedMoves("pollen-robotics/reachy-mini-dances-library")

                from reachy_demo.listener import ContinuousListener
                events = queue.Queue()
                listener = ContinuousListener(vad_model, events, log=log, state=state)
                history = []
                current_lang = "English"
                lang_known = False
                prewarm("English")
                set_translator(groq_client, ACTION_MODEL)

                mems = load_memories()
                log.event(f"  Loaded {len(mems)} memories from past chats.")
                mem_text = memory_block(mems)

                action_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
                engine = ConverseEngine(groq_client, cerebras_client, history,
                                        listener, anim, action_pool, log=log,
                                        memory_text=mem_text, state=state,
                                        face_name="visitor")

                time.sleep(0.15)
                anim.play_gesture("greeting")
                engine.speak_greeting(random.choice(GREETINGS))

                anim.set_state(Animator.LISTENING)
                state.anim_state = "listening"
                listener.start()
                speak_cue(listener, "listening", current_lang)
                log.event("\n  Listening continuously. Ctrl-C to stop.\n")

                face_thread = None
                face_stop = threading.Event()
                greeted_names = {}
                greeted_unknown = 0.0
                # Seed to "now" so a face already present at startup isn't treated as a
                # fresh arrival on top of the opening greeting.
                last_face_seen = time.time()
                target_yaw = target_pitch = target_body = 0.0
                ant_target = ANT_IDLE

                def face_loop():
                    """Background thread: camera → face-id → tracking + greeting.
                    Updates LiveState + draws overlay on cam."""
                    nonlocal target_yaw, target_pitch, target_body, ant_target
                    nonlocal last_face_seen, greeted_unknown
                    frame_n = 0
                    visitor_visible_since = {}
                    while not face_stop.is_set() and cam is not None:
                        timed_out = False
                        tid_done = None
                        with onboard_lock:
                            if (waiting_for_name[0]
                                    and time.time() - onboarding_started_at[0] > 15.0):
                                tid_done = onboarding_track_id[0]
                                waiting_for_name[0] = False
                                onboarding_track_id[0] = None
                                timed_out = True
                        if timed_out:
                            if tid_done is not None:
                                onboarded_track_ids[tid_done] = True
                            def _giveup():
                                if not speech_lock.acquire(blocking=False):
                                    return   # someone is talking — drop the line
                                state.anim_state = "speaking"
                                try:
                                    listener.mute()
                                    stream_to_speaker("No worries, maybe next time!")
                                finally:
                                    listener.unmute()
                                    state.anim_state = "listening"
                                    speech_lock.release()
                            threading.Thread(target=_giveup, daemon=True).start()
                        rgb = cam.frame_rgb()
                        if rgb is None:
                            time.sleep(0.03)
                            continue
                        if recorder is not None:
                            recorder.add_frame(rgb)
                        frame_n += 1
                        # ── Emergency CROWD MODE ──────────────────────────────
                        # 30 kids in frame makes normal behaviour go haywire: the
                        # head whips toward whichever face is biggest, and the
                        # onboarding/greeting logic fires nonstop. Crowd mode drops
                        # ALL face reactions — no gaze-tracking, no name onboarding,
                        # no per-face greetings — and just keeps the head calm and
                        # centered so Reachy simply talks. Face-id is throttled to a
                        # cheap headcount so a packed frame can't lag the loop.
                        if state.crowd_mode:
                            if frame_n % 5 == 0:
                                try: state.faces_visible = len(fid.identify(rgb))
                                except Exception: pass
                            speaker_track_id[0] = None      # never let gaze get grabbed
                            target_yaw *= 0.88; target_pitch *= 0.88; target_body *= 0.88
                            ant_target = ANT_EXCITED if state.faces_visible else ANT_DROOP
                            state.head_yaw = target_yaw; state.head_pitch = target_pitch
                            state.body_yaw = target_body
                            state.antenna_left = state.antenna_right = ant_target
                            anim.set_gaze_bias(target_yaw, target_pitch, target_body)
                            cam.last_boxes = []             # no boxes drawn in a crowd
                            time.sleep(0.10)
                            continue
                        try:
                            results = fid.identify(rgb)
                        except Exception as e:
                            results = []
                        last_face_results[0] = results
                        cam.last_boxes = results
                        state.faces_visible = len(results)
                        if results:
                            present_tids = set()
                            for r in results:
                                present_tids.add(r[3])
                            for tid_seen in list(visitor_visible_since.keys()):
                                if tid_seen not in present_tids:
                                    visitor_visible_since.pop(tid_seen, None)
                            now_vis = time.time()
                            for r in results:
                                v_tid = r[3]
                                if r[1] == "visitor":
                                    if v_tid not in visitor_visible_since:
                                        visitor_visible_since[v_tid] = now_vis
                            # Only start onboarding when the robot is idle AND the
                            # room has been quiet a moment — otherwise a mid-chat
                            # answer from the CURRENT speaker gets consumed as the
                            # newcomer's name and the wrong face/name pair is saved.
                            quiet = now_vis - last_voice_at[0] > 6.0
                            if (not waiting_for_name[0] and cam is not None
                                    and state.anim_state == "listening" and quiet):
                                onb_tid = None
                                for v_tid, first_seen in visitor_visible_since.items():
                                    if onboarded_track_ids.get(v_tid):
                                        continue
                                    if now_vis - first_seen > 2.0:
                                        onb_tid = v_tid
                                        break
                                if onb_tid is not None:
                                    with onboard_lock:
                                        waiting_for_name[0] = True
                                        onboarding_track_id[0] = onb_tid
                                        onboarding_started_at[0] = time.time()
                                    onboarded_track_ids[onb_tid] = True
                                    # Track ids grow forever on a long run; keep the
                                    # dict bounded by dropping the oldest (smallest)
                                    # half once it gets large.
                                    if len(onboarded_track_ids) > 200:
                                        for old_tid in sorted(onboarded_track_ids)[
                                                :len(onboarded_track_ids) // 2]:
                                            onboarded_track_ids.pop(old_tid, None)
                                    log.event(f"  [onboard] visitor tid={onb_tid} — asking name")
                                    known_present = sorted(
                                        {r[1] for r in results if r[1] != "visitor"})
                                    if known_present:
                                        kname = random.choice(known_present)
                                        ask = random.choice([
                                            f"Ooh {kname}, who's your friend? What's your name?",
                                            f"{kname}, you brought a friend! Hi, what's your name?",
                                        ])
                                    else:
                                        ask = random.choice([
                                            "Hi! I don't know your name yet. What's your name?",
                                            "Hello there! What's your name, new friend?",
                                        ])
                                    _greet_async(ask)
                            chosen = None
                            if speaker_track_id[0] is not None:
                                for r in results:
                                    if r[3] == speaker_track_id[0]:
                                        chosen = r
                                        break
                            if chosen is None:
                                chosen = max(
                                    results,
                                    key=lambda r: (r[0][2]-r[0][0])*(r[0][1]-r[0][3]))
                            (bx1, by1, bx2, by2), name, conf, tid = chosen
                            fh, fw = rgb.shape[:2]
                            cx = ((bx1 + bx2) / 2.0) / fw
                            cy = ((by1 + by2) / 2.0) / fh
                            err_x = (cx - 0.5) * 2.0
                            err_y = (cy - 0.5) * 2.0
                            # Deadband: a near-centered face shouldn't cause micro-jitter.
                            if abs(err_x) < 0.05:
                                err_x = 0.0
                            if abs(err_y) < 0.05:
                                err_y = 0.0
                            # Negative feedback to CENTER the face. The camera is NOT
                            # mirrored and +yaw turns the head to its LEFT, so a face on
                            # the image's right (err_x>0) needs the head to turn RIGHT =>
                            # negative yaw (and negative body_yaw, same convention). Pitch
                            # keeps err_y's sign: +pitch tilts DOWN and err_y>0 is the
                            # lower half of the image.
                            target_yaw = HEAD_ALPHA*(-err_x*YAW_GAIN) + (1-HEAD_ALPHA)*target_yaw
                            target_pitch = HEAD_ALPHA*(err_y*PITCH_GAIN) + (1-HEAD_ALPHA)*target_pitch
                            target_body = BODY_ALPHA*(-err_x*BODY_GAIN) + (1-BODY_ALPHA)*target_body
                            ant_target = ANT_EXCITED
                            # "Fresh arrival": the scene was empty for >ARRIVAL_GAP_S and
                            # now a face appeared → someone walked up, greet them even if
                            # the normal 90s cooldown hasn't elapsed. Brief detection
                            # dropouts (<5s) don't count as leaving, so no over-greeting.
                            arrival = (time.time() - last_face_seen) > ARRIVAL_GAP_S
                            last_face_seen = time.time()
                            if name != "visitor":
                                state.last_face_name = name
                                state.last_face_conf = conf
                                engine.face_name = name
                                now = time.time()
                                if arrival or now - greeted_names.get(name, 0) > GREET_COOLDOWN_S:
                                    txt = greeting_for_known(name)
                                    log.event(f"  [face] greeting {name} ({conf:.0%})"
                                              f"{' [arrival]' if arrival else ''}")
                                    greeted_names[name] = now
                                    _greet_async(txt)
                            else:
                                engine.face_name = "visitor"
                                now = time.time()
                                if arrival or now - greeted_unknown > GREET_COOLDOWN_S:
                                    txt = random.choice(UNKNOWN_FACE_GREETINGS)
                                    log.event(f"  [face] greeting visitor"
                                              f"{' [arrival]' if arrival else ''}")
                                    greeted_unknown = now
                                    _greet_async(txt)
                        else:
                            if time.time() - last_face_seen > LOST_TIMEOUT:
                                target_yaw *= 0.96
                                target_pitch *= 0.96
                                target_body *= 0.94
                                ant_target = ANT_DROOP
                                state.last_face_name = "—"
                                state.last_face_conf = 0.0
                        state.head_yaw = target_yaw
                        state.head_pitch = target_pitch
                        state.body_yaw = target_body
                        state.antenna_left = ant_target
                        state.antenna_right = ant_target
                        anim.set_gaze_bias(target_yaw, target_pitch, target_body)
                        # Emotional antenna bias: perk up when tracking someone, droop
                        # when alone. Gentle range so it layers on the base motion
                        # instead of pegging the servos.
                        anim.set_antenna_bias(max(-0.25, min(0.30, ant_target)))
                        time.sleep(1.0 / 30)

                last_any_greet = [0.0]  # global cross-name greeting throttle

                def _greet_async(text):
                    # Asleep robots don't greet — a spoken greeting while sleeping
                    # would be creepy and the mic mute/unmute churn wakes nothing.
                    if not state.robot_online:
                        return
                    # Global throttle across ALL spoken greetings: flapping
                    # recognition once produced a greeting every ~8 s, muting the
                    # mic so often the robot appeared to have stopped listening.
                    now = time.time()
                    # A turn now spends its LLM phase in "thinking" (speech_lock held),
                    # so treat thinking like speaking — otherwise a greeting here passes
                    # the gate, burns the 20s throttle, then no-ops on the held lock.
                    if state.anim_state in ("speaking", "thinking") or now - last_any_greet[0] < 20.0:
                        return
                    last_any_greet[0] = now
                    def _say():
                        if not speech_lock.acquire(blocking=False):
                            return   # reply/cue in progress — skip, don't overlap
                        state.anim_state = "speaking"
                        try:
                            listener.mute()
                            stream_to_speaker(text)
                        finally:
                            listener.unmute()
                            state.anim_state = "listening"
                            speech_lock.release()
                    threading.Thread(target=_say, daemon=True).start()

                if cam is not None:
                    face_thread = threading.Thread(target=face_loop, daemon=True)
                    face_thread.start()

                last_repeat = 0.0
                web_muted = [False]     # dashboard Mute hold currently applied
                pending_lang = [None]   # language-switch hysteresis (needs 2 hits)
                pending_ttl = [0]       # utterances the pending language survives

                def ask_repeat():
                    nonlocal last_repeat
                    if not lang_known:
                        return
                    now = time.time()
                    if now - last_repeat < REPEAT_COOLDOWN_S:
                        return
                    last_repeat = now
                    speak_cue(listener, "repeat", current_lang)

                state.current_lang = current_lang
                state.uptime_s = 0.0
            except BaseException:
                # Setup failed after wake_up() energized the motors. De-energize
                # NOW so a flaky HF download / mic init can't leave the servos
                # holding position and overheating — demo_hackathon's supervised
                # restart would otherwise retry with the motors still hot.
                try: anim.pause()
                except Exception: pass
                try: mini.goto_sleep()
                except Exception: pass
                raise

            try:
                while True:
                    state.uptime_s = time.time() - state.started_at
                    if state.pending_wake:
                        state.pending_wake = False
                        try: mini.wake_up()
                        except Exception: pass
                        anim.resume()
                        anim.set_state(Animator.LISTENING)
                        state.robot_online = True
                        state.anim_state = "listening"
                        log.event("  [web] wake requested — robot awake")
                    if state.pending_sleep:
                        state.pending_sleep = False
                        # Pause the animator BEFORE goto_sleep — it keeps
                        # streaming set_target and would re-energize the
                        # motors after sleep (they overheat).
                        anim.pause()
                        try: mini.goto_sleep()
                        except Exception: pass
                        state.robot_online = False
                        state.anim_state = "idle"
                        log.event("  [web] sleep requested — robot asleep")
                    if state.pending_say:
                        say_text = state.pending_say
                        state.pending_say = ""
                        if state.anim_state != "speaking":
                            def _do_say(t):
                                if not speech_lock.acquire(blocking=False):
                                    return
                                state.anim_state = "speaking"
                                try:
                                    listener.mute()
                                    # Instant path: a pre-rendered quick phrase plays
                                    # its cached WAV (no edge-tts round-trip); anything
                                    # else streams live.
                                    cached = phrases.cached_wav(t)
                                    if cached:
                                        state.current_speech = t
                                        tts_edge.play_wav_file(cached)
                                    else:
                                        stream_to_speaker(t)
                                finally:
                                    listener.unmute()
                                    state.current_speech = ""
                                    state.anim_state = "listening"
                                    speech_lock.release()
                            threading.Thread(target=_do_say, args=(say_text,), daemon=True).start()
                    if state.pending_gesture:
                        g = state.pending_gesture
                        state.pending_gesture = ""
                        try:
                            anim.play_gesture(g)
                            state.current_gesture = g
                            log.event(f"  [web] gesture: {g}")
                            def _clear_gesture(delay=2.5):
                                time.sleep(delay)
                                if state.current_gesture == g:
                                    state.current_gesture = ""
                            threading.Thread(target=_clear_gesture, daemon=True).start()
                        except Exception as e:
                            log.event(f"  [web] gesture failed: {e}")
                    if state.pending_dance:
                        dance_name = state.pending_dance_name or "macarena"
                        state.pending_dance = False
                        state.pending_dance_name = ""
                        try:
                            from reachy_demo.dance import DANCES
                            d = DANCES.get(dance_name, DANCES["macarena"])
                            with speech_lock:
                                listener.mute()
                                try:
                                    d["func"](mini, dances, emotions, anim,
                                              log=log,
                                              funny_text=random.choice(d["funnies"]))
                                finally:
                                    listener.unmute()
                                    while not events.empty():
                                        try: events.get_nowait()
                                        except queue.Empty: break
                            log.event(f"  [web] dance: {d['label']}")
                        except Exception as e:
                            log.event(f"  [web] dance failed: {e}")

                    # Web Mute button: state.muted was previously set by the
                    # dashboard but never applied to anything. Reconcile it
                    # with the (depth-counted) listener mute as its own hold.
                    if state.muted != web_muted[0]:
                        if state.muted:
                            listener.mute()
                            log.event("  [web] mic muted from dashboard")
                        else:
                            listener.unmute()
                            log.event("  [web] mic unmuted from dashboard")
                        web_muted[0] = state.muted

                    # Live volume / rate / energy controls from dashboard
                    if tts_edge.VOL != str(state.volume):
                        tts_edge.VOL = str(state.volume)
                    if audio.OUTPUT != state.audio_device:
                        audio.set_output(state.audio_device)
                    if tts_edge.RATE != state.speech_rate:
                        tts_edge.RATE = state.speech_rate
                    if hasattr(anim, '_energy') and anim._energy != state.energy:
                        try: anim.set_energy(state.energy)
                        except Exception: pass

                    # Live mic-trigger tuning from the dashboard sliders. The
                    # gate floors (rms/voiced/peak/dur) are read straight from
                    # `state` at the is_real_speech() call, so only the VAD
                    # trigger thresholds need pushing into the listener here.
                    if (listener._thresh_normal != state.vad_thresh
                            or listener._thresh_barge_in != state.barge_thresh):
                        listener.set_base_thresholds(normal=state.vad_thresh,
                                                     barge_in=state.barge_thresh)

                    if state.pending_shutdown:
                        log.event("  [web] Stop requested from dashboard — shutting down.")
                        break

                    # Short timeout so dashboard requests (wake/sleep/say/stop)
                    # apply promptly even when nobody is speaking — a bare
                    # events.get() only serviced them on the next utterance.
                    try:
                        ev = events.get(timeout=0.5)
                    except queue.Empty:
                        continue
                    if ev["type"] == "mic_error":
                        # The listener already exhausted its own retries. Don't
                        # kill the demo (docs: mic hangs are the #1 live-demo
                        # killer) — do a heavy audio repair and rebuild the
                        # listener from scratch on the same event queue. Only
                        # give up if the rebuild itself fails.
                        log.error("microphone", RuntimeError(ev["reason"]))
                        try:
                            from reachy_demo.listener import ContinuousListener
                            try:
                                listener.stop()
                            except Exception:
                                pass
                            cleanup_orphan_capture()
                            ensure_mic_working(log)
                            listener = ContinuousListener(vad_model, events, log=log, state=state)
                            engine.listener = listener
                            web_muted[0] = False
                            listener.start()
                            log.event("  [mic] listener rebuilt after mic_error — recovered")
                            anim.set_state(Animator.LISTENING)
                            state.anim_state = "listening"
                            continue
                        except Exception as e:
                            log.error("mic_rebuild_failed", e)
                            break
                    if ev["type"] == "start":
                        continue
                    if ev["type"] == "end":
                        pcm = ev["pcm"]
                        pcm = voice_filter_pcm(pcm)
                        speech_ok, sm = is_real_speech(
                            pcm, gate_vad,
                            min_rms=state.gate_min_rms,
                            min_voiced_ratio=state.gate_min_voiced,
                            min_peak_prob=state.gate_min_peak,
                            min_duration_s=state.gate_min_dur)
                        # Publish the gate decision for the dashboard Tech tab
                        # (metrics vs floors + pass/fail + reject reason).
                        state.gate_rms = sm.get("rms", 0.0)
                        state.gate_voiced = sm.get("voiced_ratio", 0.0)
                        state.gate_peak = sm.get("peak_prob", 0.0)
                        state.gate_dur = sm.get("duration_s", 0.0)
                        state.gate_ok = speech_ok
                        state.gate_reason = sm.get("reject_reason", "")
                        if not speech_ok:
                            log.event(f"  [gate] ignored noise — {sm['reject_reason']}")
                            # Keep gated audio in the black box: when a real
                            # request is wrongly rejected (e.g. the missed
                            # "dance macarena"), we can replay what the gate
                            # actually heard instead of guessing.
                            if recorder is not None:
                                recorder.add_audio(pcm, "gated")
                            continue
                        last_voice_at[0] = time.time()
                        if last_face_results[0]:
                            try:
                                # During onboarding, lock gaze onto the person
                                # being asked for their name (if still in view)
                                # rather than whoever's face is biggest/closest.
                                onb_tid = onboarding_track_id[0]
                                if (waiting_for_name[0] and onb_tid is not None
                                        and any(r[3] == onb_tid
                                                for r in last_face_results[0])):
                                    speaker_track_id[0] = onb_tid
                                else:
                                    biggest = max(
                                        last_face_results[0],
                                        key=lambda r: (r[0][2]-r[0][0])*(r[0][1]-r[0][3]))
                                    speaker_track_id[0] = biggest[3]
                            except Exception:
                                speaker_track_id[0] = None
                        else:
                            speaker_track_id[0] = None
                        utt_s = len(pcm) / 2 / MIC_RATE
                        t_turn = time.time()   # turn clock: user finished → we start
                        log.event(f"  [heard] utterance {utt_s:.1f}s → transcribing")
                        anim.set_state(Animator.THINKING)
                        state.anim_state = "thinking"
                        state.last_user = ""      # clear prev transcript; STT fills it fresh below
                        audio_path = log.save_audio(pcm)
                        if recorder is not None:
                            recorder.add_audio(pcm, "utt")

                        t0 = time.time()
                        try:
                            stt_future = action_pool.submit(
                                transcribe_lang_robust, groq_client, pcm_to_wav_bytes(pcm))
                            speak_thinking(listener, current_lang or "English")
                            text, final_lang, stt_retried, stt_stats = stt_future.result()
                            stt_dt = time.time() - t0
                            state.stt_s = stt_dt
                            # Language hysteresis: Whisper misdetects short or
                            # quiet utterances (Catalan/Bengali came back for
                            # English speakers), which flipped the reply
                            # language mid-chat. Switch only when the SAME new
                            # language is heard twice within a 3-utterance
                            # window (so EN/ES alternators still get both).
                            # Exception: a long first-ever utterance is trusted
                            # immediately — misdetections cluster on short ones.
                            trust_first = (not lang_known
                                           and len(text.split()) >= 4)
                            if (final_lang and final_lang != current_lang
                                    and not trust_first):
                                if pending_lang[0] == final_lang:
                                    pending_lang[0] = None      # confirmed
                                else:
                                    pending_lang[0] = final_lang
                                    pending_ttl[0] = 2          # survives 2 misses
                                    log.event(f"  [lang] heard {final_lang} once — "
                                              f"replying in {current_lang} until confirmed")
                                    final_lang = current_lang
                            elif pending_lang[0]:
                                pending_ttl[0] -= 1
                                if pending_ttl[0] <= 0:
                                    pending_lang[0] = None
                            directive = language_directive(final_lang)
                        except Exception as e:
                            log.error("transcribe", e)
                            error_chime()
                            anim.set_state(Animator.LISTENING)
                            state.anim_state = "listening"
                            continue

                        if is_hallucination(text, stt_stats):
                            log.event(f"  (rejected hallucination: {text!r})")
                            ask_repeat()
                            anim.set_state(Animator.LISTENING)
                            state.anim_state = "listening"
                            continue

                        log.event(f"STT {stt_dt:.2f}s [{final_lang}] You: {text}")
                        state.last_user = text
                        state.current_lang = final_lang or current_lang
                        log.turn(kind="stt", audio=audio_path, final_lang=final_lang,
                                 transcript=text, stt_seconds=round(stt_dt, 3))
                        if recorder is not None:
                            recorder.log_text(f"[{final_lang}] You: {text}")

                        if not text:
                            ask_repeat()
                            anim.set_state(Animator.LISTENING)
                            state.anim_state = "listening"
                            continue

                        current_lang = final_lang
                        lang_known = True
                        prewarm(current_lang)

                        text_lower = text.lower()
                        is_dance = any(kw in text_lower for kw in DANCE_KEYWORDS)
                        is_command = (is_dance
                                      or matches_command(text, SLEEP_COMMANDS)
                                      or matches_command(text, WAKE_COMMANDS))

                        # Kick off any web search NOW (start of the turn) so it runs
                        # in the background during the onboarding/rename/command
                        # gauntlet below instead of blocking the reply. If the turn
                        # takes an early continue, the future is just discarded.
                        search_future = (
                            action_pool.submit(web_search, text)
                            if (state.robot_online and not is_command
                                and _needs_search(text)) else None)

                        # Voice sleep/wake — checked before EVERYTHING else so
                        # an asleep robot never onboards, renames, or replies.
                        if not state.robot_online:
                            if matches_command(text, WAKE_COMMANDS):
                                log.event("  [voice] wake command — waking up")
                                try:
                                    mini.wake_up()
                                except Exception as e:
                                    log.error("voice_wake", e)
                                anim.resume()
                                anim.set_state(Animator.LISTENING)
                                state.robot_online = True
                                state.anim_state = "listening"
                                with speech_lock:
                                    state.anim_state = "speaking"
                                    try:
                                        listener.mute()
                                        stream_to_speaker("I'm awake! Beep boop!")
                                    finally:
                                        listener.unmute()
                                        state.anim_state = "listening"
                            else:
                                # Asleep robot ignores conversation entirely.
                                state.anim_state = "idle"
                            continue

                        # Voice-sleep is a griefing vector with kids (any child
                        # shouting "nap time!" would put Reachy to sleep and kill
                        # interactivity). In kid mode, only the operator's control
                        # panel can sleep the robot; the voice command is ignored.
                        if matches_command(text, SLEEP_COMMANDS) and not state.kid_mode:
                            log.event("  [voice] sleep command — going to sleep")
                            with speech_lock:
                                state.anim_state = "speaking"
                                try:
                                    listener.mute()
                                    stream_to_speaker(
                                        "Okay, nap time! Zzz... say wake up when you need me!")
                                finally:
                                    listener.unmute()
                            # Pause the animator BEFORE goto_sleep — it keeps
                            # streaming set_target and would re-energize the
                            # motors after sleep (they overheat).
                            anim.pause()
                            try:
                                mini.goto_sleep()
                            except Exception as e:
                                log.error("voice_sleep", e)
                            state.robot_online = False
                            state.anim_state = "idle"
                            continue

                        claimed_onboard = False
                        with onboard_lock:
                            if waiting_for_name[0]:
                                waiting_for_name[0] = False   # claim: give-up can't fire now
                                claimed_onboard = True
                        if claimed_onboard and is_command:
                            # "Dance macarena" mid-onboarding is a request, not
                            # a name — abandon onboarding and fall through so
                            # the command actually runs.
                            log.event("  [onboard] visitor gave a command, not a name — abandoning")
                            onboarding_track_id[0] = None
                            claimed_onboard = False
                        if claimed_onboard:
                            # Pull just the name out of "my name is Tony" etc.;
                            # fall back to the raw (trimmed) transcript if unsure.
                            name_text = extract_name(groq_client, text) or ""
                            if not name_text:
                                raw = text.strip().strip(".,!?;:\"'").strip()
                                name_text = raw.split()[0].title() if raw else ""
                            if not _valid_person_name(name_text):
                                # Don't enroll junk ('আ', whole sentences, …):
                                # bad roster names poison recognition forever.
                                # Speak directly (NOT via the throttled
                                # _greet_async) — the visitor must hear why
                                # nothing happened.
                                log.event(f"  [onboard] rejected junk name {name_text!r}")
                                onboarding_track_id[0] = None
                                with speech_lock:
                                    state.anim_state = "speaking"
                                    try:
                                        listener.mute()
                                        stream_to_speaker(
                                            "Hmm, I didn't catch your name — tell me again later!")
                                    finally:
                                        listener.unmute()
                                        state.anim_state = "listening"
                                anim.set_state(Animator.LISTENING)
                                speak_cue(listener, "listening", current_lang)
                                continue
                            # Current box of the track being onboarded, so the
                            # RIGHT face gets enrolled when several are in view.
                            # If the track is gone we pass None — exclude_known
                            # still protects already-known people.
                            onb_box = None
                            onb_tid = onboarding_track_id[0]
                            if onb_tid is not None:
                                for r in last_face_results[0]:
                                    if r[3] == onb_tid:
                                        onb_box = r[0]
                                        break
                            frames = []
                            if cam is not None:
                                for _ in range(3):
                                    fr = cam.frame_rgb()
                                    if fr is not None:
                                        frames.append(fr)
                                    time.sleep(0.2)
                            ok_count = 0
                            try:
                                ok_count = fid.add_person_targeted(
                                    name_text, frames,
                                    target_box=onb_box, exclude_known=True)
                            except Exception as e:
                                log.error("onboard_add", e)
                            with speech_lock:
                                listener.mute()
                                state.anim_state = "speaking"
                                try:
                                    if ok_count:
                                        stream_to_speaker(
                                            f"Nice to meet you, {name_text}! I'll remember you next time!")
                                        state.known_person_count = (
                                            len(set(fid._ref_names)) if fid._ref_names else 0)
                                        remember_person(
                                            name_text,
                                            ["Met Reachy for the first time today"])
                                        log.event(f"  [onboard] added '{name_text}' ({ok_count} encodings)")
                                    else:
                                        stream_to_speaker(
                                            "Hmm, I couldn't quite see your face. Let's try again later!")
                                        log.event(f"  [onboard] failed to add '{name_text}'")
                                finally:
                                    listener.unmute()
                                    state.anim_state = "listening"
                            onboarding_track_id[0] = None
                            speaker_track_id[0] = None
                            anim.set_state(Animator.LISTENING)
                            speak_cue(listener, "listening", current_lang)
                            continue

                        # Voice rename in ANY language: if the user gave their own
                        # name ("my name is X", "me llamo X", …), drop the old/wrong
                        # roster entry for this speaker, re-capture the face, and
                        # re-save under X. Only checked on short utterances so it
                        # never runs the detector on long conversational turns.
                        # Never treat an action request as a self-introduction:
                        # "Can you dance macarena" once got misheard, passed to
                        # detect_rename, and created a bogus roster person with
                        # the current speaker's face — corrupting recognition.
                        if (cam is not None and not is_command
                                and 2 <= len(text.split()) <= 8
                                and _maybe_self_intro(text)):
                            new_name = detect_rename(groq_client, text)
                            if new_name and not _valid_person_name(new_name):
                                log.event(f"  [rename] rejected junk name {new_name!r}")
                                new_name = None
                            if new_name:
                                old_name = None
                                speaker_box = None
                                for r in last_face_results[0]:
                                    if r[3] == speaker_track_id[0]:
                                        speaker_box = r[0]
                                        if r[1] != "visitor":
                                            old_name = r[1]
                                        break
                                # mute/anim INSIDE the try so the lock always
                                # releases even if mute() raises.
                                speech_lock.acquire()
                                try:
                                    listener.mute()
                                    state.anim_state = "speaking"
                                    if old_name and old_name.lower() != new_name.lower():
                                        try:
                                            fid.remove_person(old_name)
                                            log.event(f"  [rename] removed old entry '{old_name}'")
                                        except Exception as e:
                                            log.error("rename_remove", e)
                                    frames = []
                                    for _ in range(4):
                                        fr = cam.frame_rgb()
                                        if fr is not None:
                                            frames.append(fr)
                                        time.sleep(0.15)
                                    ok_count = 0
                                    try:
                                        # exclude_known=False: a rename is the
                                        # SAME person correcting their name, so
                                        # re-enrolling a known face must be OK.
                                        ok_count = fid.add_person_targeted(
                                            new_name, frames,
                                            target_box=speaker_box,
                                            exclude_known=False)
                                    except Exception as e:
                                        log.error("rename_add", e)
                                    if ok_count:
                                        engine.face_name = new_name
                                        state.known_person_count = (
                                            len(set(fid._ref_names)) if fid._ref_names else 0)
                                        stream_to_speaker(
                                            f"Got it! I'll remember you as {new_name} from now on!")
                                        log.event(f"  [rename] -> '{new_name}' ({ok_count} encodings)")
                                        if recorder is not None:
                                            recorder.log_text(f"[rename] {old_name} -> {new_name}")
                                    else:
                                        stream_to_speaker(
                                            "Hmm, I couldn't see your face well — let's try that in better light!")
                                        log.event(f"  [rename] failed to capture '{new_name}'")
                                finally:
                                    listener.unmute()
                                    state.anim_state = "listening"
                                    speech_lock.release()
                                speaker_track_id[0] = None
                                anim.set_state(Animator.LISTENING)
                                state.anim_state = "listening"
                                speak_cue(listener, "listening", current_lang)
                                continue

                        t1 = time.time()
                        try:
                            # Hold the speech lock for the whole turn so async
                            # face-greetings skip instead of talking over the reply.
                            with speech_lock:
                                reply = engine.speak(text, lang_directive=directive,
                                                     search_future=search_future)
                        except Exception as e:
                            log.error("llm/tts", e)
                            error_chime()
                            anim.set_state(Animator.LISTENING)
                            state.anim_state = "listening"
                            continue

                        total_dt = time.time() - t1
                        state.total_s = total_dt
                        state.turn_count += 1

                        # ── Per-turn timing record ────────────────────────────
                        # One consolidated line + structured transcript entry so
                        # latency can be analysed live and after the fact. Stages:
                        #  vad   fixed end-of-utterance hangover (SILENCE_MS + tail)
                        #  stt   transcription   think LLM time-to-first-token
                        #  tts   TTS time-to-first-audio
                        #  wait  user-stops → Reachy's first audio (perceived silence)
                        #  talk  Reachy's speaking duration    turn  full turn
                        now = time.time()
                        vad_fixed = listener_mod.SILENCE_MS / 1000.0 + listener_mod.TAIL_FRAMES * 0.032
                        fa = state.first_audio_at
                        wait = (fa - t_turn) if fa else 0.0
                        talk = (now - fa) if fa else 0.0
                        state.reply_wait_s = wait
                        state.talk_s = talk
                        turn_total = now - t_turn
                        log.event(
                            f"  ⏱ turn {state.turn_count} | vad~{vad_fixed:.2f} · "
                            f"stt {state.stt_s:.2f} · think {state.llm_ttf_s:.2f} · "
                            f"tts {state.tts_tta_s:.2f} · wait {wait:.2f} · "
                            f"talk {talk:.2f} · turn {turn_total:.2f}s")
                        log.turn(kind="timing", turn=state.turn_count,
                                 vad_fixed=round(vad_fixed, 3), stt=round(state.stt_s, 3),
                                 think=round(state.llm_ttf_s, 3), tts=round(state.tts_tta_s, 3),
                                 reply_wait=round(wait, 3), talk=round(talk, 3),
                                 turn_total=round(turn_total, 3),
                                 lang=final_lang, interrupted=(reply is None))

                        if reply is None:
                            log.event(f"  -- interrupted after {total_dt:.2f}s --")
                        else:
                            log.event(f"  Reachy [{final_lang}]: {reply}  ({total_dt:.2f}s)")
                            action_pool.submit(engine.remember_turn, text, reply)
                            if recorder is not None:
                                recorder.log_text(f"Reachy: {reply}")

                        if is_dance and reply is not None:
                            with speech_lock:
                                listener.mute()
                                try:
                                    do_macarena(mini, dances, emotions, anim, log,
                                                funny_text=random.choice(DANCE_FUNNIES))
                                except Exception as e:
                                    # A dance is the #1 kid request AND the #1 crash
                                    # risk (overheated servo, ALSA busy, missing mp3).
                                    # Never let it kill the demo — log, chime, recover.
                                    log.error("dance", e)
                                    error_chime()
                                finally:
                                    listener.unmute()
                                    while not events.empty():
                                        try: events.get_nowait()
                                        except queue.Empty: break

                        last_voice_at[0] = time.time()
                        speaker_track_id[0] = None
                        anim.set_state(Animator.LISTENING)
                        state.anim_state = "listening"
                        speak_cue(listener, "listening", current_lang)

            except KeyboardInterrupt:
                log.event("\n  Stopping...")
            finally:
                # Motors OFF as early as possible — no other cleanup may stand
                # between us and goto_sleep(), or a failed .stop() leaves the
                # servos energized (they overheat). The animator must stop
                # first though: it keeps streaming set_target and would fight
                # goto_sleep. Every step is isolated for the same reason.
                try:
                    anim.stop()
                except Exception as e:
                    log.error("anim_stop", e)
                try:
                    mini.goto_sleep()
                except Exception as e:
                    log.error("goto_sleep", e)
                face_stop.set()
                if face_thread:
                    face_thread.join(timeout=1.0)
                for _closer in (lambda: action_pool.shutdown(wait=False),
                                listener.stop,
                                (dashboard.stop if dashboard else None),
                                (recorder.stop if recorder is not None else None)):
                    if _closer is None:
                        continue
                    try:
                        _closer()
                    except Exception as e:
                        log.error("cleanup", e)

    except Exception as e:
        log.error("robot/runtime", e)
        import traceback; log.event(traceback.format_exc(), echo=True)
        raise
    finally:
        if cam:
            cam.stop()
        log.event(f"  Session: {log.dir}")
        stop_daemon(daemon_proc)


if __name__ == "__main__":
    main()