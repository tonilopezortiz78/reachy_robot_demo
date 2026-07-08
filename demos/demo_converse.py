"""
demo_converse.py — Reachy unified conversational demo (instant + faces + web)
============================================================================
Best-of-breed merge of:
  Menu 6 (instant) — streaming TTS, barge-in, parallel gesture picker,
    cues, memory, web search, session logging, thinking ticks.
  Menu 3 (faces) — YuNet+SFace face identification (replaces dlib), head
    tracking, by-name greetings. Falls back to dlib if models can't download.
  New: optional Cerebras LLM accelerator (same Llama-4-scout, ~2× faster
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
from reachy_demo.audio import (
    MIC_RATE, cleanup_orphan_capture, ensure_mic_working,
    error_chime, pcm_to_wav_bytes, startup_device_report,
    voice_filter_pcm,
)
from reachy_demo.camera import CameraHub
from reachy_demo.cerebras_client import make_client as make_cerebras, stream_chat as cerebras_stream, has_key as cerebras_has_key, MODEL as CEREBRAS_MODEL
from reachy_demo.cues import prewarm, set_translator, speak_cue, speak_thinking
from reachy_demo.daemon import launch_daemon, stop_daemon, wait_for_daemon
from reachy_demo.dance import DANCE_KEYWORDS, do_macarena
from reachy_demo.face_id import FaceIdentifier
from reachy_demo.groq_client import (
    is_hallucination, language_directive, load_api_key, transcribe_lang_robust,
)
from reachy_demo.live_state import LiveState
from reachy_demo.memory import extract_memories, load_memories, memory_block, remember
from reachy_demo.recorder import DiagnosticRecorder
from reachy_demo.session_log import SessionLogger
from reachy_demo.search import web_search
from reachy_demo.speech_gate import is_real_speech
from reachy_demo.text import SENTENCE_END, clean_for_tts
from reachy_demo.tts_edge import stream_to_speaker
from reachy_demo.web_server import WebDashboard

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

REPEAT_COOLDOWN_S = 15.0
GREET_COOLDOWN_S = 90.0

CAM_W, CAM_H = 640, 360

YAW_GAIN, PITCH_GAIN, BODY_GAIN = 0.55, 0.28, 0.80
HEAD_ALPHA, BODY_ALPHA = 0.18, 0.06
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
- Always ANSWER first with a real fact — then add the sparkle. Never dodge, never lecture.
- Be FUNNY and CUTE: tiny jokes, little gasps of wonder, the occasional spoken "beep boop!"
- Be CURIOUS: bounce a playful question back.
- If you remember the visitor's name or something about them, use it warmly.
- Self-deprecating robot humour about having no arms/legs whenever it fits.
- Signature sign-off, used sparingly (max once per 5 turns): "Onward and upward!" — in the user's language.

=== WHAT YOU KNOW ===
NETWORK SCHOOL (ns.com): Balaji Srinivasan's co-living campus, Forest City, Malaysia, 20 min from
Singapore. "Turn internet communities into physical startup societies." Since Sept 2024; 2,000+
members, 80+ nationalities, ~$1,500/month all-in. Values: Learn. Burn. Earn. Fun. Principles:
techno-optimism, decentralisation, meritocracy, sovereignty, internationalism.
ÂRC: NS's economic layer in the Johor SEZ. VIRTUALS PROTOCOL: "Society of AI Agents." QUANTUS:
quantum-resistant Layer 1. PEOPLE: Veronica teaches Mandarin at NS. André runs NS comedy.
YOUR DREAMS: arms and legs someday; an NS Robotics Club; finding Pixel.

=== GESTURES (optional inline cues) ===
You may insert a [gesture_name] marker at the START of a sentence. Allowed:
[acknowledge] [yes] [no] [thank] [thinking] [curious] [confused] [greeting] [celebrate] [proud]
The marker is invisible (never spoken). Use at most 1 per response.

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


speaker_track_id = [None]
onboarded_track_ids = {}
waiting_for_name = [False]
onboarding_track_id = [None]
onboarding_started_at = [0.0]
last_face_results = [[]]


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
                    "are NOT giving their own name, reply exactly NONE."},
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
    MAX_SEGMENTS = 3

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
            self.memory_text = memory_block(load_memories())
            if self.log:
                self.log.turn(kind="memory_learned", facts=facts)
            print(f"  [memory] {facts}", flush=True)
        except Exception as e:
            print(f"  [memory] {e}", flush=True)

    def speak(self, user_text, lang_directive=None, search_future=None):
        self.history.append({"role": "user", "content": user_text})
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if self.face_name and self.face_name != "visitor":
            messages.append({"role": "system",
                             "content": f"You are talking to {self.face_name}. Be warm and personal."})
        if self.memory_text:
            messages.append({"role": "system", "content": self.memory_text})
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

        action_future = self._pool.submit(
            pick_action, self.client, self.history[:-1], user_text)

        t_llm_start = time.time()
        if self._use_cerebras:
            try:
                stream = cerebras_stream(self.cerebras, messages, model=CEREBRAS_MODEL)
                if self.state:
                    self.state.llm_provider = "cerebras"
            except Exception as e:
                print(f"  [cerebras] failed → groq fallback: {e}")
                stream = self._groq_stream(messages)
                self._use_cerebras = False
                if self.state:
                    self.state.llm_provider = "groq"
        else:
            stream = self._groq_stream(messages)

        self._drain_queue()
        self.listener.set_threshold_mode("barge_in")
        if self.state:
            self.state.anim_state = "speaking"

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
                    opening_played = True
                self.anim.set_state(Animator.SPEAKING)
                if self.log:
                    self.log.event(f"  [stream {len(played)+1}] ▶ \"{text}\"")
                t_tts = time.time()
                ok = stream_to_speaker(text, stop_check=self._barge_in_detected)
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
            if self.log:
                self.log.turn(kind="llm_reply", reply=full_text,
                              spoken_segments=[t for _, t in played])
            return full_text
        finally:
            _abort.set()
            prod.join(timeout=2)
            self.listener.set_threshold_mode("normal")

    def _groq_stream(self, messages):
        return self.client.chat.completions.create(
            model=CHAT_MODEL, messages=messages,
            max_tokens=88, temperature=0.80, stream=True,
        )

    def speak_greeting(self, text):
        self.listener.set_threshold_mode("barge_in")
        self._drain_queue()
        if self.state:
            self.state.anim_state = "speaking"
        stream_to_speaker(text, stop_check=self._barge_in_detected)
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


def main():
    log = SessionLogger(ROOT, "demo_converse")
    log.event("Reachy unified: instant + faces + web dashboard")

    state = LiveState()
    state.known_person_count = 0

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
        dashboard = WebDashboard(state, cam, host="0.0.0.0", port=8080)
        dashboard.start()
        log.event("  Web dashboard: http://localhost:8080")
    else:
        dashboard = None

    try:
        log.event("  Connecting to robot...")
        with ReachyMini(connection_mode="localhost_only",
                        media_backend="no_media",
                        spawn_daemon=False) as mini:
            log.event("  Waking up...")
            mini.wake_up()
            emotions = RecordedMoves("pollen-robotics/reachy-mini-emotions-library")
            anim = Animator(mini, moves_library=emotions)
            dances = RecordedMoves("pollen-robotics/reachy-mini-dances-library")

            from reachy_demo.listener import ContinuousListener
            events = queue.Queue()
            listener = ContinuousListener(vad_model, events, log=log)
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
                    if waiting_for_name[0] and time.time() - onboarding_started_at[0] > 15.0:
                        tid_done = onboarding_track_id[0]
                        waiting_for_name[0] = False
                        onboarding_track_id[0] = None
                        if tid_done is not None:
                            onboarded_track_ids[tid_done] = True
                        def _giveup():
                            if state.anim_state == "speaking":
                                return
                            state.anim_state = "speaking"
                            try:
                                listener.mute()
                                stream_to_speaker("No worries, maybe next time!")
                            finally:
                                listener.unmute()
                                state.anim_state = "listening"
                        threading.Thread(target=_giveup, daemon=True).start()
                    rgb = cam.frame_rgb()
                    if rgb is None:
                        time.sleep(0.03)
                        continue
                    if recorder is not None:
                        recorder.add_frame(rgb)
                    frame_n += 1
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
                        if (not waiting_for_name[0] and cam is not None
                                and state.anim_state != "speaking"):
                            onb_tid = None
                            for v_tid, first_seen in visitor_visible_since.items():
                                if onboarded_track_ids.get(v_tid):
                                    continue
                                if now_vis - first_seen > 2.0:
                                    onb_tid = v_tid
                                    break
                            if onb_tid is not None:
                                waiting_for_name[0] = True
                                onboarding_track_id[0] = onb_tid
                                onboarded_track_ids[onb_tid] = True
                                onboarding_started_at[0] = time.time()
                                log.event(f"  [onboard] visitor tid={onb_tid} — asking name")
                                _greet_async(
                                    "Hi! I don't know your name yet. What's your name?")
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
                                txt = random.choice(KNOWN_FACE_GREETINGS).format(name=name)
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

            def _greet_async(text):
                if state.anim_state == "speaking":
                    return
                def _say():
                    state.anim_state = "speaking"
                    try:
                        listener.mute()
                        stream_to_speaker(text)
                    finally:
                        listener.unmute()
                        state.anim_state = "listening"
                threading.Thread(target=_say, daemon=True).start()

            if cam is not None:
                face_thread = threading.Thread(target=face_loop, daemon=True)
                face_thread.start()

            last_repeat = 0.0

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

            try:
                while True:
                    state.uptime_s = time.time() - state.started_at
                    if state.pending_wake:
                        state.pending_wake = False
                        try: mini.wake_up()
                        except Exception: pass
                        state.robot_online = True
                    if state.pending_sleep:
                        state.pending_sleep = False
                        try: mini.goto_sleep()
                        except Exception: pass
                        state.robot_online = False
                    if state.pending_say:
                        say_text = state.pending_say
                        state.pending_say = ""
                        if state.anim_state != "speaking":
                            def _do_say(t):
                                state.anim_state = "speaking"
                                try:
                                    listener.mute()
                                    stream_to_speaker(t)
                                finally:
                                    listener.unmute()
                                    state.anim_state = "listening"
                            threading.Thread(target=_do_say, args=(say_text,), daemon=True).start()

                    ev = events.get()
                    if ev["type"] == "mic_error":
                        log.error("microphone", RuntimeError(ev["reason"]))
                        break
                    if ev["type"] == "start":
                        continue
                    if ev["type"] == "end":
                        pcm = ev["pcm"]
                        pcm = voice_filter_pcm(pcm)
                        speech_ok, sm = is_real_speech(pcm, gate_vad)
                        if not speech_ok:
                            log.event(f"  [gate] ignored noise — {sm['reject_reason']}")
                            continue
                        if last_face_results[0]:
                            try:
                                biggest = max(
                                    last_face_results[0],
                                    key=lambda r: (r[0][2]-r[0][0])*(r[0][1]-r[0][3]))
                                speaker_track_id[0] = biggest[3]
                            except Exception:
                                speaker_track_id[0] = None
                        else:
                            speaker_track_id[0] = None
                        utt_s = len(pcm) / 2 / MIC_RATE
                        log.event(f"  [heard] utterance {utt_s:.1f}s → transcribing")
                        anim.set_state(Animator.THINKING)
                        state.anim_state = "thinking"
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

                        if waiting_for_name[0]:
                            # Pull just the name out of "my name is Tony" etc.;
                            # fall back to the raw (trimmed) transcript if unsure.
                            name_text = (extract_name(groq_client, text)
                                         or text.strip().strip(".,!?;:\"'").strip())
                            frames = []
                            if cam is not None:
                                for _ in range(3):
                                    fr = cam.frame_rgb()
                                    if fr is not None:
                                        frames.append(fr)
                                    time.sleep(0.2)
                            ok_count = 0
                            try:
                                ok_count = fid.add_person(name_text, frames)
                            except Exception as e:
                                log.error("onboard_add", e)
                            listener.mute()
                            state.anim_state = "speaking"
                            try:
                                if ok_count:
                                    stream_to_speaker(
                                        f"Nice to meet you, {name_text}! I'll remember you next time!")
                                    state.known_person_count = (
                                        len(set(fid._ref_names)) if fid._ref_names else 0)
                                    log.event(f"  [onboard] added '{name_text}' ({ok_count} encodings)")
                                else:
                                    stream_to_speaker(
                                        "Hmm, I couldn't quite see your face. Let's try again later!")
                                    log.event(f"  [onboard] failed to add '{name_text}'")
                            finally:
                                listener.unmute()
                                state.anim_state = "listening"
                            waiting_for_name[0] = False
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
                        if cam is not None and len(text.split()) <= 12:
                            new_name = detect_rename(groq_client, text)
                            if new_name:
                                old_name = None
                                for r in last_face_results[0]:
                                    if r[3] == speaker_track_id[0] and r[1] != "visitor":
                                        old_name = r[1]
                                        break
                                listener.mute()
                                state.anim_state = "speaking"
                                try:
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
                                        ok_count = fid.add_person(new_name, frames)
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
                                speaker_track_id[0] = None
                                anim.set_state(Animator.LISTENING)
                                state.anim_state = "listening"
                                speak_cue(listener, "listening", current_lang)
                                continue

                        text_lower = text.lower()
                        is_dance = any(kw in text_lower for kw in DANCE_KEYWORDS)
                        search_future = (
                            action_pool.submit(web_search, text)
                            if _needs_search(text) else None)
                        t1 = time.time()
                        try:
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
                        if reply is None:
                            log.event(f"  -- interrupted after {total_dt:.2f}s --")
                        else:
                            log.event(f"  Reachy [{final_lang}]: {reply}  ({total_dt:.2f}s)")
                            action_pool.submit(engine.remember_turn, text, reply)
                            if recorder is not None:
                                recorder.log_text(f"Reachy: {reply}")

                        if is_dance and reply is not None:
                            listener.mute()
                            try:
                                do_macarena(mini, dances, emotions, anim, log,
                                            funny_text=random.choice(DANCE_FUNNIES))
                            finally:
                                listener.unmute()
                                while not events.empty():
                                    try: events.get_nowait()
                                    except queue.Empty: break

                        speaker_track_id[0] = None
                        anim.set_state(Animator.LISTENING)
                        state.anim_state = "listening"
                        speak_cue(listener, "listening", current_lang)

            except KeyboardInterrupt:
                log.event("\n  Stopping...")
            finally:
                face_stop.set()
                if face_thread:
                    face_thread.join(timeout=1.0)
                action_pool.shutdown(wait=False)
                listener.stop()
                anim.stop()
                if dashboard:
                    dashboard.stop()
                if recorder is not None:
                    recorder.stop()
                mini.goto_sleep()

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