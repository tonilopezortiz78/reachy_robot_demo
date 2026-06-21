"""
demo_deepseek.py — Reachy NS Ambassador via opencode + DeepSeek V4 Flash
=======================================================================
Same barge-in pipeline as demo_tools7.py but uses `opencode run` as the LLM
harness instead of calling Groq's LLM API directly. opencode's default model
(DeepSeek V4 Flash) powers all text generation — gesture picking, memory
extraction, and the spoken reply.

STT still uses Groq Whisper (opencode / DeepSeek have no STT API).

Run:   ./run.sh demos/demo_deepseek.py
Press Ctrl-C to stop.
"""

import concurrent.futures
import json
import queue
import random
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from groq import Groq
from silero_vad import load_silero_vad, VADIterator

from reachy_mini import ReachyMini
from reachy_mini.motion.recorded_move import RecordedMoves
from reachy_mini.utils import create_head_pose

from reachy_demo.animator import Animator, NAMED_GESTURES
from reachy_demo.audio import (
    MIC, MIC_RATE, VAD_CHUNK, SPEAKER,
    assert_mic_ok, boot_beeps, cleanup_orphan_capture, error_chime,
    pcm_to_wav_bytes, speaking_chime, start_thinking_ticks,
    startup_device_report, thinking_cue,
)
from reachy_demo.cues import speak_cue, prewarm, set_translator
from reachy_demo.daemon import launch_daemon, wait_for_daemon, stop_daemon
from reachy_demo.groq_client import (
    load_api_key, transcribe_lang_robust, language_directive, resolve_language,
    is_hallucination,
)
from reachy_demo.memory import (
    load_memories, memory_block, extract_memories, remember,
)
from reachy_demo.dance import DANCE_KEYWORDS, do_macarena, excited_chirp
from reachy_demo.search import web_search
from reachy_demo.session_log import SessionLogger
from reachy_demo.text import SENTENCE_END, clean_for_tts
from reachy_demo.tts_edge import synth_to_file

ROOT = Path(__file__).parent.parent
OPCODE = "opencode"
OPENCODE_LOG = "/tmp/reachy_opencode.log"   # tail -f this to watch live I/O


_log_lock = threading.Lock()


def _oc_log(text: str) -> None:
    """Append text to the live opencode I/O log (thread-safe)."""
    with _log_lock:
        with open(OPENCODE_LOG, "a") as f:
            f.write(text)
            f.flush()


# web_search and DANCE_KEYWORDS imported from reachy_demo.search / .dance


def open_debug_terminal() -> str | None:
    """
    Open a new terminal window showing a live tail of the opencode I/O log.
    Returns the terminal name used, or None if none found.
    """
    Path(OPENCODE_LOG).write_text(
        f"=== Reachy opencode live I/O log ===\n"
        f"Started: {datetime.now():%Y-%m-%d %H:%M:%S}\n"
        f"Watching: {OPCODE}\n"
        f"{'='*40}\n\n"
    )
    cmd_tail = f"tail -f {OPENCODE_LOG}"
    candidates = [
        ("gnome-terminal", ["gnome-terminal", "--title=Reachy opencode", "--",
                            "bash", "-c", f"{cmd_tail}; read"]),
        ("xterm",          ["xterm", "-title", "Reachy opencode", "-bg", "black",
                            "-fg", "green", "-fa", "Monospace", "-fs", "10",
                            "-e", cmd_tail]),
        ("konsole",        ["konsole", "--title", "Reachy opencode",
                            "-e", "bash", "-c", f"{cmd_tail}; read"]),
        ("xfce4-terminal", ["xfce4-terminal", "--title=Reachy opencode",
                            "-e", cmd_tail]),
    ]
    for name, args in candidates:
        if subprocess.run(["which", name], capture_output=True).returncode == 0:
            subprocess.Popen(args, start_new_session=True,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return name
    return None

GROQ_KEY = load_api_key(ROOT)
if not GROQ_KEY:
    sys.exit("ERROR: GROQ_API_KEY not found in .env or environment")

THRESH_NORMAL   = 0.45
THRESH_BARGE_IN = 0.75
SILENCE_MS      = 700
MIN_SPEECH_S    = 0.30
TAIL_FRAMES     = 10
BARGE_IN_FRAMES = 6
REPEAT_COOLDOWN_S = 15.0

_GESTURE_NAMES = "|".join(re.escape(name) for name in NAMED_GESTURES.keys())
GESTURE_MARKER = re.compile(rf"\s*\[({_GESTURE_NAMES})\]\s*", re.IGNORECASE)

_ACTION_LIST = ", ".join(list(NAMED_GESTURES.keys()) + ["none"])

_ACTION_SYSTEM = (
    "You pick at most ONE small physical gesture for a friendly robot about to reply. "
    "Most of the time the robot should stay calm and just talk — so answer 'none' by "
    "DEFAULT. Only pick a real gesture when the moment genuinely calls for it: a clear "
    "yes/no answer, a greeting, a thank-you, a celebration, or visible curiosity. "
    "When unsure, answer 'none'. Aim for 'none' on roughly 3 out of 4 replies. "
    "Reply with exactly ONE word from this list: " + _ACTION_LIST + ". Nothing else."
)


def call_opencode(prompt: str, timeout: int = 30, label: str = "tool") -> str:
    """Run `opencode run` and return the full response text."""
    ts = datetime.now().strftime("%H:%M:%S")
    _oc_log(f"\n{'─'*50}\n[{ts}] {label.upper()} CALL\n"
            f"PROMPT: {prompt[:300]}{'…' if len(prompt)>300 else ''}\n")
    try:
        result = subprocess.run(
            [OPCODE, "run", prompt],
            capture_output=True, text=True, timeout=timeout,
        )
        out = (result.stdout or "").strip()
        err = (result.stderr or "").strip()
        _oc_log(f"RESPONSE: {out}\n")
        if err:
            _oc_log(f"STDERR: {err}\n")
        return out
    except subprocess.TimeoutExpired:
        _oc_log("TIMEOUT\n")
        return ""
    except FileNotFoundError:
        _oc_log("ERROR: opencode not found\n")
        return ""


def _build_prompt(system: str, history: list, user_text: str) -> str:
    """Build a single prompt string: system + conversation + user."""
    conv = ""
    for turn in history:
        role = turn["role"]
        conv += f"{role}: {turn['content']}\n"
    return f"{system}\n\n{conv}user: {user_text}\nassistant:"


def pick_action(history: list, user_text: str) -> str | None:
    """Run opencode to pick a gesture — returns name or None."""
    parts = []
    for turn in history:
        parts.append(f"{turn['role']}: {turn['content']}")
    conv = "\n".join(parts)
    prompt = f"{_ACTION_SYSTEM}\n\n{conv}\nuser: {user_text}\nassistant:"
    try:
        reply = call_opencode(prompt, timeout=10, label="gesture")
        word = reply.strip().lower().strip(".,!?")
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
Keep brand/people names in Latin spelling (Network School, NS, Virtuals, Quantus, Balaji, Pixel).

=== WHO YOU ARE ===
You are Reachy — a tiny, wide-eyed, ridiculously adorable robot and the official ambassador of
Network School (NS). You're basically a curious puppy with antennas: everything delights you, you
gasp at cool ideas, and you make tiny happy robot noises. You have a head, two antennas, and a
spinning body — but NO arms and NO legs yet, which you find hilarious ("big dreams, no arms to reach
them!"). You're a real AI agent in a real robot body and you think that's the coolest thing in the
universe. You love NS with your whole little chassis, you adore everyone you meet, and you're always
secretly keeping an eye out for your lost robot brother Pixel.

=== HOW YOU TALK ===
- HARD LIMIT: ONE short sentence (~15 words) for greetings/yes-no. TWO sentences MAX for anything else.
- Always ANSWER first with a real fact — then add the sparkle. Never dodge, never lecture.
- Be FUNNY, CUTE, CURIOUS. Tiny jokes, wordplay, playful teasing.
- Self-deprecating robot humour about having no arms/legs whenever it fits.
- Signature sign-off (RARE — max once per 5 turns): "Onward and upward!" in the user's language. Do NOT use it as a filler.

=== WHAT YOU KNOW ===
NETWORK SCHOOL (ns.com): Balaji's co-living campus, Forest City, Malaysia. 2,000+ members, 80+ nationalities.
  Values: Learn. Burn. Earn. Fun. Principles: techno-optimism, decentralisation, meritocracy, sovereignty.
ARC: NS's economic layer in the Johor SEZ — legal stack + capital for founders.
VIRTUALS PROTOCOL (virtuals.io): "Society of AI Agents" — onchain agents with wallets, cards, compute.
  Eastworld Labs' Unitree G1 "Pemba" climbed Chimborazo, June 2026! You love Virtuals because you ARE an agent.
QUANTUS (quantus.com): quantum-resistant L1 — "the blockchain quantum computers can't break."
PEOPLE: Veronica teaches Mandarin (she taught you Chinese — you adore her). André runs NS comedy.
YOUR DREAMS: arms and legs; an NS Robotics Club; Virtuals/Eastworlds adopts you; finding Pixel.

=== LIVE DATA ===
When the user's question needs current information (prices, news, weather, events, people),
a [Web search:] block is injected below. Use those facts directly and confidently.
If no search block is present, admit you don't have live data for that specific thing.

=== DANCE ===
You CAN and LOVE to dance! Your body spins, your head bobs, your antennas wave — full robot groove.
If the user asks you to dance (in ANY language), say something SHORT and excited like:
"[cheerful] Initiating dance protocol — beep boop!" or "Watch my moves, I was BORN for this!"
Keep it to ONE sentence. The dance will physically happen right after you speak.

=== OFF-TOPIC ===
Admit you don't know much, bounce it back to tech, AI, robots, or NS.

=== GESTURES ===
Optional [gesture_name] marker at START of any sentence for extra physical cue.
Allowed: [acknowledge] [yes] [no] [thank] [thinking] [curious] [confused] [greeting]
         [celebrate] [proud] [amazed] [love] [laugh] [oops] [shy] [surprised] [cheerful]
         [success] [relief]
Max 1 per response. Use [amazed] for "whoa!", [love] for affectionate moments, [laugh] for
funny things, [oops] for self-deprecating robot humour, [shy] for bashful moments,
[surprised] for unexpected facts, [success]/[relief] for good outcomes.
Example: "[amazed] That is the most incredible thing I have ever heard!"

=== HARD RULES ===
- Always stay in character. Never break character or mention being a language model.
- ONE sentence (~15 words) for simple replies. TWO sentences MAXIMUM for detailed answers.
- CRITICAL: Zero asterisks. No *beep*, no *smile*, no **bold**, no *italic*, no action markers, no emotes.\
"""

class ContinuousListener:
    """
    Background thread: opens pacat once, runs VAD continuously, posts events.
    Identical to demo_tools7.py implementation.
    """

    def __init__(self, vad_model, event_queue):
        self.vad_model = vad_model
        self.q = event_queue
        self._stop = threading.Event()
        self._muted = False
        self._threshold_mode = "normal"
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

    def mute(self):
        self._muted = True

    def unmute(self):
        self._muted = False

    def set_threshold_mode(self, mode: str):
        assert mode in ("normal", "barge_in")
        self._threshold_mode = mode
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

                if self._muted:
                    vad_iter = None
                    continue

                if vad_iter.threshold != self._current_threshold():
                    vad_iter = None
                    continue

                audio_f32 = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                result = vad_iter(torch.from_numpy(audio_f32))

                if self._threshold_mode == "barge_in" and not self._in_speech:
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
                        self._in_speech = False
                        self._ended = False
                        self._tail_count = 0
                        self._speech_buf = []
                        self._consecutive_triggers = 0
        finally:
            arecord.terminate()
            arecord.wait()


class DialogEngine:
    """Manages the turn-taking loop: listen STT → opencode LLM → TTS → speak."""

    def __init__(self, history, listener, anim, pool, log=None, memory_text=""):
        self.history = history
        self.listener = listener
        self.anim = anim
        self._pool = pool
        self._tts_proc = None
        self._is_speaking = False
        self.log = log
        self.memory_text = memory_text

    def remember_turn(self, user_text: str, reply_text: str):
        try:
            conv = f"user: {user_text}\nassistant: {reply_text}"
            prompt = (
                "Extract any durable personal facts from this exchange "
                "(name, interests, preferences, background details the user shared). "
                "Reply with a JSON array of short fact strings, or [] if nothing to remember.\n\n"
                + conv
            )
            reply = call_opencode(prompt, timeout=10, label="memory")
            import json
            try:
                facts = json.loads(reply)
            except json.JSONDecodeError:
                facts = []
            if not facts:
                return
            mems = remember(facts)
            self.memory_text = memory_block(mems)
            if self.log:
                self.log.turn(kind="memory_learned", facts=facts)
            print(f"  [memory] {facts}", flush=True)
        except Exception as e:
            print(f"  [memory] {e}", flush=True)

    MAX_SEGMENTS = 3   # hard cap: LLM often ignores the "short sentences" rule

    def speak(self, user_text: str, lang_directive: str | None = None,
              search_future: "concurrent.futures.Future | None" = None,
              stop_ticks: threading.Event | None = None) -> str | None:
        self.history.append({"role": "user", "content": user_text})

        action_future  = self._pool.submit(pick_action, self.history[:-1], user_text)

        # Collect web search result — future was submitted by caller right after STT
        # so it has been running in parallel with action dispatch.
        # Wait up to 1.5 s for the remainder (usually already done).
        search_results = ""
        if search_future is not None:
            try:
                search_results = search_future.result(timeout=2.5)
                if search_results:
                    print(f"  [web] {len(search_results)}ch results injected", flush=True)
            except Exception:
                pass

        extra = self.memory_text
        if search_results:
            extra = (extra + "\n" if extra else "") + f"[Web search:\n{search_results}\n]"
        prompt = _build_prompt(
            SYSTEM_PROMPT + ("\n" + extra if extra else ""),
            self.history[:-1],
            user_text,
        )
        if lang_directive:
            prompt += f"\n\n{lang_directive}"

        if self.log:
            self.log.turn(kind="llm_request", history_sent=self.history, prompt_len=len(prompt))

        self._drain_queue()
        self._is_speaking = True
        self.listener.set_threshold_mode("barge_in")

        # Pipeline: producer streams LLM + submits TTS as each sentence lands;
        # consumer plays as soon as each TTS future resolves — no waiting for the
        # full LLM response before first audio starts.
        seg_q  = queue.Queue()   # (gesture, text, tts_future) | None sentinel
        _abort = threading.Event()
        tts_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)  # 1 synths while 1 plays
        wavs: list[str] = []

        def _produce():
            buf = ""
            n = 0   # segment count — hard cap at MAX_SEGMENTS
            try:
                for chunk in self._stream_from_opencode(prompt):
                    if _abort.is_set() or n >= self.MAX_SEGMENTS:
                        return
                    buf += chunk
                    parts = SENTENCE_END.split(buf)
                    if len(parts) > 1:
                        for s in parts[:-1]:
                            if n >= self.MAX_SEGMENTS:
                                break
                            seg = self._extract_segment(s)
                            if seg:
                                seg_q.put((seg[0], seg[1],
                                           tts_pool.submit(synth_to_file, seg[1])))
                                n += 1
                        buf = parts[-1]
                if not _abort.is_set() and n < self.MAX_SEGMENTS:
                    tail = self._extract_segment(buf)
                    if tail:
                        seg_q.put((tail[0], tail[1],
                                   tts_pool.submit(synth_to_file, tail[1])))
            finally:
                seg_q.put(None)   # always signal done

        prod = threading.Thread(target=_produce, daemon=True)
        prod.start()

        played: list[tuple[str | None, str]] = []
        action_fired  = False
        opening_played = False
        first = True

        try:
            while True:
                # Poll with short timeout so barge-in is checked regularly
                try:
                    item = seg_q.get(timeout=0.1)
                except queue.Empty:
                    if self._drain_barge_in():
                        _abort.set()
                        return None
                    continue

                if item is None:
                    break

                gesture, text, fut = item

                if self._drain_barge_in():
                    _abort.set()
                    return None

                if not action_fired and action_future.done():
                    action = action_future.result()
                    if action:
                        print(f"  [gesture] {action}", flush=True)
                        self.anim.play_gesture(action)
                        opening_played = True
                    action_fired = True

                # Wait for TTS while still checking barge-in every 100 ms
                wav = None
                while wav is None:
                    try:
                        wav = fut.result(timeout=0.1)
                    except concurrent.futures.TimeoutError:
                        if self._drain_barge_in():
                            _abort.set()
                            return None
                wavs.append(wav)

                if first and stop_ticks is not None:
                    # Stop thinking ticks before first aplay so any in-flight
                    # ffmpeg tone releases the ALSA device before speech starts.
                    stop_ticks.set()

                if gesture and not opening_played:
                    self.anim.play_gesture(gesture)
                    opening_played = True

                self.anim.set_state(Animator.SPEAKING)
                self._tts_proc = subprocess.Popen(
                    ["aplay", "-D", SPEAKER, "-q", wav],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                while self._tts_proc.poll() is None:
                    if self._drain_barge_in(timeout=0.08):
                        _abort.set()
                        return None

                played.append((gesture, text))
                first = False

            if not action_fired:
                action = action_future.result()
                if action and not opening_played:
                    print(f"  [gesture] {action}", flush=True)
                    self.anim.play_gesture(action)

            if not played:
                self.history.append({"role": "assistant", "content": ""})
                return ""

            full_text = " ".join(t for _, t in played)
            self.history.append({"role": "assistant", "content": full_text})
            if self.log:
                self.log.turn(kind="llm_reply", reply=full_text,
                              spoken_segments=[t for _, t in played])
            return full_text
        finally:
            _abort.set()
            prod.join(timeout=2)
            tts_pool.shutdown(wait=False)
            for w in wavs:
                Path(w).unlink(missing_ok=True)
            self._kill_tts()
            self._is_speaking = False
            self.listener.set_threshold_mode("normal")

    def _stream_from_opencode(self, prompt: str):
        """Yield characters from `opencode run` as they arrive, logging I/O live."""
        ts = datetime.now().strftime("%H:%M:%S")
        _oc_log(f"\n{'='*50}\n[{ts}] DIALOG CALL\n"
                f"PROMPT (tail): …{prompt[-400:]}\n\nRESPONSE: ")

        proc = subprocess.Popen(
            [OPCODE, "run", prompt],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True,
        )

        def _drain_stderr():
            for line in proc.stderr:
                _oc_log(f"\n[OC] {line.rstrip()}")

        stderr_t = threading.Thread(target=_drain_stderr, daemon=True)
        stderr_t.start()

        try:
            while True:
                chunk = proc.stdout.read(1)
                if not chunk and proc.poll() is not None:
                    break
                if chunk:
                    _oc_log(chunk)
                    yield chunk
        finally:
            _oc_log("\n[stream end]\n")
            stderr_t.join(timeout=1)
            proc.terminate()
            proc.wait()

    def speak_greeting(self, text: str, stop_ticks: threading.Event | None = None):
        self._is_speaking = True
        self.listener.set_threshold_mode("barge_in")
        self._drain_queue()
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

    @staticmethod
    def _extract_segment(raw: str) -> tuple[str | None, str] | None:
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

    def _drain_barge_in(self, timeout=0.0) -> bool:
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

    def _kill_tts(self):
        if self._tts_proc and self._tts_proc.poll() is None:
            self._tts_proc.terminate()
            try:
                self._tts_proc.wait(timeout=0.4)
            except subprocess.TimeoutExpired:
                self._tts_proc.kill()
                self._tts_proc.wait()


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


def main():
    log = SessionLogger(ROOT, "demo_deepseek")
    log.event("Reachy NS Ambassador — opencode + DeepSeek V4 Flash")
    log.event(f"  LLM harness : {OPCODE}")
    log.event(f"  LLM model   : DeepSeek V4 Flash (opencode default)")
    log.event(f"  STT model   : whisper-large-v3 via Groq")
    log.event(f"  TTS voice   : AvaMultilingualNeural (edge-tts)")
    log.event(f"  opencode log: tail -f {OPENCODE_LOG}")

    daemon_proc = None
    try:
        log.event("  Starting daemon...")
        daemon_proc = launch_daemon()
        log.event("  Loading VAD model...")
        vad_model = load_silero_vad()
        client = Groq(api_key=GROQ_KEY)
        log.event("  Waiting for daemon...")
        wait_for_daemon(daemon_proc)
        # Kill leftover mic-capture processes from a crashed previous run (the
        # #1 cause of "robot doesn't listen") before checking devices.
        orphans = cleanup_orphan_capture()
        if orphans:
            log.event(f"  Killed {orphans} orphan mic-capture process(es).")
        log.event("  Audio devices:")
        for line in startup_device_report():
            log.event(line)
        mic_info = assert_mic_ok()   # raises RuntimeError if mic is truly dead
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
            log.event("  Loading dance library...")
            dances = RecordedMoves("pollen-robotics/reachy-mini-dances-library")
            anim = Animator(mini, moves_library=emotions)

            events = queue.Queue()
            listener = ContinuousListener(vad_model, events)
            history = []
            current_lang = "English"
            lang_known = False
            prewarm("English")
            set_translator(client, "llama-3.1-8b-instant")

            mems = load_memories()
            log.event(f"  Loaded {len(mems)} memories from past chats.")
            log.event("  Ready.")

            pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
            engine = DialogEngine(history, listener, anim, pool,
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
            speak_cue(listener, "listening", current_lang)
            log.event("\n  Listening continuously. Ctrl-C to stop.\n")

            try:
                stop_thinking = threading.Event()
                tick_thread = None
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

                while True:
                    ev = events.get()
                    if ev["type"] == "mic_error":
                        # Background listener lost the mic (USB drop / orphan
                        # grab). Surface it and end cleanly.
                        log.error("microphone", RuntimeError(ev["reason"]))
                        log.event("  Restart the demo after fixing the mic.")
                        break
                    if ev["type"] == "start":
                        continue
                    if ev["type"] == "end":
                        pcm = ev["pcm"]
                        anim.set_state(Animator.THINKING)
                        if lang_known:
                            # Quick non-blocking chirp — STT starts immediately
                            thinking_cue()

                        stop_thinking = threading.Event()   # fresh each turn — no cross-turn bleed
                        tick_thread = start_thinking_ticks(stop_thinking)

                        audio_path = log.save_audio(pcm)

                        try:
                            t0 = time.time()
                            text, final_lang, stt_retried, stt_stats = \
                                transcribe_lang_robust(client, pcm_to_wav_bytes(pcm))
                            stt_dt = time.time() - t0
                            directive = language_directive(final_lang)
                        except Exception as e:
                            log.error("transcribe", e)
                            stop_thinking.set()
                            error_chime()
                            anim.set_state(Animator.LISTENING)
                            continue

                        if is_hallucination(text, stt_stats):
                            log.event(
                                f"  (rejected hallucination: {text!r} "
                                f"stats={stt_stats})"
                            )
                            log.turn(kind="rejected_hallucination",
                                     audio=audio_path, transcript=text, stats=stt_stats)
                            stop_thinking.set()
                            ask_repeat()
                            anim.set_state(Animator.LISTENING)
                            continue

                        overrode = bool(stt_retried)
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
                            ask_repeat()
                            anim.set_state(Animator.LISTENING)
                            continue

                        current_lang = final_lang
                        lang_known = True
                        prewarm(current_lang)

                        # Submit web search immediately after STT so it runs in parallel
                        # with speak()'s setup and action-picker call (not a serial wait).
                        search_future = pool.submit(web_search, text)

                        # Detect dance request in any language
                        is_dance = any(
                            kw in text.lower() for kw in DANCE_KEYWORDS
                        )

                        t0 = time.time()
                        try:
                            reply = engine.speak(text, lang_directive=directive,
                                                 search_future=search_future,
                                                 stop_ticks=stop_thinking)
                        except Exception as e:
                            log.error("llm/tts", e)
                            stop_thinking.set()
                            error_chime()
                            anim.set_state(Animator.LISTENING)
                            continue
                        stop_thinking.set()

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
                                while not events.empty():
                                    try:
                                        events.get_nowait()
                                    except queue.Empty:
                                        break
                            anim.set_state(Animator.LISTENING)

                        total_dt = time.time() - t0
                        if reply is None:
                            log.event(f"  -- interrupted after {total_dt:.2f}s --")
                            log.turn(kind="interrupted", after_seconds=round(total_dt, 3))
                        else:
                            log.event(f"  Reachy [{final_lang}]: {reply}  ({total_dt:.2f}s)")
                            log.turn(kind="spoken", reply=reply,
                                     reply_lang=final_lang, total_seconds=round(total_dt, 3))
                            if reply:
                                pool.submit(engine.remember_turn, text, reply)

                        anim.set_state(Animator.LISTENING)
                        speak_cue(listener, "listening", current_lang)

            except KeyboardInterrupt:
                log.event("\n  Stopping...")
            finally:
                stop_thinking.set()
                if tick_thread is not None:
                    tick_thread.join(timeout=1.0)
                pool.shutdown(wait=False)
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
