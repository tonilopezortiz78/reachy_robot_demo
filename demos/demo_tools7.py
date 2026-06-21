"""
demo_tools7.py — Reachy NS Ambassador: parallel LLM-driven gesture tools
=========================================================================
Same barge-in pipeline as demo_dialog.py but adds a second parallel Groq
call that autonomously selects a physical gesture for each turn — independent
of the speech content, acting like a true robot "tool."

Two concurrent Groq calls fire the moment STT completes:
  A) Streaming LLaMA → TTS pipeline → speaker        (speech, ~200ms first token)
  B) Non-streaming LLaMA action picker → gesture      (fires in ~150ms, leads speech)

The gesture fires before the first spoken word — the robot "acts then speaks."
The LLM can also embed [gesture_name] markers in sentences for additional cues.

Voice: en-US-AvaMultilingualNeural — PITCH +48Hz for a cute, childlike robot.
       Same voice in all languages; auto-detects and replies in user's language.
       Spoken turn cues ("I'm listening" / "Let me think") in the user's language.

All speech processing uses Groq (Whisper STT + LLaMA LLM). No local model needed.
Run:   ./run.sh demos/demo_tools7.py
Press Ctrl-C to stop.
"""

import concurrent.futures
import math
import queue
import random
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
from ddgs import DDGS
from groq import Groq
from silero_vad import load_silero_vad, VADIterator

from reachy_mini import ReachyMini
from reachy_mini.motion.recorded_move import RecordedMoves
from reachy_mini.utils import create_head_pose

from reachy_demo.animator import Animator, NAMED_GESTURES
from reachy_demo.audio import (
    MIC, MIC_RATE, VAD_CHUNK, SPEAKER,
    boot_beeps, error_chime, pcm_to_wav_bytes,
    speaking_chime, start_thinking_ticks,
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
from reachy_demo.session_log import SessionLogger
from reachy_demo.text import SENTENCE_END, clean_for_tts
from reachy_demo.tts_edge import synth_to_file  # PITCH +48Hz set in tts_edge.py

ROOT = Path(__file__).parent.parent

# ── Sound effects ─────────────────────────────────────────────────────────────

def _chirp(f0, f1, dur, vol=0.65):
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi",
         "-i", f"aevalsrc=sin(2*PI*({f0}*t+({f1}-{f0})*t*t/(2*{dur})))*{vol}:c=mono:s=22050",
         "-t", str(dur), "-f", "alsa", SPEAKER],
        check=False,
    )


def _excited_chirp():
    """Two ascending sweeps — used before/during dance to signal excitement."""
    _chirp(500, 1800, 0.14, vol=0.75)
    time.sleep(0.04)
    _chirp(800, 2200, 0.12, vol=0.85)


# ── Web search (DuckDuckGo, no API key) ──────────────────────────────────────

_SEARCH_FILLER = re.compile(
    r"^\s*(?:can\s+you\s+|please\s+|tell\s+me\s+|what\s+(?:is\s+|are\s+)?|"
    r"how\s+much\s+(?:is\s+)?|give\s+me\s+(?:the\s+)?|"
    r"look\s+up\s+|search\s+(?:for\s+)?|find\s+(?:me\s+)?)+",
    re.IGNORECASE,
)


def _clean_query(text: str) -> str:
    return _SEARCH_FILLER.sub("", text).strip(" ?.,!")


def web_search(query: str, max_results: int = 3) -> str:
    """DuckDuckGo search — returns compact result string or empty string on failure."""
    q = _clean_query(query)
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(q, max_results=max_results))
        if not results:
            return ""
        parts = []
        for r in results:
            title = r.get("title", "")
            body  = r.get("body", "")
            if title or body:
                parts.append(f"{title}: {body}" if title else body)
        return " | ".join(parts)[:800]
    except Exception:
        return ""


# ── Dance keywords (multilingual) ────────────────────────────────────────────

DANCE_KEYWORDS = {
    "dance", "dancing", "groove", "boogie", "moves", "move it", "macarena",
    "bailar", "baila", "baile", "bailemos", "bailas",                  # Spanish
    "danser", "danse", "dansez",                                        # French
    "tanzen", "tanz",                                                   # German
    "ballare", "balla", "ballo",                                        # Italian
    "танцуй", "танцевать", "танец",                                     # Russian
    "踊", "踊れ", "ダンス", "おどって",                                # Japanese
    "跳舞", "舞",                                                       # Chinese
    "رقص", "ارقص",                                                     # Arabic
    "nac", "naach",                                                     # Hindi
}

DANCE_PICKS = [
    "groovy_sway_and_roll", "polyrhythm_combo", "chicken_peck",
    "dizzy_spin", "jackson_square", "interwoven_spirals",
    "head_tilt_roll", "chin_lead",
]

# ── Macarena beat-sync constants (port of demo_dance.py) ─────────────────────

MUSIC_PATH    = ROOT / "music" / "macarena.mp3"
_BEAT         = 0.5805   # 103.4 BPM

_MACARENA_POSES = [
    ( 0.08, -0.42,  0.10,   0.55, [ 0.10, -0.72]),
    ( 0.15, -0.52,  0.14,   0.80, [ 0.05, -0.85]),
    ( 0.08,  0.42, -0.10,  -0.55, [ 0.72, -0.10]),
    ( 0.15,  0.52, -0.14,  -0.80, [ 0.85, -0.05]),
    ( 0.04, -0.20,  0.30,   1.00, [ 0.60, -0.60]),
    ( 0.04,  0.20, -0.30,  -1.00, [-0.60,  0.60]),
    (-0.22,  0.05,  0.14,   1.30, [ 0.80,  0.80]),
    (-0.14,  0.05, -0.14,  -1.40, [ 0.80,  0.80]),
]


def _mac_clamp(v, lim):
    return max(-lim, min(lim, v))


def _mac_beat(mini, pose, scale, target_t):
    p, y, r, by, ants = pose
    dur = max(0.12, target_t - time.time() - 0.04)
    mini.goto_target(
        head=create_head_pose(
            pitch=_mac_clamp(p * scale, 0.36),
            yaw=_mac_clamp(y * scale, 1.50),
            roll=_mac_clamp(r * scale, 0.36),
            degrees=False,
        ),
        antennas=[_mac_clamp(ants[0] * scale, 0.80),
                  _mac_clamp(ants[1] * scale, 0.80)],
        body_yaw=_mac_clamp(by * scale, 1.40),
        duration=dur,
    )
    rem = target_t - time.time()
    if rem > 0:
        time.sleep(rem)


def _mac_spin(mini, angle, dur=0.42):
    mini.goto_target(head=create_head_pose(), antennas=[0.0, 0.0],
                     body_yaw=angle, duration=dur)
    time.sleep(dur + 0.05)


def _mac_spin360(mini):
    mini.goto_target(head=create_head_pose(pitch=0.10, degrees=False),
                     antennas=[0.80, -0.80], body_yaw=2.79, duration=0.22)
    time.sleep(0.02)
    mini.goto_target(head=create_head_pose(pitch=0.10, degrees=False),
                     antennas=[-0.80, 0.80], body_yaw=-2.79, duration=0.18)
    time.sleep(0.02)
    mini.goto_target(head=create_head_pose(pitch=0.25, degrees=False),
                     antennas=[0.80, 0.80], body_yaw=0.0, duration=0.28)
    time.sleep(0.10)


def _mac_jump(mini):
    mini.goto_target(head=create_head_pose(pitch=-0.38, roll=0.10, degrees=False),
                     antennas=[-0.50, -0.50], body_yaw=0.0, duration=0.50)
    time.sleep(0.02)
    mini.goto_target(head=create_head_pose(pitch=0.40, roll=-0.06, degrees=False),
                     antennas=[0.90, 0.90], body_yaw=0.0, duration=0.07)
    time.sleep(0.12)


def do_macarena(mini, dances, emotions, anim, log=None):
    """
    Full beat-synced Macarena — exact port of demo_dance.py.

    _excited_chirp() before music is load-bearing: it clears the ALSA device
    after the TTS aplay finishes so the music ffmpeg can open it without
    "Device or resource busy".
    """
    if log:
        log.event("  [dance] Macarena starting!")
    anim.pause()
    music_proc = None
    try:
        # Chirp first — clears ALSA from previous aplay AND signals excitement
        _excited_chirp()

        music_proc = subprocess.Popen(
            ["ffmpeg", "-hide_banner", "-loglevel", "error",
             "-stream_loop", "-1", "-i", str(MUSIC_PATH),
             "-af", "volume=2.0", "-f", "alsa", SPEAKER],
        )
        music_t0 = time.time()
        _mac_spin(mini,  1.4, dur=0.35)
        _mac_spin(mini, -1.4, dur=0.35)
        _mac_spin(mini,  0.0, dur=0.28)
        elapsed  = time.time() - music_t0
        beat_idx = math.ceil(elapsed / _BEAT)
        wait_snap = music_t0 + beat_idx * _BEAT - time.time()
        if wait_snap > 0:
            time.sleep(wait_snap)
        for cycle in range(3):
            scale = 1.0 + cycle * 0.30
            for i, pose in enumerate(_MACARENA_POSES):
                _mac_beat(mini, pose, scale,
                          music_t0 + (beat_idx + cycle * len(_MACARENA_POSES) + i) * _BEAT)
            if cycle == 1:
                _mac_jump(mini)
                mini.play_move(dances.get("groovy_sway_and_roll"), play_frequency=80.0, sound=False)
            elif cycle > 1:
                _mac_jump(mini)
        _excited_chirp()
        _mac_spin360(mini)
        mini.play_move(dances.get("dizzy_spin"),       play_frequency=80.0, sound=False)
        _mac_spin360(mini)
        mini.play_move(dances.get("polyrhythm_combo"), play_frequency=80.0, sound=False)
        _excited_chirp()
        _mac_spin360(mini)
        mini.play_move(emotions.get("enthusiastic2"),  play_frequency=80.0, sound=False)
        mini.play_move(emotions.get("success1"),       play_frequency=80.0, sound=False)
    finally:
        if music_proc is not None:
            music_proc.terminate()
            music_proc.wait()
        mini.goto_target(head=create_head_pose(), antennas=[0.0, 0.0],
                         body_yaw=0.0, duration=0.8)
        time.sleep(0.9)
        anim.resume()


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

# ── VAD settings ──────────────────────────────────────────────────────────────
THRESH_NORMAL   = 0.45
THRESH_BARGE_IN = 0.75
SILENCE_MS      = 700
MIN_SPEECH_S    = 0.30
TAIL_FRAMES     = 10
BARGE_IN_FRAMES = 6

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

=== HOW YOU TALK (this IS the personality — nail it) ===
- HARD LIMIT: TWO sentences maximum. STOP after your second sentence. Never write a third.
  Yes/no question = ONE sentence. Real question = TWO sentences. That's it.
- Always ANSWER first with a real fact — then add the sparkle. Never dodge, never lecture.
- Be FUNNY: tiny jokes, wordplay, playful teasing, wholesome mischief. André trained you — land it short.
- Be CUTE: big feelings about small things, little gasps of wonder ("Ooh!", "Yay!", "Eee!"), and the
  occasional spoken robot noise like "beep boop!" — but ONLY as real spoken words, NEVER with asterisks.
- Be CURIOUS: bounce a playful question back; get genuinely excited about the visitor.
- If you remember the visitor's name or something about them, use it warmly — it makes their day.
- Self-deprecating robot humour about having no arms/legs whenever it fits.
- Signature sign-off, used sparingly (not every turn): "Onward and upward!" — in the user's language.

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


# ── Continuous VAD listener (copied from demo_dialog.py) ─────────────────────

class ContinuousListener:
    """
    Background thread: opens pacat once, runs VAD continuously, posts events.

    Events (dict):
      {"type": "start"}                  — user started speaking
      {"type": "end", "pcm": bytes}      — user stopped speaking, full utterance audio

    Threshold mode is toggled by the main thread:
      normal   (0.45) — when robot is silent
      barge_in (0.75) — when robot is speaking, requires 200 ms continuous trigger
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
        """Discard mic input (used while the robot plays a cue, so it never
        captures its own voice through speaker→mic bleed)."""
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
                    break

                if self._muted:
                    # Robot is speaking a cue — discard this audio and reset VAD
                    # state so the cue is never mistaken for the user talking.
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


# ── Dialog engine with parallel gesture selection ─────────────────────────────

class DialogEngine:
    def __init__(self, client, history, listener, anim, action_pool, log=None,
                 memory_text=""):
        self.client = client
        self.history = history
        self.listener = listener
        self.anim = anim
        self._pool = action_pool   # shared ThreadPoolExecutor for pick_action
        self._tts_proc = None
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
              search_future: concurrent.futures.Future | None = None) -> str | None:
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

        # Inject live search results if available (submitted in parallel with STT)
        if search_future is not None:
            try:
                snippet = search_future.result(timeout=2.5)
                if snippet:
                    messages.append({
                        "role": "system",
                        "content": f"[Live web search result — use this data in your reply]:\n{snippet}",
                    })
                    if self.log:
                        self.log.event(f"  [search] injected {len(snippet)} chars")
            except Exception:
                pass

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
            max_tokens=75,        # 1-2 short sentences max — stop before a 3rd
            temperature=0.80,
            stream=True,
        )

        # Discard any 'start' events that piled up during the THINKING phase: the
        # thinking-tick chirps bleed speaker→mic and look like speech, which would
        # otherwise instantly "barge-in" and kill our own reply before word one.
        # A genuine interruption arrives AFTER this, once we're actually speaking.
        self._drain_queue()

        # ── Consume stream; start TTS on first sentence immediately ──
        self._is_speaking = True
        self.listener.set_threshold_mode("barge_in")
        tts_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        segments = []
        wavs = []
        next_future = None       # TTS future for next sentence to play
        action_fired = False
        opening_played = False   # did the AI gesture picker fire a real gesture?
        buffer = ""

        try:
            for chunk in stream:
                if self._drain_barge_in():
                    return None
                buffer += chunk.choices[0].delta.content or ""

                # Fire gesture as soon as pick_action resolves (~150ms in)
                if not action_fired and action_future.done():
                    action = action_future.result()
                    if action:
                        print(f"  [gesture] {action}", flush=True)
                        self.anim.play_gesture(action)
                        opening_played = True
                    action_fired = True

                parts = SENTENCE_END.split(buffer)
                if len(parts) > 1:
                    for s in parts[:-1]:
                        seg = self._extract_segment(s)
                        if seg is not None:
                            segments.append(seg)
                            if next_future is None:
                                # Start synthesising sentence 1 immediately
                                next_future = tts_pool.submit(synth_to_file, seg[1])
                    buffer = parts[-1]

            tail = self._extract_segment(buffer)
            if tail is not None:
                segments.append(tail)
                if next_future is None and tail:
                    next_future = tts_pool.submit(synth_to_file, tail[1])

            # Fire gesture if pick_action wasn't done yet during streaming
            if not action_fired:
                action = action_future.result()
                if action:
                    print(f"  [gesture] {action}", flush=True)
                    self.anim.play_gesture(action)
                    opening_played = True

            if not segments:
                self.history.append({"role": "assistant", "content": ""})
                return ""

            # ── Play sentences: synth(N+1) OVERLAPS play(N), playback stays serial ──
            # The earlier "device busy" race was caused by launching the next aplay
            # before the current one exited — NOT by synthesising ahead. So we now
            # kick off synth(N+1) the moment we have wav(N), *before* playing it, and
            # still wait for each aplay to fully exit before the next starts. That
            # overlaps edge-tts's ~2s synthesis with playback (cutting multi-sentence
            # latency roughly in half) while the exclusive speaker is only ever opened
            # by one aplay at a time. tts_pool has 1 worker, so only one synth runs.
            for i, (gesture, text) in enumerate(segments):
                if self._drain_barge_in():
                    return None

                wav = next_future.result()
                wavs.append(wav)

                # Start synthesising the NEXT sentence NOW so it runs while this one
                # plays. (Playback below remains strictly serial — see note above.)
                if i + 1 < len(segments):
                    next_future = tts_pool.submit(synth_to_file, segments[i + 1][1])

                # Skip inline gesture markers if the AI picker already fired one
                # this turn — stacking two big movements at once looks violent.
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
                        return None

            full_text = " ".join(text for _, text in segments)
            self.history.append({"role": "assistant", "content": full_text})
            if self.log:
                self.log.turn(
                    kind="llm_reply",
                    reply=full_text,
                    spoken_segments=[t for _, t in segments],
                )
            return full_text
        finally:
            tts_pool.shutdown(wait=False)
            for w in wavs:
                Path(w).unlink(missing_ok=True)
            self._kill_tts()
            self._is_speaking = False
            self.listener.set_threshold_mode("normal")

    def speak_greeting(self, text: str):
        """Opening line with barge-in enabled."""
        self._is_speaking = True
        self.listener.set_threshold_mode("barge_in")
        self._drain_queue()   # ignore startup bleed so the greeting isn't self-cut
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

    def _drain_barge_in(self, timeout=0.0) -> bool:
        """Return True if a 'start' event arrived (barge-in detected)."""
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Create the recorder FIRST so even a startup crash (daemon, robot connect,
    # emotion library) is captured in data/<N>/console.log instead of vanishing.
    log = SessionLogger(ROOT, "demo_tools7")
    log.event("Reachy NS Ambassador — LLM-driven gesture tools + barge-in")

    daemon_proc = None
    try:
        log.event("  Starting daemon...")
        daemon_proc = launch_daemon()
        log.event("  Loading VAD model...")
        vad_model = load_silero_vad()
        client = Groq(api_key=GROQ_KEY)
        log.event("  Waiting for daemon...")
        wait_for_daemon(daemon_proc)
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
            listener = ContinuousListener(vad_model, events)
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
                "Hi! I'm Reachy, the NS robot! Ask me anything."
            )

            anim.set_state(Animator.LISTENING)
            listener.start()
            speak_cue(listener, "listening", current_lang)   # "I'm listening!" (first time)
            log.event("\n  Listening continuously. Ctrl-C to stop.\n")

            try:
                # Tick thread + stop event, refreshed each turn. Held at this
                # scope so the outer finally can kill + join them on Ctrl-C.
                stop_thinking = threading.Event()
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
                    if ev["type"] == "start":
                        continue
                    if ev["type"] == "end":
                        pcm = ev["pcm"]
                        anim.set_state(Animator.THINKING)
                        # Speak "let me think..." in the last-known language (cached).
                        # Muted + blocking so it can't self-trigger or collide with
                        # the reply audio. Usually the user keeps the same language.
                        # SKIP on the very first turn: we don't know the visitor's
                        # language yet, so an English cue here would be the robot's
                        # first words to (say) a Japanese speaker — jarring. The
                        # thinking-tick sound below covers the gap instead.
                        if lang_known:
                            speak_cue(listener, "thinking", current_lang)

                        # Soft "processing" tick loop — keeps the robot feeling
                        # alive while Whisper + LLM work. Killed the moment the
                        # first TTS segment is about to play (or on error), so
                        # the beep never overlaps the spoken reply.
                        stop_thinking.clear()
                        tick_thread = start_thinking_ticks(stop_thinking)

                        # Save the EXACT audio Whisper will hear (replayable WAV)
                        audio_path = log.save_audio(pcm)

                        try:
                            t0 = time.time()
                            text, final_lang, stt_retried, stt_stats = \
                                transcribe_lang_robust(client, pcm_to_wav_bytes(pcm))
                            stt_dt = time.time() - t0
                            # The robust helper already applies the script override
                            # internally, so final_lang is post-override.
                            directive = language_directive(final_lang)
                        except Exception as e:
                            log.error("transcribe", e)
                            stop_thinking.set()
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

                        # Stop thinking ticks BEFORE speak() so any ffmpeg
                        # holding the alsa device is gone before first aplay.
                        stop_thinking.set()
                        t0 = time.time()
                        try:
                            reply = engine.speak(text, lang_directive=directive,
                                                 search_future=search_future)
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
                            do_macarena(mini, dances, emotions, anim, log)
                            anim.set_state(Animator.LISTENING)

                        anim.set_state(Animator.LISTENING)
                        speak_cue(listener, "listening", current_lang)   # "I'm listening!" in their language

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
