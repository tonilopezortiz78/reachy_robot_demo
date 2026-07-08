"""
reachy_demo/cues.py — short spoken turn-taking cues in the user's language.

Reachy says a tiny phrase so the visitor always knows whose turn it is:
  - "I'm listening!"  → your turn to talk
  - "Let me think..." → I'm processing, hold on

The phrase is spoken in whatever language the conversation is currently in
(Japanese → Japanese, Spanish → Spanish, …). Each (language, kind) WAV is
synthesised once via edge-tts (the cute +32Hz voice) and cached in
cache/cue_<lang>_<kind>.wav, so after the first use it replays instantly with
zero synthesis latency.

Languages not in the table fall back to English.
"""
import hashlib
import json
import random
import shutil
import subprocess
import threading
from pathlib import Path

from reachy_demo.audio import SPEAKER
from reachy_demo.tts_edge import synth_to_file

ROOT = Path(__file__).parent.parent
CACHE = ROOT / "cache"

# (listening phrase, thinking phrase, repeat phrase) keyed by the language NAME
# that resolve_language() / Whisper returns ("English", "Japanese", "Spanish", …).
# The "repeat" phrase is spoken when Reachy couldn't understand the utterance
# (empty transcript or a rejected hallucination) — a polite "say it again?".
CUE_PHRASES = {
    "English":    ("I'm listening!",       "Let me think...",        "Sorry, I didn't catch that — could you repeat?"),
    "Spanish":    ("¡Te escucho!",         "Déjame pensar...",       "Perdona, no te entendí. ¿Puedes repetir?"),
    "French":     ("Je t'écoute !",        "Laisse-moi réfléchir...", "Désolé, je n'ai pas compris. Tu peux répéter ?"),
    "German":     ("Ich höre zu!",         "Lass mich überlegen...", "Sorry, das habe ich nicht verstanden. Kannst du es wiederholen?"),
    "Portuguese": ("Estou ouvindo!",       "Deixa eu pensar...",     "Desculpa, não entendi. Pode repetir?"),
    "Italian":    ("Ti ascolto!",          "Fammi pensare...",       "Scusa, non ho capito. Puoi ripetere?"),
    "Japanese":   ("聞いてるよ！",          "ちょっと考えるね…",        "ごめん、聞き取れなかった。もう一回言ってくれる？"),
    "Chinese":    ("我在听！",              "让我想想…",               "抱歉，我没听清，可以再说一遍吗？"),
    "Korean":     ("듣고 있어요!",          "잠깐 생각해 볼게요…",       "미안해요, 잘 못 들었어요. 다시 말해 줄래요?"),
    "Arabic":     ("أنا أستمع!",           "دعني أفكّر…",             "آسف، لم أفهم. هل يمكنك التكرار؟"),
    "Russian":    ("Я слушаю!",            "Дай подумать…",          "Извини, я не расслышал. Повтори, пожалуйста?"),
    "Hindi":      ("मैं सुन रहा हूँ!",       "मुझे सोचने दो…",          "माफ़ करना, मैं समझ नहीं पाया। फिर से कहोगे?"),
}

_LISTENING, _THINKING, _REPEAT = 0, 1, 2
_KIND_INDEX = {"listening": _LISTENING, "thinking": _THINKING, "repeat": _REPEAT}

# Cue WAVs are ~0.5-2 s; if aplay blocks far longer the exclusive speaker device
# is wedged. Bound the wait so a stuck aplay can never leave the listener muted
# forever (which would make the robot permanently deaf).
_PLAY_TIMEOUT_S = 10.0


def _wait_or_kill(proc: subprocess.Popen, timeout: float = _PLAY_TIMEOUT_S):
    """Wait for an aplay proc, killing it if the speaker device is wedged."""
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()

# Varied, natural "thinking out loud" fillers spoken the moment the user stops,
# so the robot acknowledges immediately ("Hmm, let me think...") instead of going
# silent. A random one is picked per turn so it never sounds robotic. Languages
# without a list fall back to the single CUE_PHRASES "thinking" phrase.
THINKING_VARIANTS = {
    "English":    ["Right...", "Okay...", "Got it, one sec...",
                   "Hmm, let me think...", "Let me see...", "One moment...",
                   "Umm, let me check...", "Good one, let me think..."],
    "Spanish":    ["Vale...", "Claro...", "Ya veo...",
                   "Mmm, déjame pensar...", "A ver...", "Un momento...",
                   "Déjame ver..."],
    "French":     ["D'accord...", "OK...", "Je vois...",
                   "Hmm, laisse-moi réfléchir...", "Voyons voir...",
                   "Un instant...", "Attends, je réfléchis..."],
    "German":     ["Okay...", "Klar...", "Verstanden...",
                   "Hmm, lass mich überlegen...", "Mal sehen...",
                   "Einen Moment...", "Warte, ich denke nach..."],
    "Portuguese": ["Certo...", "Tá bom...", "Entendi...",
                   "Hmm, deixa eu pensar...", "Deixa ver...", "Um momento..."],
    "Italian":    ["Va bene...", "Capito...", "Ok...",
                   "Mmm, fammi pensare...", "Vediamo...", "Un attimo..."],
    "Japanese":   ["なるほど…", "うん…", "了解…",
                   "うーん、ちょっと考えるね…", "ええと…", "ちょっと待ってね…"],
    "Chinese":    ["好的…", "嗯…", "明白了…",
                   "嗯，让我想想…", "我看看…", "稍等一下…"],
    "Korean":     ["네…", "알겠어요…", "그렇군요…",
                   "음, 생각해 볼게요…", "어디 보자…", "잠시만요…"],
    "Arabic":     ["حسناً…", "تمام…", "فهمت…",
                   "مم، دعني أفكّر…", "لنرَ…", "لحظة…"],
    "Russian":    ["Ясно…", "Понял…", "Хорошо…",
                   "Хм, дай подумать…", "Посмотрим…", "Секундочку…"],
    "Hindi":      ["ठीक है…", "अच्छा…", "समझ गया…",
                   "हम्म, सोचने दो…", "देखता हूँ…", "एक पल…"],
}

# ── On-the-fly translation for languages not in CUE_PHRASES ───────────────────
# A visitor speaking Thai, Vietnamese, Polish, Turkish, … would otherwise hear
# the cue in English, snapping the immersion. If a Groq client is registered via
# set_translator(), we translate the two short phrases ONCE per new language and
# persist them in cache/cue_translations.json so it's a one-time cost ever.

_TRANSLATOR = None          # Groq client, set by the demo at startup
_TRANSLATE_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
_XLATE_FILE = CACHE / "cue_translations.json"
_xlate_lock = threading.Lock()


def set_translator(client, model: str | None = None):
    """Register a Groq client so cues for unlisted languages get translated
    (and cached) instead of falling back to English."""
    global _TRANSLATOR, _TRANSLATE_MODEL
    _TRANSLATOR = client
    if model:
        _TRANSLATE_MODEL = model


def _load_xlate() -> dict:
    try:
        return json.loads(_XLATE_FILE.read_text())
    except Exception:
        return {}


def _save_xlate(data: dict):
    try:
        CACHE.mkdir(parents=True, exist_ok=True)
        _XLATE_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    except Exception:
        pass


def _translate(english: str, lang: str) -> str | None:
    """Translate one short cue phrase into `lang` via Groq, cached on disk.
    Returns None on any failure (caller falls back to English)."""
    if _TRANSLATOR is None:
        return None
    key = f"{lang}|{english}"
    with _xlate_lock:
        cache = _load_xlate()
        if key in cache:
            return cache[key]
    try:
        resp = _TRANSLATOR.chat.completions.create(
            model=_TRANSLATE_MODEL,
            messages=[
                {"role": "system", "content":
                    "Translate the short, friendly phrase into the named language. "
                    "Keep it casual and warm, like a cute robot. Reply with ONLY the "
                    "translation — no quotes, no notes, no transliteration."},
                {"role": "user", "content": f"Language: {lang}\nPhrase: {english}"},
            ],
            max_tokens=30, temperature=0.3, stream=False,
        )
        out = (resp.choices[0].message.content or "").strip().strip('"').strip()
    except Exception:
        return None
    if not out:
        return None
    with _xlate_lock:
        cache = _load_xlate()
        cache[key] = out
        _save_xlate(cache)
    return out


def _phrase(kind: str, lang: str) -> str:
    idx = _KIND_INDEX.get(kind, _LISTENING)
    if lang in CUE_PHRASES:
        return CUE_PHRASES[lang][idx]
    english = CUE_PHRASES["English"][idx]
    return _translate(english, lang) or english


def cue_wav(kind: str, lang: str) -> str | None:
    """Return a cached WAV path for (kind, lang), synthesising it once if needed."""
    safe = "".join(c if c.isalnum() else "_" for c in lang)
    path = CACHE / f"cue_{safe}_{kind}.wav"
    if not path.exists():
        try:
            CACHE.mkdir(parents=True, exist_ok=True)
            tmp = synth_to_file(_phrase(kind, lang))
            # shutil.move, not Path.rename: /tmp is tmpfs, so a rename into
            # cache/ crosses filesystems and raises EXDEV.
            shutil.move(str(tmp), str(path))
        except Exception:
            return None
    return str(path)


def play_cue(kind: str, lang: str, block: bool = False):
    """Speak the cue for (kind, lang). Non-blocking by default. Returns the proc."""
    wav = cue_wav(kind, lang)
    if not wav:
        return None
    proc = subprocess.Popen(["aplay", "-D", SPEAKER, "-q", wav],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if block:
        _wait_or_kill(proc)
    return proc


def speak_cue(listener, kind: str, lang: str):
    """
    Play a turn-taking cue SAFELY and block until it finishes.

    Two protections, both required on this hardware:
      1. Mute the listener only during PLAYBACK (not synthesis) — the robot
         speaker and mic are the same USB device, so muting prevents the robot
         hearing its own cue as a user turn. We synthesise the WAV BEFORE muting
         so the listener isn't silenced for 2-3 s on first-use synthesis; it only
         misses the ~0.5-1 s of actual audio playback.
      2. Play to completion — the speaker (plughw:CARD=Audio) is exclusive; if a
         cue overlaps the reply TTS or another cue, one is silently dropped
         ("device busy"). Blocking serialises speaker access.

    `listener` may be None (e.g. before it exists) — then we just play the cue.
    """
    # Synthesise first (possibly 2-3 s on first use) WITHOUT muting
    wav = cue_wav(kind, lang)
    if listener is not None:
        listener.mute()
    try:
        if wav:
            proc = subprocess.Popen(
                ["aplay", "-D", SPEAKER, "-q", wav],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            _wait_or_kill(proc)
    finally:
        if listener is not None:
            listener.unmute()


# ── Verbal thinking fillers ("Hmm, let me think...") ──────────────────────────

def _thinking_phrase(lang: str) -> str:
    """Pick a random natural filler for `lang`, falling back to the single
    CUE_PHRASES thinking line (translated on the fly) for unlisted languages."""
    variants = THINKING_VARIANTS.get(lang)
    if variants:
        return random.choice(variants)
    return _phrase("thinking", lang)


def _wav_for_text(text: str) -> str | None:
    """Cache a synthesised WAV keyed by the text itself (for the random fillers,
    which don't fit the fixed (kind, lang) cue cache). One synth per phrase ever."""
    h = hashlib.md5(text.encode("utf-8")).hexdigest()[:12]
    path = CACHE / f"think_{h}.wav"
    if not path.exists():
        try:
            CACHE.mkdir(parents=True, exist_ok=True)
            tmp = synth_to_file(text)
            # shutil.move, not Path.rename: /tmp is tmpfs (cross-device).
            shutil.move(str(tmp), str(path))
        except Exception:
            return None
    return str(path)


def speak_thinking(listener, lang: str):
    """Speak a random 'thinking out loud' filler in `lang`, blocking until done,
    with the listener muted during playback (same speaker→mic protection as
    speak_cue). Intended to run WHILE STT executes in another thread, so it adds
    no latency — the robot acknowledges instantly and STT finishes underneath."""
    wav = _wav_for_text(_thinking_phrase(lang))
    if listener is not None:
        listener.mute()
    try:
        if wav:
            proc = subprocess.Popen(
                ["aplay", "-D", SPEAKER, "-q", wav],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            _wait_or_kill(proc)
    finally:
        if listener is not None:
            listener.unmute()


def prewarm(lang: str = "English"):
    """Pre-generate cues + thinking fillers for a language in the background so
    first use is instant."""
    def _gen():
        cue_wav("listening", lang)
        cue_wav("thinking", lang)
        cue_wav("repeat", lang)
        for phrase in THINKING_VARIANTS.get(lang, []):
            _wav_for_text(phrase)
    threading.Thread(target=_gen, daemon=True).start()
