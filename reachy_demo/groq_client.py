"""
reachy_demo/groq_client.py — Groq API helpers (STT + LLM).

Provides:
  - load_api_key() → str | None
  - transcribe(client, wav_bytes, language=None) → str
  - transcribe_lang(client, wav_bytes) → (text, language)  — detects spoken language
  - language_directive(lang) → str | None  — strong "reply in this language" instruction
  - stream_chat(client, messages, model, system) → iterator of text chunks
"""

import os
from pathlib import Path
from typing import Iterator

from groq import Groq

# ── API key ───────────────────────────────────────────────────────────────────

def load_api_key(root: Path | None = None) -> str | None:
    """
    Read GROQ_API_KEY from <root>/.env (supports both ':' and '=' separators)
    or fall back to the GROQ_API_KEY environment variable.
    If root is None, defaults to the repository root (two levels above this file).
    """
    if root is None:
        root = Path(__file__).parent.parent
    env = root / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "GROQ_API_KEY" in line:
                return line.replace("GROQ_API_KEY:", "").replace("GROQ_API_KEY=", "").strip()
    return os.environ.get("GROQ_API_KEY")

# ── STT ───────────────────────────────────────────────────────────────────────

def transcribe(client: Groq, wav_bytes: bytes, language: str | None = None) -> str:
    """
    Transcribe WAV audio bytes via Groq Whisper.

    Uses whisper-large-v3 (the full model, NOT -turbo) for best multilingual
    accuracy. Turbo is faster but noticeably worse at non-English speech, which
    matters because Reachy must understand any language a visitor speaks.

    Pass language="en" to force a single language (faster, slightly more accurate
    when you know the language up front). Leave as None for auto-detection — this
    is what lets the robot hear Spanish, French, Japanese, Arabic, etc.
    """
    kwargs = dict(
        file=("audio.wav", wav_bytes, "audio/wav"),
        model="whisper-large-v3",
        response_format="text",
    )
    if language is not None:
        kwargs["language"] = language
    transcription = client.audio.transcriptions.create(**kwargs)
    return transcription.strip()


def transcribe_lang_verbose(client: Groq, wav_bytes: bytes) -> tuple[str, str, dict]:
    """
    Transcribe AND detect language AND return per-segment confidence stats.

    Returns (text, language, stats) where `stats` aggregates Whisper's
    segment-level quality signals used by `is_hallucination()` to reject
    phantom transcripts (see that function). `stats` is {} if the model
    returned no segment metadata.
    """
    resp = client.audio.transcriptions.create(
        file=("audio.wav", wav_bytes, "audio/wav"),
        model="whisper-large-v3",
        response_format="verbose_json",
    )
    text = (getattr(resp, "text", "") or "").strip()
    lang = (getattr(resp, "language", "") or "").strip()

    segments = getattr(resp, "segments", None) or []
    stats = {}
    if segments:
        def _vals(key):
            out = []
            for s in segments:
                v = s.get(key) if isinstance(s, dict) else getattr(s, key, None)
                if v is not None:
                    out.append(v)
            return out
        logprobs = _vals("avg_logprob")
        nospeech = _vals("no_speech_prob")
        compress = _vals("compression_ratio")
        stats = {
            "n_segments":         len(segments),
            "mean_avg_logprob":   (sum(logprobs) / len(logprobs)) if logprobs else None,
            "max_no_speech_prob": max(nospeech) if nospeech else None,
            "max_compression":    max(compress) if compress else None,
        }
    return text, lang, stats


def transcribe_lang(client: Groq, wav_bytes: bytes) -> tuple[str, str]:
    """
    Transcribe AND detect the spoken language in one call (verbose_json).

    Returns (text, language) where `language` is Whisper's detected language —
    usually a full English name like "english", "chinese", "spanish", or a code
    like "en". Empty string if detection failed.

    This is the reliable way to make Reachy reply in the visitor's language:
    instead of hoping the LLM infers the language from the text, we KNOW what was
    spoken and command the LLM explicitly (see language_directive).
    """
    text, lang, _ = transcribe_lang_verbose(client, wav_bytes)
    return text, lang


def is_hallucination(text: str, stats: dict) -> bool:
    """
    Decide whether a Whisper transcript is a phantom (hallucinated on silence,
    breath, or speaker→mic bleed) rather than real speech.

    whisper-large-v3 confidently invents text on near-silent audio — classically
    "Thank you." / "you" in English or "ご視聴ありがとうございました" in Japanese.
    These phantom turns are the #1 multilingual failure: they fire a fake turn in
    the wrong language. We gate on Whisper's own segment-level quality signals:

      - max_no_speech_prob — high (>0.6) means the model itself thinks the clip is
        mostly silence; combined with a weak avg_logprob that's a phantom.
      - mean_avg_logprob — very low (< -1.0) means the model was guessing.
      - max_compression — > 2.4 means looped/repeated text (a repetition spiral).

    Conservative by design: when stats are missing we DON'T reject (return False),
    so a real utterance is never dropped just because metadata was absent.
    """
    if not text:
        return True
    if not stats:
        return False
    logp  = stats.get("mean_avg_logprob")
    nosp  = stats.get("max_no_speech_prob")
    comp  = stats.get("max_compression")
    if logp is not None and logp < -1.0:
        return True
    if nosp is not None and logp is not None and nosp > 0.6 and logp < -0.4:
        return True
    if comp is not None and comp > 2.4:
        return True
    return False


# Whisper sometimes mislabels East-Asian audio as a random language and
# transliterates the text into romaji ("Konnichiwa" instead of "こんにちは"),
# which defeats our script-based language override. The misdetections cluster
# in a small set — empirically: Indonesian, Malay, Vietnamese, Filipino,
# sometimes Turkish. If we land in that set and the transcript has no
# non-Latin script, retry once with a forced-language transcription. We pick
# Japanese as the forced language because it's both the most commonly
# misdetected AND because for non-Japanese audio the forced transcribe still
# returns usable text in the actual language.
_MISDECT_LANGS = {"indonesian", "malay", "vietnamese", "filipino", "tagalog", "turkish"}


def transcribe_lang_robust(client: Groq, wav_bytes: bytes) -> tuple[str, str, bool, dict]:
    """
    Same as transcribe_lang(), but retries once with forced language when the
    first pass looks like a romaji transliteration of an East-Asian language.

    Returns (text, lang, retried, stats) where `retried` is True if we did a
    second call, `lang` is always the *final* language to use (post-override),
    and `stats` is the first pass's confidence metadata for `is_hallucination()`.
    """
    text, lang, stats = transcribe_lang_verbose(client, wav_bytes)
    # Fast path: non-Latin script in the transcript, OR a non-ambiguous lang
    if script_language(text) is not None or lang.lower() not in _MISDECT_LANGS:
        return text, resolve_language(text, lang), False, stats
    # Misdetection suspected: re-transcribe with a forced language. We try
    # Japanese first because it's both the most common misdetection target
    # AND forces Whisper to output kana if the audio really was Japanese
    # (which the script override then catches correctly).
    try:
        text2 = transcribe(client, wav_bytes, language="ja").strip()
    except Exception:
        return text, resolve_language(text, lang), False, stats
    if script_language(text2) is not None:
        return text2, resolve_language(text2, "japanese"), True, stats
    # Forced pass didn't reveal any kana either — keep the original (which
    # was already in Latin script and may genuinely have been the detected lang).
    return text, resolve_language(text, lang), True, stats


def script_language(text: str) -> str | None:
    """
    Detect language from the SCRIPT of the transcribed text (Unicode ranges).

    This is far more reliable than Whisper's audio-based language label for
    non-Latin scripts: if the transcript contains Japanese kana it IS Japanese,
    no matter what Whisper guessed. This is the fix for "asked in Japanese,
    replied in Spanish" — Whisper mislabels the audio, but the kana in the text
    gives it away. Returns None for Latin-script text (can't tell from script).
    """
    if not text:
        return None
    has_kana    = any('぀' <= c <= 'ヿ' for c in text)   # hiragana + katakana
    has_hangul  = any('가' <= c <= '힣' or 'ᄀ' <= c <= 'ᇿ' for c in text)
    has_cjk     = any('一' <= c <= '鿿' for c in text)   # CJK ideographs
    has_arabic  = any('؀' <= c <= 'ۿ' for c in text)
    has_cyrillic= any('Ѐ' <= c <= 'ӿ' for c in text)
    has_thai    = any('฀' <= c <= '๿' for c in text)
    has_hebrew  = any('֐' <= c <= '׿' for c in text)
    has_greek   = any('Ͱ' <= c <= 'Ͽ' for c in text)
    has_hindi   = any('ऀ' <= c <= 'ॿ' for c in text)   # Devanagari
    if has_kana:     return "Japanese"   # kana is unique to Japanese
    if has_hangul:   return "Korean"
    if has_cjk:      return "Chinese"     # CJK ideographs without kana → Chinese
    if has_arabic:   return "Arabic"
    if has_cyrillic: return "Russian"
    if has_thai:     return "Thai"
    if has_hebrew:   return "Hebrew"
    if has_greek:    return "Greek"
    if has_hindi:    return "Hindi"
    return None


def resolve_language(text: str, whisper_lang: str) -> str:
    """
    Pick the best language signal: script detection wins for non-Latin text
    (reliable), otherwise fall back to Whisper's audio label. Used to build the
    reply-language directive.
    """
    return script_language(text) or whisper_lang


# Whisper sometimes returns ISO codes instead of names — map the common ones.
_LANG_NAMES = {
    "en": "English", "zh": "Chinese", "es": "Spanish", "fr": "French",
    "de": "German", "it": "Italian", "pt": "Portuguese", "ja": "Japanese",
    "ko": "Korean", "ar": "Arabic", "hi": "Hindi", "ru": "Russian",
    "nl": "Dutch", "pl": "Polish", "tr": "Turkish", "sv": "Swedish",
    "id": "Indonesian", "vi": "Vietnamese", "th": "Thai", "uk": "Ukrainian",
    "he": "Hebrew", "el": "Greek", "cs": "Czech", "ro": "Romanian",
    "ms": "Malay", "fil": "Filipino", "tl": "Filipino",
}


def language_directive(lang: str) -> str | None:
    """
    Build a strong, recency-biased instruction telling the LLM to reply in the
    language Whisper detected. Returns None if the language is unknown (let the
    LLM decide). Inject the returned string as a system message placed AFTER the
    conversation history so it dominates the model's choice of output language.
    """
    if not lang:
        return None
    key = lang.strip().lower()
    name = _LANG_NAMES.get(key, lang.strip().title())
    return (
        f"LANGUAGE OVERRIDE — HARD RULE — THIS OVERRIDES EVERYTHING ABOVE:\n"
        f"The user's current message is in {name}.\n"
        f"You MUST reply 100% in {name}. Not one word of any other language.\n"
        f"Even if your previous replies in this conversation were in a different "
        f"language, SWITCH NOW. {name} only. Starting immediately.\n"
        f"Brand/people names stay in Latin spelling: "
        f"Network School, NS, Virtuals, Quantus, Ârc, Balaji, Pixel."
    )

# ── LLM streaming ─────────────────────────────────────────────────────────────

def stream_chat(
    client: Groq,
    messages: list,
    model: str,
    system: str,
) -> Iterator[str]:
    """
    Stream LLM chat completions. Yields text delta strings as they arrive.
    `messages` should be the conversation history (user/assistant turns only).
    `system` is prepended as the system message.
    NOTE: max_tokens is hardcoded to 70 and cannot be overridden by callers.
    """
    # with_options(timeout=...) bounds connect/read stalls so a dead connection
    # can't freeze the robot mid-reply; per-chunk reads of the stream inherit it.
    stream = client.with_options(timeout=20.0).chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}] + messages,
        max_tokens=70,
        temperature=0.90,
        stream=True,
    )
    for chunk in stream:
        # A terminal usage/keep-alive chunk can carry an empty `choices` list;
        # indexing [0] on it would raise IndexError and kill the reply mid-stream.
        if not chunk.choices:
            continue
        yield chunk.choices[0].delta.content or ""
