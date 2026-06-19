"""
reachy_demo/groq_client.py — Groq API helpers (STT + LLM).

Provides:
  - load_api_key() → str | None
  - transcribe(client, wav_bytes, language=None) → str
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
    Pass language="en" to force English-only decoding (faster, more accurate for English).
    """
    kwargs = dict(
        file=("audio.wav", wav_bytes, "audio/wav"),
        model="whisper-large-v3-turbo",
        response_format="text",
    )
    if language is not None:
        kwargs["language"] = language
    transcription = client.audio.transcriptions.create(**kwargs)
    return transcription.strip()

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
    """
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}] + messages,
        max_tokens=70,
        temperature=0.90,
        stream=True,
    )
    for chunk in stream:
        yield chunk.choices[0].delta.content or ""
