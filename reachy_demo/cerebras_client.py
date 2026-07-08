"""
reachy_demo/cerebras_client.py — optional low-latency LLM accelerator.

Cerebras runs the SAME Llama-4-scout model as Groq at ~2,000 tok/s (vs Groq's
~450 tok/s). Same OpenAI-compatible streaming API — we just point the `openai`
SDK at their endpoint. The demo tries Cerebras first; if the API key is
missing or the request fails, it transparently falls back to the Groq path.

Setup:
    Get a free key at  https://cloud.cerebras.ai/
    Add to .env:        CEREBRAS_API_KEY=csk-xxxxx

If the key is absent, this module is a no-op — callers ignore the return and
use the existing Groq client.
"""

import os
from pathlib import Path

try:
    from openai import OpenAI as _OAClient
except ImportError:
    _OAClient = None

CEREBRAS_BASE = "https://api.cerebras.ai/v1"
MODEL = "meta-llama/Llama-4-Scout-17B-16E-Instruct"


def load_cerebras_key(root: Path | None = None) -> str | None:
    if root is None:
        root = Path(__file__).parent.parent
    env = root / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "CEREBRAS_API_KEY" in line:
                return line.replace("CEREBRAS_API_KEY:", "").replace("CEREBRAS_API_KEY=", "").strip()
    return os.environ.get("CEREBRAS_API_KEY")


def make_client(root: Path | None = None):
    """Return an OpenAI-compatible client pointed at Cerebras, or None if no key."""
    if _OAClient is None:
        return None
    key = load_cerebras_key(root)
    if not key:
        return None
    try:
        return _OAClient(base_url=CEREBRAS_BASE, api_key=key)
    except Exception:
        return None


def stream_chat(client, messages, *, model=MODEL, max_tokens=88, temperature=0.80):
    """Drop-in replacement for groq_client.stream_chat.
    Yields text deltas. The caller handles the empty-choice case."""
    stream = client.chat.completions.create(
        model=model, messages=messages,
        max_tokens=max_tokens, temperature=temperature, stream=True,
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        yield chunk.choices[0].delta.content or ""


def has_key(root: Path | None = None) -> bool:
    return load_cerebras_key(root) is not None