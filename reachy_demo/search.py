"""
reachy_demo/search.py — DuckDuckGo web search helper.

Single entry point: web_search(query) → str
  Strips conversational filler before querying so STT output like
  "Can you tell me the price of Bitcoin, please?" becomes "price of Bitcoin",
  which resolves 3-4× faster on DuckDuckGo.

Designed to be submitted to a ThreadPoolExecutor immediately after STT so it
runs in parallel with TTS synthesis and doesn't add wall-clock latency.
"""
import re


_SEARCH_FILLER = re.compile(
    r'\b(can you|could you|please|tell me|do you know|i want to know|'
    r'what is|what are|what was|how much is|how much are|'
    r'give me|show me|let me know)\b',
    re.IGNORECASE,
)


def clean_query(text: str) -> str:
    """Strip conversational filler so DuckDuckGo gets a tighter search term."""
    q = _SEARCH_FILLER.sub("", text)
    q = re.sub(r'\s{2,}', ' ', q).strip(" ,?!.")
    return q or text


def web_search(query: str, max_results: int = 3) -> str:
    """
    Search via DuckDuckGo (no API key needed) and return a compact summary.
    Returns "" on any error — callers treat empty string as "no search result."
    """
    try:
        from ddgs import DDGS
        clean = clean_query(query)
        with DDGS() as ddgs:
            hits = list(ddgs.text(clean, max_results=max_results))
        if not hits:
            return ""
        lines = [f"- {h['title']}: {h['body'][:200]}" for h in hits]
        return "\n".join(lines)
    except Exception:
        return ""
