"""
reachy_demo/text.py — Text utilities for TTS pipelines.

Provides:
  - SENTENCE_END  — compiled regex for sentence boundary splitting
  - clean_for_tts(text) → str  — strip markdown/emote markup before speaking
"""

import re

# Matches whitespace that follows a sentence-ending punctuation mark.
# Use: SENTENCE_END.split(buffer) to split a running buffer into sentences.
SENTENCE_END = re.compile(r'(?<=[.!?])\s+')


def clean_for_tts(text: str) -> str:
    """Strip markdown and roleplay emotes that TTS would read as literal symbols."""
    # Remove roleplay action/emote markers entirely — *beep*, *smile*, *blush*, etc.
    # These are single-word (no spaces) wrapped in 1-3 asterisks. Remove word too.
    text = re.sub(r'\*{1,3}[^*\n]+\*{1,3}', '', text)
    # Strip remaining asterisks from bold/italic (**text** → text, *text* → text)
    text = re.sub(r'\*+([^*]+)\*+', r'\1', text)
    text = re.sub(r'\*+', '', text)
    text = re.sub(r'_+', '', text)                           # __ _
    text = re.sub(r'`+', '', text)                           # ` ``
    text = re.sub(r'#+\s*', '', text)                        # ## headings
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)   # [text](url) → text
    # Strip any remaining bracketed tags so they are never spoken. The LLM
    # sometimes invents emotion tags in the reply language — e.g. [高兴], [laughs],
    # [sourit] — which our English-only gesture regex won't catch. Remove all of
    # them here (half-width [] and CJK full-width 【】) before TTS.
    text = re.sub(r'\[[^\]]*\]', '', text)
    text = re.sub(r'【[^】]*】', '', text)
    text = re.sub(r'^\s*[-•–]\s*', '', text, flags=re.M)     # bullet points
    text = re.sub(r'\s+', ' ', text).strip()
    return text
