"""Shared BM25 tokenizer.

Index and retrieval must tokenize identically, so both import this. Kept simple
and deterministic: lowercase, split on non-alphanumerics. Legal identifiers like
``42(g)`` split into ``42`` and ``g`` -- fine for term matching.
"""
from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def bm25_tokens(text: str) -> list[str]:
    """Lowercase, alphanumeric token list for BM25 scoring."""
    return _TOKEN_RE.findall(text.lower())
