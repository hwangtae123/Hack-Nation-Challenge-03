"""PDF -> markdown parsing with page markers and boilerplate removal.

This is the only layer that would touch the network (via LlamaParse). No
LlamaParse key is configured, so we use pdfplumber with layout-preserving text
extraction, which keeps the column spacing the table strategies rely on and
needs no network at all -- so chunking and its tests run fully offline against
the cached markdown this module writes.

HUD PDFs repeat a header/footer on every page (e.g. "Exhibit 5-1   4350.3 REV-1",
"HUD Occupancy Handbook   06/07"). Left in, that noise pollutes every chunk and
degrades embeddings, so we detect it by frequency (a normalized edge line that
recurs on >= 50% of pages) with a small regex backstop, and strip it.

Parsed output is cached at ``corpus/cache/{source_id}.md``; re-runs reuse the
cache unless ``force=True``.
"""
from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path
from typing import Optional

import pdfplumber

from rag import config

logger = logging.getLogger(__name__)

# How many non-empty lines at each page edge to consider as header/footer.
_EDGE_LINES = 3
# A normalized edge line seen on at least this fraction of pages is boilerplate.
_BOILERPLATE_RATIO = 0.5
# Regex backstop for known HUD 4350.3 running heads/feet and bare page numbers.
_BOILERPLATE_RE = [
    re.compile(r"^\s*\d+\s*$"),                       # bare page number
    re.compile(r"4350\.3\s*REV", re.IGNORECASE),
    re.compile(r"HUD\s+Occupancy\s+Handbook", re.IGNORECASE),
    re.compile(r"^\s*Chapter\s+\d+:", re.IGNORECASE),
    re.compile(r"\b\d{1,2}/\d{2}\b"),                 # mm/yy date stamp
]


def _normalize(line: str) -> str:
    """Collapse whitespace and mask digits so page-varying edges still match."""
    return re.sub(r"\d", "#", re.sub(r"\s+", " ", line).strip())


def _edge_lines(lines: list[str]) -> list[str]:
    """The first and last few non-empty lines of a page (header/footer band)."""
    non_empty = [ln for ln in lines if ln.strip()]
    return non_empty[:_EDGE_LINES] + non_empty[-_EDGE_LINES:]


def _detect_boilerplate(pages: list[list[str]]) -> set[str]:
    """Return the set of normalized edge lines that recur across pages."""
    if len(pages) < 3:
        return set()
    counts: dict[str, int] = {}
    for lines in pages:
        for ln in set(_edge_lines(lines)):
            counts[_normalize(ln)] = counts.get(_normalize(ln), 0) + 1
    threshold = _BOILERPLATE_RATIO * len(pages)
    return {norm for norm, c in counts.items() if norm and c >= threshold}


def _is_boilerplate(line: str, boiler: set[str]) -> bool:
    norm = _normalize(line)
    if norm in boiler:
        return True
    return any(rx.search(line) for rx in _BOILERPLATE_RE)


def _strip_page(lines: list[str], boiler: set[str]) -> list[str]:
    """Drop boilerplate, but only when it sits in the page's edge band."""
    non_empty = [ln for ln in lines if ln.strip()]
    edge = set(non_empty[:_EDGE_LINES] + non_empty[-_EDGE_LINES:])
    out: list[str] = []
    for ln in lines:
        if ln in edge and _is_boilerplate(ln, boiler):
            continue
        out.append(ln.rstrip())
    return out


def _collapse_blanks(text: str) -> str:
    """Squeeze 3+ blank lines down to a single blank line."""
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


def _page_indices(pdf, target_pages: Optional[list[tuple[int, int]]]) -> list[int]:
    """Zero-based page indices to extract, honoring an optional page window."""
    n = len(pdf.pages)
    if not target_pages:
        return list(range(n))
    idx: list[int] = []
    for start, end in target_pages:  # 1-based inclusive
        idx.extend(range(start - 1, min(end, n)))
    return sorted(set(i for i in idx if 0 <= i < n))


def _parse_pdf(pdf_path: Path, target_pages: Optional[list[tuple[int, int]]]) -> str:
    """Extract layout text per page, strip boilerplate, keep page markers."""
    with pdfplumber.open(str(pdf_path)) as pdf:
        indices = _page_indices(pdf, target_pages)
        raw_pages = [
            (i + 1, (pdf.pages[i].extract_text(layout=True) or "").splitlines())
            for i in indices
        ]
    boiler = _detect_boilerplate([lines for _, lines in raw_pages])
    parts: list[str] = []
    for page_no, lines in raw_pages:
        cleaned = "\n".join(_strip_page(lines, boiler)).strip()
        if cleaned:
            parts.append(f"<!-- page {page_no} -->\n{cleaned}")
    return _collapse_blanks("\n\n".join(parts))


def _parse_with_llama(pdf_path: Path, target_pages) -> Optional[str]:
    """Placeholder for the LlamaParse path (only used if a key is configured).

    Isolated here so LlamaParse can drop in without touching the rest of the
    pipeline. Returns None so callers fall back to pdfplumber.
    """
    if not config.get_llama_api_key():
        return None
    logger.info("LLAMA_CLOUD_API_KEY present but LlamaParse path not implemented; using pdfplumber.")
    return None


def parse_document(source_id: str, force: bool = False) -> str:
    """Parse one source document to markdown, using the cache when possible."""
    cache = config.cache_path(source_id)
    if cache.exists() and not force:
        logger.info("parse cache hit: %s", source_id)
        return cache.read_text(encoding="utf-8")

    meta = config.DOCS[source_id]
    src = config.doc_path(source_id)
    cache.parent.mkdir(parents=True, exist_ok=True)

    if src.suffix.lower() == ".md":
        # Already markdown (e.g. usc_42_g.md); pass through unchanged.
        text = src.read_text(encoding="utf-8")
        logger.info("parse (markdown passthrough): %s", source_id)
    else:
        text = _parse_with_llama(src, meta.get("target_pages"))
        if text is None:
            text = _parse_pdf(src, meta.get("target_pages"))
        logger.info("parse (pdfplumber): %s -> %d chars", source_id, len(text))

    cache.write_text(text, encoding="utf-8")
    return text


def parse_all(force: bool = False) -> dict[str, str]:
    """Parse every registered document; returns {source_id: markdown}."""
    return {sid: parse_document(sid, force=force) for sid in config.DOCS}
