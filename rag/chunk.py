"""Strategy-specific chunkers: markdown -> list[Chunk] (pure functions).

Each document is split per its ``config.DOCS[sid]["strategy"]``:
  * statute  - by clause markers ((g), (1), (A), (i)...), small clauses merged
  * prose    - by section heading (5-6, 5-6 A.), long sections token-windowed
  * table    - exhibit5_1/exhibit4_1 by list item; appendix3 by per-page block
  * category - irs pub windowed and filtered to income/certification content

Page numbers come from the ``<!-- page N -->`` markers parse.py leaves in the
markdown; those markers are stripped from chunk text. Chunks whose text carries
an income-limit-magnitude dollar amount (a lookup, not RAG) or another program's
rules are dropped. chunk_index is per-document and contiguous after filtering.
"""
from __future__ import annotations

import logging
import re
from typing import Callable, Optional

import tiktoken

from rag import config, parse
from rag.schema import Chunk, base_metadata

logger = logging.getLogger(__name__)

_ENC = tiktoken.get_encoding("cl100k_base")

_PAGE_RE = re.compile(r"^<!--\s*page\s+(\d+)\s*-->\s*$")
_MONEY_RE = re.compile(r"\$\s?\d{2,3},\d{3}")  # income-limit magnitude -> skip
_OTHER_PROGRAM_RE = re.compile(r"\bSection\s+236\b|\bRent\s+Supplement\b|\bRAP\b")
_CLAUSE_RE = re.compile(r"^\(([0-9]{1,2}|[A-Za-z]{1,4})\)")
_HEADING_RE = re.compile(r"^(5-\d+)\b\s*(.*)$")
_ITEM_RE = re.compile(r"^\((\d+|[a-z])\)\s")

# Candidate = (text, breadcrumb, page, form_8823_category)
Candidate = tuple[str, str, int, Optional[str]]


def _num_tokens(text: str) -> int:
    return len(_ENC.encode(text))


def _clean(text: str) -> str:
    """Collapse whitespace and blank lines into flowing text."""
    text = re.sub(r"[ \t]+", " ", text)
    lines = [ln.strip() for ln in text.splitlines()]
    return " ".join(ln for ln in lines if ln).strip()


def _lines_with_pages(markdown: str) -> list[tuple[int, str]]:
    """Return (page_no, line) pairs, consuming the page-marker comments."""
    page = 1
    out: list[tuple[int, str]] = []
    for ln in markdown.splitlines():
        m = _PAGE_RE.match(ln.strip())
        if m:
            page = int(m.group(1))
            continue
        out.append((page, ln))
    return out


def _token_windows(text: str, target: int, max_tokens: int, overlap: int) -> list[str]:
    """Split text into overlapping token windows (no-op if already small)."""
    toks = _ENC.encode(text)
    if len(toks) <= max_tokens:
        return [text]
    out: list[str] = []
    start = 0
    while start < len(toks):
        end = min(start + target, len(toks))
        out.append(_ENC.decode(toks[start:end]).strip())
        if end == len(toks):
            break
        start = max(end - overlap, start + 1)
    return out


# --------------------------------------------------------------------------
# statute: usc_42_g, cfr_1_42_5
# --------------------------------------------------------------------------
def _chunk_statute(sid: str, markdown: str) -> list[Candidate]:
    citation = config.DOCS[sid]["citation"]
    segments: list[tuple[str, int, list[str]]] = []  # (marker, page, lines)
    for page, line in _lines_with_pages(markdown):
        m = _CLAUSE_RE.match(line.strip())
        if m:
            segments.append((m.group(1), page, [line]))
        elif segments:
            segments[-1][2].append(line)
        else:
            segments.append(("", page, [line]))  # preamble

    out: list[Candidate] = []
    buf_lines: list[str] = []
    buf_marker: str = ""
    buf_page: int = 1
    for marker, page, lines in segments:
        if not buf_lines:
            buf_marker, buf_page = marker, page
        buf_lines.extend(lines)
        if _num_tokens(_clean("\n".join(buf_lines))) >= 300:
            bc = f"[{citation} > ({buf_marker})]" if buf_marker else f"[{citation}]"
            out.append((_clean("\n".join(buf_lines)), bc, buf_page, None))
            buf_lines = []
    if buf_lines:
        bc = f"[{citation} > ({buf_marker})]" if buf_marker else f"[{citation}]"
        out.append((_clean("\n".join(buf_lines)), bc, buf_page, None))
    return out


# --------------------------------------------------------------------------
# prose: hud_4350_3_ch5
# --------------------------------------------------------------------------
def _chunk_prose(sid: str, markdown: str) -> list[Candidate]:
    citation = config.DOCS[sid]["citation"]
    sections: list[tuple[str, int, list[str]]] = []
    for page, line in _lines_with_pages(markdown):
        m = _HEADING_RE.match(line.strip())
        if m:
            heading = f"{m.group(1)} {m.group(2)}".strip()
            sections.append((heading, page, []))
        elif sections:
            sections[-1][2].append(line)
    out: list[Candidate] = []
    for heading, page, lines in sections:
        body = _clean("\n".join(lines))
        if not body:
            continue
        bc = f"[HUD Handbook 4350.3 > Chapter 5 > {heading}]"
        for piece in _token_windows(body, target=700, max_tokens=1000, overlap=120):
            out.append((piece, bc, page, None))
    return out


# --------------------------------------------------------------------------
# table: exhibit5_1 (numbered items), exhibit4_1 (small list), appendix3 (blocks)
# --------------------------------------------------------------------------
def _chunk_exhibit_items(sid: str, markdown: str) -> list[Candidate]:
    citation = config.DOCS[sid]["citation"]
    section = ""
    items: list[tuple[str, str, int, list[str]]] = []  # (section, id, page, lines)
    for page, line in _lines_with_pages(markdown):
        s = line.strip()
        up = s.upper()
        if "INCOME INCLUSIONS" in up:
            section = "Income Inclusions"
            continue
        if "INCOME EXCLUSIONS" in up:
            section = "Income Exclusions"
            continue
        m = _ITEM_RE.match(s)
        if m:
            items.append((section, m.group(1), page, [line]))
        elif items:
            items[-1][3].append(line)
    out: list[Candidate] = []
    for sec, item_id, page, lines in items:
        text = _clean("\n".join(lines))
        if not text:
            continue
        crumb = f"[{citation} > {sec} > ({item_id})]" if sec else f"[{citation} > ({item_id})]"
        out.append((text, crumb, page, None))
    return out


def _chunk_blocks(sid: str, markdown: str) -> list[Candidate]:
    """Per-page blank-line-separated blocks (used for the messy appendix3 table)."""
    citation = config.DOCS[sid]["citation"]
    out: list[Candidate] = []
    by_page: dict[int, list[str]] = {}
    for page, line in _lines_with_pages(markdown):
        by_page.setdefault(page, []).append(line)
    for page, lines in by_page.items():
        block: list[str] = []
        for line in lines + [""]:
            if line.strip():
                block.append(line)
            elif block:
                text = _clean("\n".join(block))
                block = []
                if _num_tokens(text) >= 12:  # drop tiny noise fragments
                    out.append((text, f"[{citation} > p.{page}]", page, None))
    return out


def _chunk_table(sid: str, markdown: str) -> list[Candidate]:
    # exhibit5_1 is a clean numbered list; the others are free-form and split
    # more reliably into per-page blocks.
    if sid == "hud_4350_3_exhibit5_1":
        return _chunk_exhibit_items(sid, markdown)
    return _chunk_blocks(sid, markdown)


# --------------------------------------------------------------------------
# category: irs_pub5913 (window + income/certification relevance filter)
# --------------------------------------------------------------------------
_IRS_KEEP_CATEGORIES = ("11a", "11c", "11d", "11i")
_IRS_RELEVANCE = re.compile(
    r"household income|income eligibility|income limit|over-?income|recertif|"
    r"annual certification|income certification|tenant income|available unit rule|"
    r"gross annual income|certify|next available unit|student",
    re.IGNORECASE,
)
_IRS_CATEGORY_RE = re.compile(r"\bCategory\s+(11[a-q])\b", re.IGNORECASE)
_IRS_MAX_CHUNKS = 30


def _chunk_category(sid: str, markdown: str) -> list[Candidate]:
    citation = config.DOCS[sid]["citation"]
    # Build ~550-token windows, remembering each window's start page.
    windows: list[tuple[int, str]] = []
    buf: list[str] = []
    buf_page = 1
    for page, line in _lines_with_pages(markdown):
        if not buf:
            buf_page = page
        buf.append(line)
        if _num_tokens(_clean("\n".join(buf))) >= 550:
            windows.append((buf_page, _clean("\n".join(buf))))
            buf = []
    if buf:
        windows.append((buf_page, _clean("\n".join(buf))))

    scored: list[tuple[int, int, str, Optional[str]]] = []  # (score, page, text, category)
    for page, text in windows:
        hits = len(_IRS_RELEVANCE.findall(text))
        if hits == 0:
            continue
        cats = _IRS_CATEGORY_RE.findall(text)
        category = next((c.lower() for c in cats if c.lower() in _IRS_KEEP_CATEGORIES), None)
        scored.append((hits, page, text, category))

    scored.sort(key=lambda x: -x[0])
    kept = scored[:_IRS_MAX_CHUNKS]
    kept.sort(key=lambda x: x[1])  # back to document order
    out: list[Candidate] = []
    for _, page, text, category in kept:
        label = f"Form 8823 Category {category}" if category else "Income & Certification"
        out.append((text, f"[{citation} > {label}]", page, category))
    return out


_STRATEGY: dict[str, Callable[[str, str], list[Candidate]]] = {
    "statute": _chunk_statute,
    "prose": _chunk_prose,
    "table": _chunk_table,
    "category": _chunk_category,
}


def _make(sid: str, idx: int, cand: Candidate) -> Optional[Chunk]:
    text, breadcrumb, page, category = cand
    text = text.strip()
    if not text:
        return None
    if _MONEY_RE.search(text):
        logger.warning("skip %s#%d: income-limit money pattern", sid, idx)
        return None
    if _OTHER_PROGRAM_RE.search(text):
        logger.warning("skip %s#%d: non-LIHTC program content", sid, idx)
        return None
    return Chunk(
        text=text,
        chunk_index=idx,
        breadcrumb=breadcrumb,
        source_page=page,
        form_8823_category=category,
        **base_metadata(sid),
    )


def chunk_document(source_id: str, markdown: str | None = None) -> list[Chunk]:
    """Chunk one document per its configured strategy."""
    md = markdown if markdown is not None else parse.parse_document(source_id)
    strategy = config.DOCS[source_id]["strategy"]
    candidates = _STRATEGY[strategy](source_id, md)
    chunks: list[Chunk] = []
    idx = 0
    for cand in candidates:
        chunk = _make(source_id, idx, cand)
        if chunk is not None:
            chunks.append(chunk)
            idx += 1
    logger.info("chunked %s -> %d chunks (%s)", source_id, len(chunks), strategy)
    return chunks


def chunk_all() -> list[Chunk]:
    """Chunk every registered document."""
    chunks: list[Chunk] = []
    for source_id in config.DOCS:
        chunks.extend(chunk_document(source_id))
    return chunks
