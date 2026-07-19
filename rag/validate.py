"""Corpus validation checklist (a hard gate before indexing).

Runs the rag/claude.md checklist over the chunk set. A non-empty result means
the corpus must not be indexed until fixed.

One deliberate refinement: the "no eligibility verdict" rule is enforced as a
verdict-PHRASE check, not a bare-word ban. HUD/IRS rule prose is saturated with
legitimate technical uses -- "eligible basis", "income-eligible household",
"Changes in Eligible Basis" -- and banning the word would delete the corpus's
most important income-eligibility content. What must never appear is a
synthesized decision about a person ("you are eligible", "the applicant is
approved"). That is what we detect.
"""
from __future__ import annotations

import re

from rag import config
from rag.schema import Chunk

_MONEY_RE = re.compile(r"\$\s?\d{2,3},\d{3}")
_OTHER_PROGRAM_RE = re.compile(r"\bSection\s+236\b|\bRent\s+Supplement\b|\bRAP\b")
_BOILERPLATE_RE = re.compile(r"HUD Occupancy Handbook|4350\.3\s*REV", re.IGNORECASE)
_VERDICT_RE = re.compile(
    r"\byou\s+(?:are|may be)\s+(?:not\s+)?(?:eligible|approved|denied|qualified)\b"
    r"|\b(?:applicant|renter|household|tenant|family)s?\s+(?:is|are|has been|have been)"
    r"\s+(?:hereby\s+)?(?:approved|denied|disqualified|prioritized)\b"
    r"|\bmark(?:ed|s|ing)?\b[^.\n]{0,30}\b(?:approved|denied|eligible|ineligible)\b"
    r"|\b(?:is|are)\s+hereby\s+(?:approved|denied)\b",
    re.IGNORECASE,
)
_STAGES = ("profile", "understand", "prepare")


class ValidationError(Exception):
    """Raised by assert_valid when the corpus fails the checklist."""


def validate_chunks(chunks: list[Chunk]) -> list[str]:
    """Return a list of checklist violations (empty means the corpus is valid)."""
    problems: list[str] = []

    for c in chunks:
        cid = c.chunk_id
        if not c.citation or not c.source_url:
            problems.append(f"{cid}: missing citation or source_url")
        if not c.breadcrumb:
            problems.append(f"{cid}: missing breadcrumb")
        if _MONEY_RE.search(c.text):
            problems.append(f"{cid}: contains income-limit money pattern (numbers are a lookup)")
        if _VERDICT_RE.search(c.text):
            problems.append(f"{cid}: contains an eligibility verdict phrase")
        if _OTHER_PROGRAM_RE.search(c.text):
            problems.append(f"{cid}: contains non-LIHTC program content")
        if _BOILERPLATE_RE.search(c.text):
            problems.append(f"{cid}: contains residual header/footer boilerplate")

    total = len(chunks)
    if not (config.CHUNK_TOTAL_MIN <= total <= config.CHUNK_TOTAL_MAX):
        problems.append(
            f"total chunk count {total} outside [{config.CHUNK_TOTAL_MIN}, {config.CHUNK_TOTAL_MAX}]"
        )

    if total:
        irs = sum(1 for c in chunks if c.source_id == "irs_pub5913")
        if irs / total > config.IRS_MAX_CHUNK_SHARE:
            problems.append(
                f"irs_pub5913 share {irs / total:.1%} exceeds {config.IRS_MAX_CHUNK_SHARE:.0%}"
            )

    covered = {s for c in chunks for s in c.stage}
    for stage in _STAGES:
        if stage not in covered:
            problems.append(f"stage {stage!r} has no chunks")

    return problems


def assert_valid(chunks: list[Chunk]) -> None:
    """Raise ValidationError if the corpus fails any checklist item."""
    problems = validate_chunks(chunks)
    if problems:
        raise ValidationError(
            f"{len(problems)} validation problem(s):\n  " + "\n  ".join(problems)
        )
