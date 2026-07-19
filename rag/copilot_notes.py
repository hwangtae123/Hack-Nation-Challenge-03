"""Optional, non-authoritative LLM commentary for Profile and Prepare.

Every number and status in this app is already decided deterministically:
``src/fields.py`` extracts values, ``rag/prepare.py`` builds the checklist
grid, ``rag/calculate.py`` is the only place income is compared to a frozen
threshold. This module adds nothing to that math. It only asks an LLM to
phrase a short, human-readable note about completeness/consistency --
"this field looks blank", "these two numbers don't reconcile" -- for a renter
to review before confirming or exporting. It is a suggestion, never a
determination, and it is always shown as clearly secondary to the
deterministic output it comments on.

Defense in depth, mirroring ``understand.py``'s approach:
  1. A hardened system prompt forbids eligibility/threshold/AMI language.
  2. Every generated note is scanned afterward for banned terms; if any slip
     through, the note is discarded and the caller gets an abstain result
     instead of the tainted text -- the same fail-closed pattern as
     retrieval abstention.
  3. ``prepare_notes`` structurally cannot see income amounts, thresholds, or
     AMI percentages: its signature only accepts the document checklist grid
     and flag codes, never ``income_comparison``. There is nothing to leak
     because it is never given the data in the first place.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

from rag import config

_BANNED_RE = re.compile(
    r"\b(eligible|ineligible|approve\w*|den(y|ial|ied)|qualif\w*|disqualif\w*|"
    r"prioritiz\w*|ami|area median income|income limit\w*|threshold\w*)\b"
    r"|\d{1,3}\s?%",
    re.IGNORECASE,
)

_ABSTAIN_REASON = (
    "Could not generate a safe review note for this right now. Please check "
    "the values yourself -- the checklist/fields above are the authoritative "
    "record either way."
)

PROFILE_SYSTEM_PROMPT = """You help a renter double-check ONE document's extracted \
values before they confirm them. You are not a reviewer of eligibility or income limits \
-- you only comment on whether the values look complete and internally consistent (for \
example: does gross pay reconcile with hourly rate times hours; is a required field \
blank or obviously malformed; does a date look stale). Never mention eligibility, \
qualification, approval, denial, income limits, AMI, or any percentage threshold. Never \
invent a value you were not given. Reply with 1-3 short bullet points, or the single \
line "Nothing stands out." if there is nothing worth flagging."""

PREPARE_SYSTEM_PROMPT = """You help a renter understand what to fix before their \
document packet goes to a human reviewer. You only comment on document completeness \
(present, missing, or expired) and any flagged internal conflicts you are given -- \
never on income eligibility, limits, AMI, or any percentage threshold; that data is \
deliberately withheld from you. Reply with 1-3 short, actionable bullet points, or the \
single line "Nothing stands out." if the checklist looks complete."""


@dataclass
class ReviewNotes:
    """An LLM-generated suggestion. Carries no verdict; always dismissible."""

    notes: str
    abstained: bool
    abstain_reason: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {"notes": self.notes, "abstained": self.abstained, "abstain_reason": self.abstain_reason}


def _to_review_notes(raw_text: str) -> ReviewNotes:
    """Apply the banned-term safety net to a raw LLM reply (pure, testable offline)."""
    if _BANNED_RE.search(raw_text):
        return ReviewNotes(notes="", abstained=True, abstain_reason=_ABSTAIN_REASON)
    return ReviewNotes(notes=raw_text.strip(), abstained=False)


def _call_llm(system_prompt: str, user_payload: str) -> str:
    import openai

    client = openai.OpenAI(api_key=config.get_openai_api_key())
    resp = client.chat.completions.create(
        model=config.CHAT_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_payload},
        ],
    )
    return (resp.choices[0].message.content or "").strip()


def profile_notes(document_type: str, fields: list[dict[str, Any]]) -> ReviewNotes:
    """Consistency notes for one document's extracted (possibly unconfirmed) fields.

    ``fields`` is the same allowlisted field list the Profile step already
    shows the renter; quarantined ``untrusted_instruction_text`` is dropped
    before it ever reaches the prompt, same as everywhere else downstream.
    """
    rows = [
        f"- {f['field']}: {f['value']} (confidence {f.get('confidence')})"
        for f in fields
        if f.get("field") != "untrusted_instruction_text"
    ]
    if not rows:
        return ReviewNotes(notes="Nothing stands out.", abstained=False)
    payload = f"Document type: {document_type}\nExtracted fields:\n" + "\n".join(rows)
    return _to_review_notes(_call_llm(PROFILE_SYSTEM_PROMPT, payload))


def prepare_notes(document_status: list[dict[str, Any]], flags: list[dict[str, Any]]) -> ReviewNotes:
    """Completeness/consistency notes for a household's document checklist grid.

    Deliberately takes no ``income_comparison``/threshold/AMI data as input --
    there is nothing here an LLM could comment on relative to an income limit
    even if it tried, because it is never given that data.
    """
    if not document_status and not flags:
        return ReviewNotes(notes="Nothing stands out.", abstained=False)
    rows = [f"- {d['label']}: {d['status']} ({d['detail']})" for d in document_status]
    payload = "Document checklist:\n" + "\n".join(rows)
    if flags:
        payload += "\n\nFlagged issues:\n" + "\n".join(f"- {fl['code']}: {fl['detail']}" for fl in flags)
    return _to_review_notes(_call_llm(PREPARE_SYSTEM_PROMPT, payload))
