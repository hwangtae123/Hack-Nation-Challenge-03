"""Stage 3 (Prepare): deterministic document-readiness flags, each rule-cited.

Given a household's required vs. present documents (and optionally document
dates and known internal conflicts), this produces reason-coded readiness flags
and a READY_TO_REVIEW / NEEDS_REVIEW status -- all deterministically, each flag
backed by a citation from the frozen rule corpus. It also attaches a NEUTRAL
income-vs-threshold comparison from calculate.py.

It never decides eligibility. Readiness is about whether the documents are
present, current (60-day convention), and internally consistent -- not about
whether income qualifies. Income level never drives the readiness status.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from typing import Any, Optional

from rag import config
from rag.calculate import IncomeSource, summarize_income

RULES_PATH = config.REPO_ROOT / "rules" / "rule_corpus.jsonl"

# The frozen 60-day currency convention (CH-READINESS-001).
CURRENCY_WINDOW_DAYS = 60

# Reason codes.
MISSING_DOCUMENT = "MISSING_DOCUMENT"
EXPIRED_DOCUMENT = "EXPIRED_DOCUMENT"

READY = "READY_TO_REVIEW"
NEEDS_REVIEW = "NEEDS_REVIEW"

_DECISION_BOUNDARY = (
    "No eligibility, approval, denial, or priority decision is made here. "
    "This is a document-readiness check for a human reviewer."
)


@lru_cache(maxsize=1)
def _rules() -> dict[str, dict[str, Any]]:
    rules: dict[str, dict[str, Any]] = {}
    with RULES_PATH.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rules[r["rule_id"]] = r
    return rules


def _cite(rule_id: str) -> dict[str, Any]:
    """Return a citation record for a rule id from the frozen corpus."""
    r = _rules().get(rule_id, {})
    return {
        "rule_id": rule_id,
        "text": r.get("text"),
        "source_url": r.get("source_url"),
        "effective_date": r.get("effective_date"),
        "source_locator": r.get("source_locator"),
    }


@dataclass
class ReadinessFlag:
    """One reason a household is not yet ready, with a rule citation."""

    code: str
    detail: str
    rule: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "detail": self.detail, "rule": self.rule}


@dataclass
class ReadinessAssessment:
    """Deterministic readiness result. Carries no eligibility verdict."""

    household_id: str
    readiness_status: str
    flags: list[ReadinessFlag] = field(default_factory=list)
    income_comparison: dict[str, Any] = field(default_factory=dict)
    citations: list[dict[str, Any]] = field(default_factory=list)
    decision_boundary: str = _DECISION_BOUNDARY

    def to_dict(self) -> dict[str, Any]:
        return {
            "household_id": self.household_id,
            "readiness_status": self.readiness_status,
            "flags": [f.to_dict() for f in self.flags],
            "income_comparison": self.income_comparison,
            "citations": self.citations,
            "decision_boundary": self.decision_boundary,
        }


def _expired_docs(
    document_dates: dict[str, str], reference_date: date
) -> list[tuple[str, int]]:
    """Return (doc_type, age_in_days) for documents older than the window."""
    stale: list[tuple[str, int]] = []
    for doc_type, iso in document_dates.items():
        age = (reference_date - date.fromisoformat(iso)).days
        if age > CURRENCY_WINDOW_DAYS:
            stale.append((doc_type, age))
    return stale


def assess_readiness(
    household_id: str,
    household_size: int,
    required_document_types: list[str],
    present_document_types: list[str],
    income_sources: list[IncomeSource],
    document_dates: Optional[dict[str, str]] = None,
    reference_date: Optional[date] = None,
    known_conflicts: Optional[list[str]] = None,
    threshold_pct: int = 60,
) -> ReadinessAssessment:
    """Assess document readiness deterministically. No eligibility decision."""
    reference_date = reference_date or date.today()
    present = set(present_document_types)
    flags: list[ReadinessFlag] = []

    # Missing required documents.
    for doc_type in required_document_types:
        if doc_type not in present:
            flags.append(
                ReadinessFlag(
                    code=MISSING_DOCUMENT,
                    detail=f"Required document '{doc_type}' is not present.",
                    rule=_cite("CH-READINESS-001"),
                )
            )

    # Documents older than the 60-day currency window.
    for doc_type, age in _expired_docs(document_dates or {}, reference_date):
        flags.append(
            ReadinessFlag(
                code=EXPIRED_DOCUMENT,
                detail=f"Document '{doc_type}' is {age} days old, beyond the "
                f"{CURRENCY_WINDOW_DAYS}-day currency window.",
                rule=_cite("CH-READINESS-001"),
            )
        )

    # Caller-supplied internal-consistency conflicts (e.g. PAY_STUB_TOTAL_CONFLICT).
    for code in known_conflicts or []:
        flags.append(
            ReadinessFlag(
                code=code,
                detail="Internally inconsistent evidence flagged during extraction.",
                rule=_cite("CH-READINESS-001"),
            )
        )

    status = NEEDS_REVIEW if flags else READY
    comparison = summarize_income(income_sources, household_size, threshold_pct).to_dict()

    citations = [_cite("CH-READINESS-001"), _cite("CH-DECISION-001")]
    return ReadinessAssessment(
        household_id=household_id,
        readiness_status=status,
        flags=flags,
        income_comparison=comparison,
        citations=citations,
    )
