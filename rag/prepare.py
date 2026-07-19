"""Stage 3 (Prepare): deterministic document-readiness flags, each rule-cited.

Given a household's required vs. present documents (and optionally document
dates and known internal conflicts), this produces reason-coded readiness flags,
a full per-document-type status grid (present / missing / expired), and a
READY_TO_REVIEW / NEEDS_REVIEW status -- all deterministically, each flag backed
by a citation from the frozen rule corpus. It also attaches a NEUTRAL
income-vs-threshold comparison from calculate.py.

Currency windows are per document type, read from ``checklist_lihtc_2026.json``
(not hardcoded): a pay stub and a benefit letter do not need to be equally
fresh. See ``is_expired`` for the pure expiry rule and ``_checklist`` for the
loader.

It never decides eligibility. Readiness is about whether the documents are
present, current, and internally consistent -- not about whether income
qualifies. Income level never drives the readiness status.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from typing import Any, Optional

from rag import config
from rag.calculate import IncomeSource, summarize_income

logger = logging.getLogger(__name__)

RULES_PATH = config.REPO_ROOT / "rules" / "rule_corpus.jsonl"
CHECKLIST_PATH = config.RAG_DIR / "checklist_lihtc_2026.json"

# Fallback currency window for a document type that isn't in the checklist
# file at all. CH-READINESS-001's convention; the checklist file is the real
# source of truth and should list every document type this pipeline handles.
DEFAULT_MAX_AGE_DAYS = 60

# Reason codes.
MISSING_DOCUMENT = "MISSING_DOCUMENT"
EXPIRED_DOCUMENT = "EXPIRED_DOCUMENT"

READY = "READY_TO_REVIEW"
NEEDS_REVIEW = "NEEDS_REVIEW"

# document_status statuses (distinct from the flag/readiness vocabulary above).
PRESENT = "present"
MISSING = "missing"
EXPIRED = "expired"

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


@lru_cache(maxsize=1)
def _checklist() -> dict[str, dict[str, Any]]:
    """Load the structured gold checklist: doc_type -> {label, required, max_age_days}."""
    with CHECKLIST_PATH.open(encoding="utf-8") as f:
        rows = json.load(f)
    return {r["doc_type"]: r for r in rows}


def checklist_entry(doc_type: str) -> dict[str, Any]:
    """Return the checklist entry for a document type, with a safe fallback
    for any type not listed in the file (logged, since the file should be the
    single source of truth for every document type this pipeline handles)."""
    entry = _checklist().get(doc_type)
    if entry is None:
        logger.warning("doc_type %r not in checklist file; using default max_age_days", doc_type)
        return {"doc_type": doc_type, "label": doc_type, "required": False, "max_age_days": DEFAULT_MAX_AGE_DAYS}
    return entry


def is_expired(doc_date: Optional[str], max_age_days: Optional[int], reference_date: date) -> bool:
    """Return True iff ``doc_date`` (ISO string) is older than ``max_age_days``.

    ``max_age_days=None`` means the document type never expires (e.g. a
    self-declared application form) -- always returns False. ``doc_date=None``
    means no date evidence was given; the absence of a date is not itself
    treated as staleness, so this also returns False (the caller decides
    separately whether an undated document is acceptable).
    """
    if max_age_days is None or doc_date is None:
        return False
    age = (reference_date - date.fromisoformat(doc_date)).days
    return age > max_age_days


@dataclass
class ReadinessFlag:
    """One reason a household is not yet ready, with a rule citation."""

    code: str
    detail: str
    rule: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "detail": self.detail, "rule": self.rule}


@dataclass
class DocumentStatus:
    """The checklist status of one document type: present, missing, or expired.

    This is the full grid (every relevant document type), distinct from
    ``flags`` (only the problems). A document type appears here if it is
    required for this household or if it was actually supplied.
    """

    doc_type: str
    label: str
    status: str  # PRESENT | MISSING | EXPIRED
    detail: str
    max_age_days: Optional[int]
    age_days: Optional[int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_type": self.doc_type,
            "label": self.label,
            "status": self.status,
            "detail": self.detail,
            "max_age_days": self.max_age_days,
            "age_days": self.age_days,
        }


@dataclass
class ReadinessAssessment:
    """Deterministic readiness result. Carries no eligibility verdict."""

    household_id: str
    readiness_status: str
    flags: list[ReadinessFlag] = field(default_factory=list)
    document_status: list[DocumentStatus] = field(default_factory=list)
    income_comparison: dict[str, Any] = field(default_factory=dict)
    citations: list[dict[str, Any]] = field(default_factory=list)
    decision_boundary: str = _DECISION_BOUNDARY

    def to_dict(self) -> dict[str, Any]:
        return {
            "household_id": self.household_id,
            "readiness_status": self.readiness_status,
            "flags": [f.to_dict() for f in self.flags],
            "document_status": [d.to_dict() for d in self.document_status],
            "income_comparison": self.income_comparison,
            "citations": self.citations,
            "decision_boundary": self.decision_boundary,
        }


def _build_document_status(
    required_document_types: list[str],
    present_document_types: list[str],
    document_dates: dict[str, str],
    reference_date: date,
) -> list[DocumentStatus]:
    """Build the full present/missing/expired grid for every relevant doc type.

    "Relevant" means required for this household (caller-supplied or flagged
    required in the checklist file) or actually present -- we don't clutter
    the grid with document types nobody asked for and nobody supplied.
    """
    required = set(required_document_types)
    present = set(present_document_types)
    checklist_required = {dt for dt, e in _checklist().items() if e.get("required")}
    relevant = list(_checklist().keys())  # preserve checklist file order first
    relevant += [dt for dt in (required | present) if dt not in _checklist()]

    rows: list[DocumentStatus] = []
    for doc_type in relevant:
        is_required = doc_type in required or doc_type in checklist_required
        is_present = doc_type in present
        if not is_required and not is_present:
            continue
        entry = checklist_entry(doc_type)
        label = entry.get("label", doc_type)
        max_age = entry.get("max_age_days")

        if not is_present:
            rows.append(DocumentStatus(doc_type, label, MISSING, f"{label} is not present.", max_age, None))
            continue

        doc_date = document_dates.get(doc_type)
        age_days = (reference_date - date.fromisoformat(doc_date)).days if doc_date else None
        if is_expired(doc_date, max_age, reference_date):
            detail = f"{label} is {age_days} days old, beyond the {max_age}-day currency window."
            rows.append(DocumentStatus(doc_type, label, EXPIRED, detail, max_age, age_days))
        else:
            rows.append(DocumentStatus(doc_type, label, PRESENT, f"{label} is present.", max_age, age_days))
    return rows


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
    document_dates = document_dates or {}
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

    # Documents older than their document type's currency window.
    for doc_type in present_document_types:
        doc_date = document_dates.get(doc_type)
        max_age = checklist_entry(doc_type).get("max_age_days")
        if is_expired(doc_date, max_age, reference_date):
            age = (reference_date - date.fromisoformat(doc_date)).days
            flags.append(
                ReadinessFlag(
                    code=EXPIRED_DOCUMENT,
                    detail=f"Document '{doc_type}' is {age} days old, beyond the "
                    f"{max_age}-day currency window.",
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
    document_status = _build_document_status(
        required_document_types, present_document_types, document_dates, reference_date
    )

    citations = [_cite("CH-READINESS-001"), _cite("CH-DECISION-001")]
    return ReadinessAssessment(
        household_id=household_id,
        readiness_status=status,
        flags=flags,
        document_status=document_status,
        income_comparison=comparison,
        citations=citations,
    )
