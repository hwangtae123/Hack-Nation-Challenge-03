"""Tests for Stage 3 deterministic readiness (offline: no network, no key)."""
import json
from datetime import date

from rag.calculate import IncomeSource
from rag.prepare import (
    EXPIRED_DOCUMENT,
    MISSING_DOCUMENT,
    NEEDS_REVIEW,
    READY,
    assess_readiness,
)

BANNED = ("eligible", "ineligible", "approved", "denied", "prioritiz", "qualif")
REF = date(2026, 7, 19)
REQUIRED = ["application_summary", "pay_stub", "employment_letter"]
INCOME = [IncomeSource(amount=2000, frequency="monthly", source_document_id="D1")]


def test_complete_household_is_ready():
    a = assess_readiness(
        "HH-001", 1, REQUIRED, REQUIRED, INCOME, reference_date=REF
    )
    assert a.readiness_status == READY
    assert a.flags == []
    assert a.income_comparison["comparison"] in ("below_or_equal", "above", "no_frozen_threshold")
    assert a.income_comparison["effective_date"] == "2026-05-01"


def test_missing_document_needs_review():
    a = assess_readiness(
        "HH-003", 3, REQUIRED, ["application_summary", "pay_stub"], INCOME, reference_date=REF
    )
    assert a.readiness_status == NEEDS_REVIEW
    codes = [f.code for f in a.flags]
    assert MISSING_DOCUMENT in codes
    missing = next(f for f in a.flags if f.code == MISSING_DOCUMENT)
    assert missing.rule["rule_id"] == "CH-READINESS-001"
    assert missing.rule["source_url"]


def test_expired_document_flagged():
    a = assess_readiness(
        "HH-009", 2, REQUIRED, REQUIRED, INCOME,
        document_dates={"pay_stub": "2026-01-01"},  # ~199 days before REF
        reference_date=REF,
    )
    assert a.readiness_status == NEEDS_REVIEW
    assert EXPIRED_DOCUMENT in [f.code for f in a.flags]


def test_recent_document_not_flagged():
    a = assess_readiness(
        "HH-010", 2, REQUIRED, REQUIRED, INCOME,
        document_dates={"pay_stub": "2026-06-27"},  # 22 days before REF
        reference_date=REF,
    )
    assert EXPIRED_DOCUMENT not in [f.code for f in a.flags]


def test_known_conflict_surfaces():
    a = assess_readiness(
        "HH-002", 2, REQUIRED, REQUIRED, INCOME,
        reference_date=REF, known_conflicts=["PAY_STUB_TOTAL_CONFLICT"],
    )
    assert a.readiness_status == NEEDS_REVIEW
    assert "PAY_STUB_TOTAL_CONFLICT" in [f.code for f in a.flags]


def test_result_carries_human_boundary_and_no_verdict():
    a = assess_readiness("HH-001", 1, REQUIRED, REQUIRED, INCOME, reference_date=REF)
    ids = [c["rule_id"] for c in a.citations]
    assert "CH-DECISION-001" in ids and "CH-READINESS-001" in ids
    blob = json.dumps(a.to_dict()).lower()
    # The rule prose intentionally names verdict words; check only our own fields.
    own = json.dumps({
        "status": a.readiness_status,
        "flags": [{"code": f.code, "detail": f.detail} for f in a.flags],
        "comparison": a.income_comparison.get("comparison"),
        "boundary": a.decision_boundary,
    }).lower()
    for w in BANNED:
        assert w not in own, f"verdict word leaked into our output: {w}"
