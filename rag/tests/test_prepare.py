"""Tests for Stage 3 deterministic readiness (offline: no network, no key)."""
import json
from datetime import date

from rag.calculate import IncomeSource
from rag.prepare import (
    EXPIRED,
    EXPIRED_DOCUMENT,
    MISSING,
    MISSING_DOCUMENT,
    NEEDS_REVIEW,
    PRESENT,
    READY,
    assess_readiness,
    is_expired,
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


def test_is_expired_pure_function():
    assert is_expired("2020-01-01", None, REF) is False  # never-expires type
    assert is_expired(None, 60, REF) is False  # no date evidence isn't staleness
    assert is_expired("2026-04-10", 60, REF) is True  # ~100 days > 60
    assert is_expired("2026-04-10", 120, REF) is False  # ~100 days < 120


def test_expiry_window_differs_by_document_type():
    # pay_stub (60-day window) and benefit_letter (120-day window) dated the
    # same ~100 days ago must NOT be treated the same: only pay_stub expires.
    a = assess_readiness(
        "HH-003", 3,
        required_document_types=["application_summary", "pay_stub", "benefit_letter"],
        present_document_types=["application_summary", "pay_stub", "benefit_letter"],
        income_sources=INCOME,
        document_dates={
            "application_summary": "2020-01-01",  # never expires (max_age_days=null)
            "pay_stub": "2026-04-10",
            "benefit_letter": "2026-04-10",
        },
        reference_date=REF,
    )
    codes = [f.code for f in a.flags]
    assert codes == [EXPIRED_DOCUMENT]
    by_type = {d.doc_type: d for d in a.document_status}
    assert by_type["pay_stub"].status == EXPIRED
    assert by_type["benefit_letter"].status == PRESENT
    assert by_type["application_summary"].status == PRESENT


def test_document_status_grid_present_missing():
    a = assess_readiness(
        "HH-003", 3, REQUIRED, ["application_summary", "pay_stub"], INCOME, reference_date=REF,
    )
    by_type = {d.doc_type: d for d in a.document_status}
    assert by_type["application_summary"].status == PRESENT
    assert by_type["pay_stub"].status == PRESENT
    assert by_type["employment_letter"].status == MISSING


def test_document_status_omits_irrelevant_types():
    # gig_statement is neither required nor present for this household -> not listed.
    a = assess_readiness("HH-001", 1, REQUIRED, REQUIRED, INCOME, reference_date=REF)
    doc_types = {d.doc_type for d in a.document_status}
    assert "gig_statement" not in doc_types


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
