"""Tests for the deterministic calculation module.

These run fully offline (no network, no API key): the threshold table is a local
JSON lookup and the math is pure.
"""
import json
import math

import pytest

from rag import config
from rag.calculate import (
    ComparisonResult,
    IncomeSource,
    annualize,
    compare_to_threshold,
    limit_for,
    summarize_income,
)

BANNED_VERDICT_WORDS = ("eligible", "ineligible", "approved", "denied", "qualif", "prioritiz")


@pytest.mark.parametrize(
    "amount,freq,expected",
    [
        (1000, "weekly", 52000.0),
        (1000, "biweekly", 26000.0),
        (1000, "semimonthly", 24000.0),
        (1000, "monthly", 12000.0),
        (1000, "annual", 1000.0),
        (28.5 * 76, "biweekly", round(28.5 * 76 * 26, 2)),
    ],
)
def test_annualize(amount, freq, expected):
    assert math.isclose(annualize(amount, freq), expected, abs_tol=0.01)


def test_annualize_rejects_bad_input():
    with pytest.raises(ValueError):
        annualize(100, "hourly")
    with pytest.raises(ValueError):
        annualize(-1, "weekly")


def test_limit_for_matches_threshold_json():
    limits = config.load_thresholds()["limits"]
    for pct in ("50", "60"):
        for size in range(1, 9):
            assert limit_for(size, int(pct)) == int(limits[pct][str(size)])


def test_limit_for_off_table_is_none():
    assert limit_for(9, 60) is None
    assert limit_for(4, 999) is None


def test_boundary_is_inclusive():
    # income exactly equal to the threshold is treated as below_or_equal
    assert compare_to_threshold(102840, 102840) == "below_or_equal"
    assert compare_to_threshold(102839.99, 102840) == "below_or_equal"
    assert compare_to_threshold(102840.01, 102840) == "above"
    assert compare_to_threshold(50000, None) == "no_frozen_threshold"


def test_summarize_income_sums_and_compares():
    # Two sources: 40h/wk-ish, exact numbers chosen to stay under the size-4 limit.
    sources = [
        IncomeSource(amount=2000, frequency="monthly", source_document_id="D1"),
        IncomeSource(amount=500, frequency="monthly", source_document_id="D2"),
    ]
    res = summarize_income(sources, household_size=4, threshold_pct=60)
    assert isinstance(res, ComparisonResult)
    assert res.annualized_income == 30000.0  # (2000 + 500) * 12
    assert res.threshold == limit_for(4, 60)
    assert res.comparison == "below_or_equal"
    assert res.effective_date == "2026-05-01"
    assert res.area.startswith("Boston")
    assert len(res.per_source) == 2
    assert res.status == "computed"
    assert res.abstain_reason is None


def test_summarize_income_abstains_when_no_sources():
    # An empty list almost always means "nothing confirmed yet," not a
    # verified zero income -- this must not present as a real $0 comparison.
    res = summarize_income([], household_size=4, threshold_pct=60)
    assert res.status == "abstain"
    assert res.abstain_reason
    assert res.annualized_income == 0.0


def test_summarize_income_above_threshold():
    sources = [IncomeSource(amount=200000, frequency="annual")]
    res = summarize_income(sources, household_size=1, threshold_pct=60)
    assert res.comparison == "above"


def test_no_eligibility_verdict_in_output():
    sources = [IncomeSource(amount=1000, frequency="weekly")]
    res = summarize_income(sources, household_size=4)
    blob = json.dumps(res.to_dict()).lower()
    for word in BANNED_VERDICT_WORDS:
        assert word not in blob, f"verdict-like word leaked into output: {word}"
    assert res.comparison in ("below_or_equal", "above", "no_frozen_threshold")
