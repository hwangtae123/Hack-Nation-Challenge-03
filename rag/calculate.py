"""Deterministic income annualization and threshold comparison.

This module is the ONLY place income math happens, and it is fully deterministic:
limits are looked up from the Boston FY2026 threshold JSON (never retrieved,
never derived), income is annualized with fixed frequency multipliers, and the
result is a neutral side-by-side comparison.

Hard boundary (challenge rule CH-DECISION-001): this module compares an amount
with a frozen threshold and returns ``below_or_equal`` / ``above`` /
``no_frozen_threshold``. It NEVER labels anyone eligible, ineligible, approved,
denied, or prioritized. The final determination is human and program-specific.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from rag import config

# Fixed pay-frequency -> periods-per-year multipliers (challenge convention).
FREQUENCY: dict[str, int] = {
    "weekly": 52,
    "biweekly": 26,
    "semimonthly": 24,
    "monthly": 12,
    "annual": 1,
}

Comparison = Literal["below_or_equal", "above", "no_frozen_threshold"]
Status = Literal["computed", "abstain"]


@dataclass(frozen=True)
class IncomeSource:
    """One recurring, independently documented income source."""

    amount: float
    frequency: str
    source_document_id: str | None = None


@dataclass(frozen=True)
class ComparisonResult:
    """A neutral comparison of annualized income against a frozen limit.

    Deliberately carries no verdict field. ``comparison`` is a factual relation,
    not an eligibility decision.

    ``status`` is ``"abstain"`` when there isn't enough confirmed evidence to
    trust the comparison yet (currently: no income sources at all) -- an empty
    list almost always means "nothing confirmed yet," not "verified zero
    income," so showing a numeric $0/below_or_equal result would misrepresent
    an incomplete workflow as a real, confirmed comparison. Callers should
    render an abstain state instead of the numeric comparison when this is set.
    """

    annualized_income: float
    household_size: int
    threshold_pct: int
    threshold: int | None
    comparison: Comparison
    area: str
    rule_year: str
    effective_date: str
    source_url: str
    per_source: list[dict[str, Any]] = field(default_factory=list)
    status: Status = "computed"
    abstain_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "annualized_income": self.annualized_income,
            "household_size": self.household_size,
            "threshold_pct": self.threshold_pct,
            "threshold": self.threshold,
            "comparison": self.comparison,
            "area": self.area,
            "rule_year": self.rule_year,
            "effective_date": self.effective_date,
            "source_url": self.source_url,
            "per_source": self.per_source,
            "status": self.status,
            "abstain_reason": self.abstain_reason,
        }


def annualize(amount: float, frequency: str) -> float:
    """Annualize a recurring gross amount using its explicit pay frequency."""
    if frequency not in FREQUENCY:
        raise ValueError(f"Unsupported frequency: {frequency!r}")
    if amount < 0:
        raise ValueError("Amount must be non-negative")
    return round(float(amount) * FREQUENCY[frequency], 2)


def limit_for(household_size: int, threshold_pct: int = 60) -> int | None:
    """Look up the frozen income limit for a household size and percent band.

    Returns None when there is no frozen row for that household size (e.g. sizes
    outside the published 1-8 table). We never derive an off-table limit.
    """
    limits = config.load_thresholds()["limits"]
    band = limits.get(str(threshold_pct))
    if band is None:
        return None
    value = band.get(str(household_size))
    return int(value) if value is not None else None


def compare_to_threshold(annual_income: float, threshold: int | None) -> Comparison:
    """Compare income to a frozen threshold. Boundary is inclusive (== is below).

    Returns ``no_frozen_threshold`` when no threshold is available.
    """
    if threshold is None:
        return "no_frozen_threshold"
    if annual_income < 0:
        raise ValueError("Income must be non-negative")
    return "below_or_equal" if annual_income <= threshold else "above"


def summarize_income(
    income_sources: list[IncomeSource],
    household_size: int,
    threshold_pct: int = 60,
) -> ComparisonResult:
    """Annualize and sum independent income sources, then compare to the limit.

    The result is a neutral, fully-cited comparison object. It contains no
    eligibility conclusion of any kind.
    """
    thresholds = config.load_thresholds()
    per_source: list[dict[str, Any]] = []
    total = 0.0
    for src in income_sources:
        annual = annualize(src.amount, src.frequency)
        total += annual
        per_source.append(
            {
                "source_document_id": src.source_document_id,
                "amount": src.amount,
                "frequency": src.frequency,
                "periods_per_year": FREQUENCY[src.frequency],
                "annualized": annual,
                "formula": f"{src.amount} x {FREQUENCY[src.frequency]} = {annual}",
            }
        )
    total = round(total, 2)
    threshold = limit_for(household_size, threshold_pct)
    status: Status = "computed"
    abstain_reason = None
    if not income_sources:
        status = "abstain"
        abstain_reason = "No confirmed income sources yet; nothing to compare."
    return ComparisonResult(
        annualized_income=total,
        household_size=household_size,
        threshold_pct=threshold_pct,
        threshold=threshold,
        comparison=compare_to_threshold(total, threshold),
        area=thresholds["area"],
        rule_year=config.RULE_YEAR,
        effective_date=thresholds["effective_date"],
        source_url=thresholds["source_url"],
        per_source=per_source,
        status=status,
        abstain_reason=abstain_reason,
    )
