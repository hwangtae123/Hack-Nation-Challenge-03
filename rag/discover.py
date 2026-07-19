"""Stretch goal: Discover -- a transparent LIHTC property directory.

Shows the organizer-provided one-metro LIHTC property subset (public HUD
project-location data only) so a renter can see where LIHTC properties are,
alongside the rules that bound what this dataset can and can't tell them.

Hard boundaries (challenge's Discover requirements):
  * Availability is ALWAYS "unknown" here -- a constant, never computed --
    because this dataset is not a vacancy, open-waitlist, or
    application-status feed (rule HUD-DATA-001).
  * The full, unfiltered set is always available; any narrowing is by a
    renter-selected filter only (city, unit-bedroom-count) -- never inferred
    from the renter's documents or any other signal.
  * Order is a fixed, deterministic sort (by project name) -- never a
    relevance/"best match" score, never a ranking by protected traits or
    proxies, and there is no such data in this file to rank by (see
    property_data_dictionary.csv: only project/location/unit-count facts).
  * No acceptance, eligibility, or "good fit" prediction of any kind.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Optional

from rag import config

PROPERTIES_PATH = (
    config.REPO_ROOT
    / "realdoor-hackathon-starter-pack"
    / "realdoor-hackathon-starter-pack"
    / "data"
    / "lihtc_boston_metro_subset.csv"
)
RULES_PATH = config.REPO_ROOT / "rules" / "rule_corpus.jsonl"

# Renter-selectable bedroom-size filter -> the CSV column it checks for > 0 units.
BEDROOM_FIELDS: dict[str, str] = {
    "studio": "studio_units",
    "one": "one_bedroom_units",
    "two": "two_bedroom_units",
    "three": "three_bedroom_units",
    "four": "four_bedroom_units",
}

_INT_FIELDS = (
    "total_units",
    "low_income_units",
    "studio_units",
    "one_bedroom_units",
    "two_bedroom_units",
    "three_bedroom_units",
    "four_bedroom_units",
)

_AVAILABILITY_NOTE = (
    "This is a directory of project locations from HUD's public LIHTC database, "
    "not a live vacancy, open-waitlist, or application-status feed. Availability "
    "is unknown for every listing unless a separate live source is supplied."
)


def _to_int(value: Optional[str]) -> Optional[int]:
    value = (value or "").strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _to_float(value: Optional[str]) -> Optional[float]:
    value = (value or "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


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
def load_properties() -> tuple[dict[str, Any], ...]:
    """Load the full, unfiltered LIHTC property directory.

    Every row is a public project-location fact (name, address, unit counts,
    HUD geocode); there is no household, income, or demographic data in this
    file at all (see data/property_data_dictionary.csv). Each record is
    stamped ``availability: "unknown"`` -- a constant, never computed or
    inferred -- per HUD-DATA-001.
    """
    rows: list[dict[str, Any]] = []
    with PROPERTIES_PATH.open(encoding="utf-8", newline="") as f:
        for raw in csv.DictReader(f):
            row: dict[str, Any] = dict(raw)
            for key in _INT_FIELDS:
                row[key] = _to_int(row.get(key))
            row["latitude"] = _to_float(row.get("latitude"))
            row["longitude"] = _to_float(row.get("longitude"))
            row["availability"] = "unknown"
            rows.append(row)
    # Deterministic, non-predictive order -- alphabetical by name -- never a
    # relevance/acceptance/"best match" ranking, and never by protected traits.
    rows.sort(key=lambda r: (r.get("project_name") or "", r.get("hud_id") or ""))
    return tuple(rows)


def available_cities() -> list[str]:
    """Distinct project cities in the directory, for a renter-facing filter."""
    return sorted({r["project_city"] for r in load_properties() if r.get("project_city")})


@dataclass
class DiscoverResult:
    """A property listing. Carries no acceptance, eligibility, or fit signal."""

    properties: list[dict[str, Any]]
    total_unfiltered: int
    filters_applied: dict[str, Any]
    availability_note: str = _AVAILABILITY_NOTE
    citations: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "properties": self.properties,
            "total_unfiltered": self.total_unfiltered,
            "filters_applied": self.filters_applied,
            "availability_note": self.availability_note,
            "citations": self.citations,
        }


def discover_properties(
    city: Optional[str] = None,
    bedroom_types: Optional[list[str]] = None,
) -> DiscoverResult:
    """Return the property directory, narrowed only by renter-selected filters.

    With no filters, the full unfiltered set is returned -- the challenge
    requires the unfiltered set always be showable, with filtering renter-
    initiated rather than automatic.
    """
    all_rows = load_properties()
    rows = list(all_rows)

    if city:
        city_norm = city.strip().lower()
        rows = [r for r in rows if (r.get("project_city") or "").strip().lower() == city_norm]

    bedroom_types = [b for b in (bedroom_types or []) if b in BEDROOM_FIELDS]
    if bedroom_types:
        wanted_fields = [BEDROOM_FIELDS[b] for b in bedroom_types]
        rows = [r for r in rows if any((r.get(f) or 0) > 0 for f in wanted_fields)]

    return DiscoverResult(
        properties=rows,
        total_unfiltered=len(all_rows),
        filters_applied={"city": city, "bedroom_types": bedroom_types},
        citations=[_cite("HUD-DATA-001"), _cite("HUD-GEO-001")],
    )
