"""Tests for the Discover stretch goal (offline: no network, no key)."""
from rag.discover import discover_properties, load_properties

BANNED = ("eligible", "ineligible", "approved", "denied", "match", "recommend", "best fit", "score")


def test_unfiltered_set_is_the_full_directory():
    all_rows = load_properties()
    result = discover_properties()
    assert len(result.properties) == len(all_rows)
    assert result.total_unfiltered == len(all_rows)
    assert len(all_rows) > 0


def test_availability_is_always_unknown():
    result = discover_properties()
    assert all(p["availability"] == "unknown" for p in result.properties)


def test_city_filter_is_renter_selected_and_narrows_the_set():
    all_rows = load_properties()
    some_city = all_rows[0]["project_city"]
    result = discover_properties(city=some_city)
    assert 0 < len(result.properties) <= len(all_rows)
    assert all(p["project_city"] == some_city for p in result.properties)
    assert result.total_unfiltered == len(all_rows)  # unfiltered count still reported


def test_bedroom_filter_only_returns_matching_properties():
    result = discover_properties(bedroom_types=["studio"])
    assert result.properties  # the fixture data has at least one studio-unit property
    assert all((p.get("studio_units") or 0) > 0 for p in result.properties)


def test_order_is_deterministic_alphabetical_not_a_ranking():
    result = discover_properties()
    names = [p["project_name"] for p in result.properties]
    assert names == sorted(names)


def test_citations_present_and_no_verdict_or_ranking_language():
    result = discover_properties()
    assert result.citations
    ids = [c["rule_id"] for c in result.citations]
    assert "HUD-DATA-001" in ids and "HUD-GEO-001" in ids
    for c in result.citations:
        assert c["source_url"]
    blob = (result.availability_note + " ".join(p["project_name"] for p in result.properties)).lower()
    for w in BANNED:
        assert w not in blob, f"banned word leaked into Discover output: {w}"


def test_no_demographic_or_income_fields_in_a_property_row():
    row = load_properties()[0]
    banned_keys = {"income", "race", "ethnicity", "disability", "immigration", "household"}
    assert not (set(row.keys()) & banned_keys)
