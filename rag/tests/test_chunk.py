"""Tests for the chunkers (offline: uses the already-parsed cache)."""
import re

import pytest

from rag import config
from rag.chunk import chunk_all, chunk_document

_MONEY_RE = re.compile(r"\$\s?\d{2,3},\d{3}")


@pytest.fixture(scope="module")
def chunks():
    return chunk_all()


def test_total_in_range(chunks):
    assert config.CHUNK_TOTAL_MIN <= len(chunks) <= config.CHUNK_TOTAL_MAX


def test_every_strategy_present(chunks):
    by_source = {c.source_id for c in chunks}
    # every registered document contributed at least one chunk
    assert by_source == set(config.DOCS)


def test_statute_breadcrumbs(chunks):
    usc = [c for c in chunks if c.source_id == "usc_42_g"]
    assert usc
    assert all("42(g)" in c.breadcrumb for c in usc)


def test_exhibit5_1_has_inclusion_items(chunks):
    ex = [c for c in chunks if c.source_id == "hud_4350_3_exhibit5_1"]
    assert ex
    assert any("Income Inclusions" in c.breadcrumb for c in ex)


def test_no_money_pattern(chunks):
    assert not [c.chunk_id for c in chunks if _MONEY_RE.search(c.text)]


def test_metadata_complete(chunks):
    for c in chunks:
        assert c.citation and c.source_url and c.breadcrumb
        assert c.stage and isinstance(c.stage, list)
        assert c.program == "LIHTC" and c.rule_year == "FY2026"


def test_irs_share_capped(chunks):
    irs = sum(1 for c in chunks if c.source_id == "irs_pub5913")
    assert irs / len(chunks) <= config.IRS_MAX_CHUNK_SHARE


def test_source_page_set_for_pdf(chunks):
    # multi-page PDFs should attribute a page to their chunks
    ch5 = [c for c in chunks if c.source_id == "hud_4350_3_ch5"]
    assert ch5 and all(c.source_page is not None for c in ch5)


def test_irs_categories_are_income_related(chunks):
    cats = {c.form_8823_category for c in chunks if c.source_id == "irs_pub5913" and c.form_8823_category}
    # only income/certification-related categories should survive filtering
    assert cats <= {"11a", "11c", "11d", "11i"}
