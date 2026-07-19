"""Tests for the corpus validation checklist (offline)."""
import pytest

from rag.chunk import chunk_all
from rag.schema import Chunk, base_metadata
from rag.validate import ValidationError, assert_valid, validate_chunks


def _chunk(text: str, sid: str = "usc_42_g", breadcrumb: str = "[ctx]", **over) -> Chunk:
    meta = base_metadata(sid)
    meta.update(over)
    return Chunk(text=text, chunk_index=0, breadcrumb=breadcrumb, source_page=1, **meta)


def test_real_corpus_is_valid():
    assert validate_chunks(chunk_all()) == []


def test_missing_citation_flagged():
    bad = _chunk("Some rule text.", source_url="", citation="")
    problems = validate_chunks([bad] + chunk_all())
    assert any("missing citation" in p for p in problems)


def test_income_limit_number_flagged():
    problems = validate_chunks([_chunk("The 60% limit is $102,840 for a family of four.")])
    assert any("money pattern" in p for p in problems)


def test_verdict_phrase_flagged():
    problems = validate_chunks([_chunk("Based on the review, the applicant is approved.")])
    assert any("verdict" in p for p in problems)


def test_technical_eligible_not_flagged():
    # legitimate rule prose containing "eligible" must NOT be flagged
    ok = _chunk("Changes in the eligible basis of the building are reported.")
    assert not any("verdict" in p for p in validate_chunks([ok]))


def test_other_program_flagged():
    problems = validate_chunks([_chunk("This applies under Section 236 of the Act.")])
    assert any("non-LIHTC" in p for p in problems)


def test_assert_valid_raises_on_bad_corpus():
    with pytest.raises(ValidationError):
        assert_valid([_chunk("Text with $102,840 income limit.")])
