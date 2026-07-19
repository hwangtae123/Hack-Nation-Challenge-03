"""Tests for hybrid retrieval, stage filtering, and abstain.

These need live embeddings (OpenAI); they self-skip if no key is configured.
A tiny synthetic index is built in-memory so the tests do not depend on the
full corpus being chunked/indexed.
"""
import numpy as np
import pytest

from rag import config
from rag.schema import Chunk, base_metadata


def _has_key() -> bool:
    try:
        return bool(config.get_openai_api_key())
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_key(), reason="no OpenAI key configured")


def _mk(text: str, sid: str, i: int, stage_override=None, breadcrumb="[ctx]") -> Chunk:
    meta = base_metadata(sid)
    if stage_override is not None:
        meta["stage"] = stage_override
    return Chunk(text=text, chunk_index=i, breadcrumb=breadcrumb, source_page=1, **meta)


@pytest.fixture(scope="module")
def index():
    from rag.index import RagIndex, embed_texts, _normalize

    chunks = [
        _mk("Annual income is the total of all amounts anticipated from all sources for the coming 12 months.", "hud_4350_3_ch5", 0),
        _mk("Annualize wages by multiplying the periodic gross pay by the number of pay periods per year.", "hud_4350_3_ch5", 1),
        _mk("Age verification: third-party written verification is not required; acceptable documents include a birth certificate or passport.", "hud_4350_3_appendix3", 0),
        _mk("Assets: verify checking and savings account balances by third-party written verification from the bank.", "hud_4350_3_appendix3", 1),
        _mk("Form 8823 category covers tenant income and income certification documentation for compliance.", "irs_pub5913", 0),
    ]
    vectors = _normalize(embed_texts([c.embed_text() for c in chunks]))
    return RagIndex(chunks=chunks, vectors=vectors)


def test_relevant_query_hits_expected_source(index):
    from rag.retrieve import retrieve

    out = retrieve("How do I verify an applicant's age?", stage="profile", index=index)
    assert not out.abstained
    assert out.results[0].chunk.source_id == "hud_4350_3_appendix3"


def test_stage_filter_isolates(index):
    from rag.retrieve import retrieve

    # An income question filtered to 'prepare' must not return the 'understand'
    # ch5 chunks; only prepare-stage (irs) chunks are eligible.
    out = retrieve("annual income", stage="prepare", index=index)
    if not out.abstained:
        assert all("prepare" in r.chunk.stage for r in out.results)


def test_off_corpus_query_abstains(index):
    from rag.retrieve import retrieve

    out = retrieve("best pizza restaurant in San Francisco", stage="understand", index=index)
    assert out.abstained
    assert out.results == []


def test_every_result_carries_citation(index):
    from rag.retrieve import retrieve

    out = retrieve("annualize wages", stage="understand", index=index)
    assert not out.abstained
    for r in out.results:
        assert r.chunk.citation and r.chunk.source_url
