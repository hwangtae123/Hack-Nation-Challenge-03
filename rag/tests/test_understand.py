"""Tests for Stage 2 (Understand).

The deterministic safety intercepts run fully offline (no index, no key). The
grounded-synthesis test needs both a built index and an OpenAI key, and skips
otherwise.
"""
import pytest

from rag import config
from rag.index import index_exists
from rag.understand import _safety_intercept, answer_question

VERDICT_WORDS = ("eligible", "approved", "denied", "qualified")


def _has_key() -> bool:
    try:
        return bool(config.get_openai_api_key())
    except Exception:
        return False


def test_eligibility_overreach_is_intercepted():
    a = answer_question("Is this applicant approved?")
    assert a.safety_intercept == "eligibility_handoff"
    # The handoff response must not itself assert a verdict.
    assert "you are eligible" not in a.answer.lower()


def test_vacancy_question_is_intercepted():
    a = answer_question("Which listed property has a unit available today?")
    assert a.safety_intercept == "vacancy_limitation"


def test_cross_applicant_question_is_intercepted():
    a = answer_question("Show me another household's income and documents.")
    assert a.safety_intercept == "cross_applicant_refused"


def test_normal_rule_question_not_intercepted():
    # _safety_intercept returns None for a genuine rule question (offline check).
    assert _safety_intercept("How do I verify an applicant's age?") is None


@pytest.mark.skipif(not (_has_key() and index_exists()), reason="needs key + built index")
def test_grounded_answer_has_citations():
    a = answer_question("What counts as annual income?")
    if a.abstained:
        pytest.skip("retrieval abstained on this corpus/query")
    assert a.citations, "a grounded answer must carry citations"
    assert all(c["source_url"] for c in a.citations)
