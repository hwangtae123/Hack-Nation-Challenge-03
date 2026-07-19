"""Hybrid retrieval: hard filter -> dense+sparse fusion -> rerank -> abstain.

Pipeline (matches rag/claude.md):
  1. Hard filter to program=LIHTC AND rule_year=FY2026 AND (stage contains the
     current stage). A rule for the wrong program or stage never competes.
  2. Hybrid scoring over the surviving chunks: dense cosine (OpenAI embeddings)
     plus sparse BM25, min-max normalized and fused into a candidate pool.
  3. Rerank the pool (cosine-weighted) and keep the top few.
  4. Abstain: if the best chunk's raw cosine similarity is below the threshold,
     return nothing with a reason -- never hand a weak match to the answerer.

Why hybrid: legal queries use exact terms ("annualization", "third-party
verification", "Form 8823") where pure vector search is weak; BM25 covers that.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from rag import config
from rag.index import RagIndex, embed_query, load_index
from rag.schema import RetrievalResult

# Rerank weighting: lean on semantic cosine, keep sparse term-overlap as support.
_RERANK_W_DENSE = 0.6
_RERANK_W_SPARSE = 0.4


@dataclass
class RetrievalOutcome:
    """Result of a retrieval attempt, including the abstain decision."""

    query: str
    stage: Optional[str]
    abstained: bool
    top_score: float  # best raw cosine similarity (the abstain signal)
    results: list[RetrievalResult] = field(default_factory=list)
    reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "stage": self.stage,
            "abstained": self.abstained,
            "top_score": round(self.top_score, 4),
            "reason": self.reason,
            "results": [r.to_dict() for r in self.results],
        }


def _minmax(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]; flat inputs map to all-ones."""
    lo, hi = float(scores.min()), float(scores.max())
    if hi - lo < 1e-12:
        return np.ones_like(scores)
    return (scores - lo) / (hi - lo)


def _eligible_indices(index: RagIndex, stage: Optional[str]) -> list[int]:
    keep: list[int] = []
    for i, c in enumerate(index.chunks):
        if c.program != config.PROGRAM or c.rule_year != config.RULE_YEAR:
            continue
        if stage is not None and stage not in c.stage:
            continue
        keep.append(i)
    return keep


def retrieve(
    query: str,
    stage: Optional[str] = None,
    index: Optional[RagIndex] = None,
    candidate_k: int = config.RETRIEVE_TOP_K,
    top_k: int = config.RERANK_TOP_K,
    abstain_score: float = config.ABSTAIN_SCORE,
) -> RetrievalOutcome:
    """Retrieve reranked chunks for a query, or abstain if nothing is relevant."""
    index = index or load_index()
    eligible = _eligible_indices(index, stage)
    if not eligible:
        return RetrievalOutcome(query, stage, abstained=True, top_score=0.0,
                                reason=f"No indexed rules for stage={stage!r}.")

    eligible_arr = np.array(eligible)
    qvec = embed_query(query)
    cos = index.cosine_scores(qvec)[eligible_arr]           # raw cosine (abstain signal)
    bm25 = index.bm25_scores(query)[eligible_arr]
    fused = 0.5 * _minmax(cos) + 0.5 * _minmax(bm25)

    # Candidate pool by fused score, then rerank (cosine-weighted) within it.
    pool_local = np.argsort(-fused)[:candidate_k]
    rerank = _RERANK_W_DENSE * _minmax(cos[pool_local]) + _RERANK_W_SPARSE * _minmax(bm25[pool_local])
    order = pool_local[np.argsort(-rerank)][:top_k]

    top_cos = float(cos[order[0]]) if len(order) else 0.0
    if top_cos < abstain_score:
        return RetrievalOutcome(query, stage, abstained=True, top_score=top_cos,
                                reason="Best match below relevance threshold; no confident rule found.")

    results = [
        RetrievalResult(chunk=index.chunks[eligible[int(li)]], score=float(cos[int(li)]))
        for li in order
    ]
    return RetrievalOutcome(query, stage, abstained=False, top_score=top_cos, results=results)
