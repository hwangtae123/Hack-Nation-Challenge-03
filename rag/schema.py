"""Shared data contracts for the RAG pipeline.

Pinned here so the chunking, indexing, and retrieval layers agree on one shape.
Every chunk carries its full citation metadata (a chunk with no citation /
source_url is forbidden by the validation checklist), the stage(s) it serves
(so retrieval can hard-filter by the current stage), and a page number for
precise citations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from rag import config


@dataclass
class Chunk:
    """One retrievable unit of rule prose with its citation metadata.

    Numbers (income limits) must never live in ``text`` -- those are a lookup,
    enforced by the validation checklist. ``breadcrumb`` is prepended to the
    embedded text so a chunk read in isolation still carries its context.
    """

    text: str
    source_id: str
    chunk_index: int
    # doc-level metadata (copied from config.DOCS via base_metadata)
    program: str
    rule_year: str
    doc_type: str
    citation: str
    source_url: str
    authority: str
    effective_date: Optional[str]
    stage: list[str]
    strategy: str
    # chunk-level metadata
    breadcrumb: str = ""
    source_page: Optional[int] = None
    form_8823_category: Optional[str] = None

    @property
    def chunk_id(self) -> str:
        return f"{self.source_id}#{self.chunk_index}"

    def embed_text(self) -> str:
        """Text sent to the embedder: breadcrumb-prefixed for standalone context."""
        return f"{self.breadcrumb}\n{self.text}".strip() if self.breadcrumb else self.text

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source_id": self.source_id,
            "chunk_index": self.chunk_index,
            "program": self.program,
            "rule_year": self.rule_year,
            "doc_type": self.doc_type,
            "citation": self.citation,
            "source_url": self.source_url,
            "authority": self.authority,
            "effective_date": self.effective_date,
            "stage": list(self.stage),
            "strategy": self.strategy,
            "breadcrumb": self.breadcrumb,
            "source_page": self.source_page,
            "form_8823_category": self.form_8823_category,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Chunk":
        return cls(
            text=d["text"],
            source_id=d["source_id"],
            chunk_index=d["chunk_index"],
            program=d["program"],
            rule_year=d["rule_year"],
            doc_type=d["doc_type"],
            citation=d["citation"],
            source_url=d["source_url"],
            authority=d["authority"],
            effective_date=d.get("effective_date"),
            stage=list(d.get("stage", [])),
            strategy=d["strategy"],
            breadcrumb=d.get("breadcrumb", ""),
            source_page=d.get("source_page"),
            form_8823_category=d.get("form_8823_category"),
        )


def base_metadata(source_id: str) -> dict[str, Any]:
    """Doc-level metadata for a source, pulled from the config.DOCS SSOT.

    Chunkers merge this with per-chunk fields (index, breadcrumb, page, etc.)
    so citation metadata always traces back to one authoritative registry.
    """
    meta = config.DOCS[source_id]
    return {
        "source_id": source_id,
        "program": config.PROGRAM,
        "rule_year": config.RULE_YEAR,
        "doc_type": meta["doc_type"],
        "citation": meta["citation"],
        "source_url": meta["source_url"],
        "authority": meta["authority"],
        "effective_date": meta["effective_date"],
        "stage": list(meta["stage"]),
        "strategy": meta["strategy"],
    }


@dataclass
class RetrievalResult:
    """A retrieved chunk with its final (reranked) score."""

    chunk: Chunk
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {"score": self.score, **self.chunk.to_dict()}
