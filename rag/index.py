"""Hybrid index: OpenAI dense embeddings + BM25 sparse, persisted to disk.

The index is written to ``rag/index/`` (vectors as ``.npz``, chunk metadata as
``.json``) and per-text embeddings are cached, so a rebuild re-embeds only new
text and a demo never stalls on a live embedding call -- the built index loads
straight from disk. BM25 is rebuilt from the stored chunk texts at load time
(cheap at a few hundred chunks).
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

from rag import config
from rag.schema import Chunk
from rag.tokenize import bm25_tokens

logger = logging.getLogger(__name__)

_VECTORS_PATH = config.INDEX_DIR / "index.npz"
_CHUNKS_PATH = config.INDEX_DIR / "index.json"
_EMBED_CACHE_PATH = config.INDEX_DIR / "embed_cache.json"
_EMBED_BATCH = 100


def _client():
    """Lazily construct an OpenAI client (import kept local so tests can skip)."""
    import openai

    return openai.OpenAI(api_key=config.get_openai_api_key())


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_embed_cache() -> dict[str, list[float]]:
    if _EMBED_CACHE_PATH.exists():
        with _EMBED_CACHE_PATH.open(encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_embed_cache(cache: dict[str, list[float]]) -> None:
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with _EMBED_CACHE_PATH.open("w", encoding="utf-8") as f:
        json.dump(cache, f)


def embed_texts(texts: list[str], use_cache: bool = True) -> np.ndarray:
    """Embed texts with the configured model, reusing cached vectors by hash.

    Returns an (N, EMBED_DIM) float32 array in the input order.
    """
    cache = _load_embed_cache() if use_cache else {}
    missing = [t for t in texts if _text_hash(t) not in cache]
    if missing:
        client = _client()
        for i in range(0, len(missing), _EMBED_BATCH):
            batch = missing[i : i + _EMBED_BATCH]
            resp = client.embeddings.create(model=config.EMBED_MODEL, input=batch)
            for text, item in zip(batch, resp.data):
                cache[_text_hash(text)] = item.embedding
            logger.info("embedded %d/%d new texts", min(i + _EMBED_BATCH, len(missing)), len(missing))
        if use_cache:
            _save_embed_cache(cache)
    return np.array([cache[_text_hash(t)] for t in texts], dtype=np.float32)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize rows so cosine similarity is a plain dot product."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


@dataclass
class RagIndex:
    """In-memory hybrid index over a fixed set of chunks."""

    chunks: list[Chunk]
    vectors: np.ndarray  # (N, EMBED_DIM), L2-normalized

    def __post_init__(self) -> None:
        self._bm25 = BM25Okapi([bm25_tokens(c.embed_text()) for c in self.chunks])

    def bm25_scores(self, query: str) -> np.ndarray:
        return np.asarray(self._bm25.get_scores(bm25_tokens(query)), dtype=np.float32)

    def cosine_scores(self, query_vector: np.ndarray) -> np.ndarray:
        return self.vectors @ _normalize(query_vector.reshape(1, -1))[0]


def build_index(chunks: list[Chunk], use_cache: bool = True) -> RagIndex:
    """Embed chunks and persist the index (vectors .npz + chunks .json)."""
    if not chunks:
        raise ValueError("Cannot build an index from zero chunks")
    vectors = _normalize(embed_texts([c.embed_text() for c in chunks], use_cache=use_cache))
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(_VECTORS_PATH, vectors=vectors)
    with _CHUNKS_PATH.open("w", encoding="utf-8") as f:
        json.dump([c.to_dict() for c in chunks], f, ensure_ascii=False, indent=2)
    logger.info("built index: %d chunks -> %s", len(chunks), config.INDEX_DIR)
    return RagIndex(chunks=chunks, vectors=vectors)


def index_exists() -> bool:
    return _VECTORS_PATH.exists() and _CHUNKS_PATH.exists()


def load_index() -> RagIndex:
    """Load the persisted index from disk (no network)."""
    if not index_exists():
        raise FileNotFoundError("No index on disk; run build_index first.")
    with _CHUNKS_PATH.open(encoding="utf-8") as f:
        chunks = [Chunk.from_dict(d) for d in json.load(f)]
    vectors = np.load(_VECTORS_PATH)["vectors"].astype(np.float32)
    return RagIndex(chunks=chunks, vectors=vectors)


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string (cached like any other text)."""
    return embed_texts([query])[0]
