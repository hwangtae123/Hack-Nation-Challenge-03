"""Configuration and single-source-of-truth registries for the RealDoor RAG.

Three responsibilities live here, and nothing else reaches out to the network:
  * ``DOCS`` - the document inventory. Because PDFs cannot carry front matter,
    this dict is the sole source of every source document's citation metadata.
    Adding a new rule document means adding a ``DOCS`` entry; no code changes.
  * ``load_thresholds`` - loads the San Diego FY2026 income-limit LOOKUP table.
    Income numbers never come from retrieval; they come from this JSON only.
  * key loading - reads API keys from ``.env`` (never hardcoded, never logged).
    The repo's ``.env`` stores the OpenAI key under the non-standard name
    ``OPEN_AI_API``; we map it to the conventional name here.

Scope is frozen: LIHTC / San Diego-Chula Vista-Carlsbad, CA MSA / FY2026.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from dotenv import dotenv_values

# --- paths -----------------------------------------------------------------
RAG_DIR = Path(__file__).resolve().parent
REPO_ROOT = RAG_DIR.parent
CORPUS_RAW = RAG_DIR / "corpus" / "raw"
CORPUS_CACHE = RAG_DIR / "corpus" / "cache"
INDEX_DIR = RAG_DIR / "index"
THRESHOLDS_PATH = RAG_DIR / "san_diego_mtsp_thresholds_fy2026.json"
ENV_PATH = REPO_ROOT / ".env"

# --- frozen scope ----------------------------------------------------------
PROGRAM = "LIHTC"
RULE_YEAR = "FY2026"
AREA = "San Diego-Chula Vista-Carlsbad, CA MSA (San Diego County)"
EFFECTIVE_DATE = "2026-05-01"

# --- model / retrieval knobs ----------------------------------------------
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
CHUNK_TOTAL_MIN = 150
CHUNK_TOTAL_MAX = 300
RETRIEVE_TOP_K = 20          # hybrid candidate pool
RERANK_TOP_K = 5             # returned to the answerer
# Best raw cosine (text-embedding-3-small) below this -> abstain. Calibrated
# empirically: on-topic queries score ~0.50-0.65, off-topic ~0.00-0.10, so 0.35
# separates them cleanly. (claude.md suggests 0.5; that over-abstains at this
# embedding scale, where a valid query measured 0.516.)
ABSTAIN_SCORE = 0.35
IRS_MAX_CHUNK_SHARE = 0.30  # irs_pub5913 must not exceed this share of chunks

# Canonical HUD 4350.3 handbook landing page. Exhibit 5-1 has a known direct
# URL; the other 4350.3 documents fall back to the handbook page.
# TODO: replace the handbook-page fallbacks with each document's direct URL.
_HUD_4350_3 = "https://www.hud.gov/program_offices/administration/hudclips/handbooks/hsgh/4350.3"

# --- document inventory (SSOT for citation metadata) -----------------------
# Each entry: file name under corpus/raw, citation string, source_url, authority,
# doc_type, stage(s) the doc serves, chunking strategy, effective_date, and an
# optional target_pages excerpt window (used for the 215-page IRS pub).
DOCS: dict[str, dict[str, Any]] = {
    "hud_4350_3_exhibit5_1": {
        "file": "HUD 4350.3 Exhibit 5-1 — Income Inclusions and Exclusions.pdf",
        "citation": "HUD Handbook 4350.3, Exhibit 5-1",
        "source_url": "https://www.hud.gov/sites/documents/doc_35699.pdf",
        "authority": "official_hud",
        "doc_type": "appendix",
        "stage": ["profile"],
        "strategy": "table",
        "effective_date": None,
        "target_pages": None,
    },
    "hud_4350_3_appendix3": {
        "file": "HUD 4350.3 Appendix 3 — Acceptable Forms of Verification.pdf",
        "citation": "HUD Handbook 4350.3, Appendix 3",
        "source_url": _HUD_4350_3,
        "authority": "official_hud",
        "doc_type": "appendix",
        "stage": ["profile"],
        "strategy": "table",
        "effective_date": None,
        "target_pages": None,
    },
    "irs_pub5913": {
        "file": "p5913.pdf",
        "citation": "IRS Publication 5913 (Form 8823 Guide)",
        "source_url": "https://www.irs.gov/pub/irs-pdf/p5913.pdf",
        "authority": "official_irs",
        "doc_type": "irs_guide",
        "stage": ["prepare"],
        "strategy": "category",
        "effective_date": None,
        # 215 pages; must be excerpted to tenant income / certification categories.
        # TODO: pin the exact page window once the Form 8823 sections are located.
        "target_pages": None,
    },
    "usc_42_g": {
        "file": "usc_42_g.md",
        "citation": "26 U.S.C. 42(g)",
        "source_url": "https://uscode.house.gov/view.xhtml?req=granuleid:USC-prelim-title26-section42&num=0&edition=prelim",
        "authority": "official_federal",
        "doc_type": "statute",
        "stage": ["understand"],
        "strategy": "statute",
        "effective_date": None,
        "target_pages": None,
    },
    "hud_4350_3_ch5": {
        "file": "DETERMINING INCOME AND CALCULATING RENT.pdf",
        "citation": "HUD Handbook 4350.3, Chapter 5",
        "source_url": _HUD_4350_3,
        "authority": "official_hud",
        "doc_type": "handbook",
        "stage": ["understand", "profile"],
        "strategy": "prose",
        "effective_date": None,
        "target_pages": None,
    },
    "hud_4350_3_exhibit4_1": {
        "file": "43503e4-1hsgh.pdf",
        "citation": "HUD Handbook 4350.3, Exhibit 4-1",
        "source_url": _HUD_4350_3,
        "authority": "official_hud",
        "doc_type": "appendix",
        "stage": ["prepare"],
        "strategy": "table",
        "effective_date": None,
        "target_pages": None,
    },
    "cfr_1_42_5": {
        "file": "26 CFR §1.42-5 (컴플라이언스 모니터링).pdf",
        "citation": "26 CFR 1.42-5",
        "source_url": "https://www.ecfr.gov/current/title-26/section-1.42-5",
        "authority": "official_federal",
        "doc_type": "regulation",
        "stage": ["prepare"],
        "strategy": "statute",
        "effective_date": None,
        "target_pages": None,
    },
}


def doc_path(source_id: str) -> Path:
    """Absolute path to a source document under corpus/raw."""
    return CORPUS_RAW / DOCS[source_id]["file"]


def cache_path(source_id: str) -> Path:
    """Absolute path to a source document's cached parsed markdown."""
    return CORPUS_CACHE / f"{source_id}.md"


# --- thresholds lookup -----------------------------------------------------
@lru_cache(maxsize=1)
def load_thresholds() -> dict[str, Any]:
    """Load the San Diego FY2026 income-limit lookup table (cached)."""
    with THRESHOLDS_PATH.open(encoding="utf-8") as f:
        return json.load(f)


# --- API keys (never hardcoded / logged) -----------------------------------
def _env() -> dict[str, str | None]:
    """Read .env values (falls back to an empty mapping if absent)."""
    return dotenv_values(ENV_PATH) if ENV_PATH.exists() else {}


def get_openai_api_key() -> str:
    """Return the OpenAI API key from .env (stored as OPEN_AI_API)."""
    env = _env()
    key = env.get("OPENAI_API_KEY") or env.get("OPEN_AI_API")
    if not key:
        raise RuntimeError(
            "OpenAI API key not found. Set OPEN_AI_API (or OPENAI_API_KEY) in .env."
        )
    return key


def get_llama_api_key() -> Optional[str]:
    """Return the LlamaParse key if present, else None (pdfplumber fallback)."""
    return _env().get("LLAMA_CLOUD_API_KEY") or None
