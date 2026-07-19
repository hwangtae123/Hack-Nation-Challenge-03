"""Filesystem locations for the challenge data.

Paths are resolved relative to the repository root so the pipeline works no
matter what the current working directory is. Nothing here reads gold values
into the extraction path -- gold is only ever consumed by the test suite as a
scoring reference.
"""
from __future__ import annotations

from pathlib import Path

# repo_root/src/config.py -> repo_root
REPO_ROOT = Path(__file__).resolve().parent.parent

# The starter pack unzips into a doubly-nested directory of the same name.
_PACK = REPO_ROOT / "realdoor-hackathon-starter-pack" / "realdoor-hackathon-starter-pack"

DOCUMENTS_DIR = _PACK / "synthetic_documents" / "documents"
GOLD_PATH = _PACK / "synthetic_documents" / "gold" / "document_gold.jsonl"
MANIFEST_PATH = _PACK / "synthetic_documents" / "gold" / "document_manifest.csv"
FIELD_SCHEMA_PATH = _PACK / "synthetic_documents" / "gold" / "field_schema.json"


def document_path(file_name: str) -> Path:
    """Return the absolute path to a document by its file name."""
    return DOCUMENTS_DIR / file_name
