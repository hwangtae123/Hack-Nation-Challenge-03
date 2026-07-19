"""CLI: build the retrieval index end to end.

    parse (cache) -> chunk (per strategy) -> validate (checklist) -> embed + index

Run:  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 python -m rag.build_index
Add --force to re-parse and re-embed from scratch.

Validation is a hard gate: if the corpus fails the checklist, the index is NOT
built and the offending problems are printed.
"""
from __future__ import annotations

import argparse
import logging
import sys

from rag import config
from rag.chunk import chunk_all
from rag.index import build_index
from rag.parse import parse_all
from rag.validate import validate_chunks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the RealDoor RAG index.")
    parser.add_argument("--force", action="store_true", help="re-parse and re-embed from scratch")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parse_all(force=args.force)
    chunks = chunk_all()

    problems = validate_chunks(chunks)
    if problems:
        print(f"VALIDATION FAILED ({len(problems)} problem(s)); index NOT built:")
        for p in problems:
            print(f"  - {p}")
        return 1

    # Per-source counts for a quick sanity readout.
    counts: dict[str, int] = {}
    for c in chunks:
        counts[c.source_id] = counts.get(c.source_id, 0) + 1
    total = len(chunks)
    irs = counts.get("irs_pub5913", 0)
    print("chunks per source:")
    for sid in config.DOCS:
        print(f"  {sid:26} {counts.get(sid, 0):>4}")
    print(f"total={total}  irs_share={irs / total:.1%}")

    build_index(chunks, use_cache=not args.force)
    print(f"index built -> {config.INDEX_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
