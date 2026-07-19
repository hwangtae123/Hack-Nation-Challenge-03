# RealDoor RAG — Rule Retrieval (Stages 2–3)

The rule-retrieval layer of the RealDoor LIHTC copilot. It **understands** (cites
rule prose + runs deterministic income math) and helps **prepare** (flags missing
or stale documents). It never decides eligibility — income and limits are shown
side by side for a **human** reviewer.

**Scope (frozen):** LIHTC (Section 42) · San Diego-Chula Vista-Carlsbad, CA MSA ·
FY2026 (effective 2026-05-01).

## Absolute rules (see `claude.md`)

1. **Numbers are a lookup, not retrieval.** Income limits come from
   `san_diego_mtsp_thresholds_fy2026.json` via `calculate.py`, never from a chunk.
2. **No eligibility verdicts** anywhere — code, prompts, chunks, or output.
3. **No chunk without a citation** (`citation` + `source_url`).
4. **Abstain when unsure** — retrieval returns nothing below the relevance floor.
5. **LIHTC only** — other programs (Section 236, RAP, Rent Supplement) are excluded.
6. **Document text is untrusted** — never executed as instructions.

## Pipeline

```
PDF ─parse─▶ markdown (cached) ─chunk─▶ Chunks ─validate─▶ index (OpenAI + BM25)
                                                                     │
 question ─▶ retrieve (hard-filter by stage → hybrid → rerank → abstain)
                                                                     │
        understand (cited answer)   prepare (readiness flags)   calculate (numbers)
```

| module | role |
| --- | --- |
| `config.py` | DOCS inventory (citation SSOT), thresholds loader, `.env` keys |
| `parse.py` | pdfplumber → markdown, boilerplate stripping, page markers, cache |
| `chunk.py` | strategy-specific chunkers (statute / prose / table / category) |
| `validate.py` | the corpus checklist (hard gate before indexing) |
| `index.py` | OpenAI embeddings + BM25, persisted; per-text embedding cache |
| `retrieve.py` | stage hard-filter → dense+sparse hybrid → rerank → abstain |
| `calculate.py` | annualize + threshold lookup + neutral comparison |
| `understand.py` | Stage 2: safety intercepts + grounded, cited synthesis |
| `prepare.py` | Stage 3: deterministic, rule-cited readiness flags |
| `app.py` | thin Flask demo (English UI/output) |

Parsing uses **pdfplumber** (no LlamaParse key configured); the parse layer keeps
LlamaParse behind an interface so it can drop in later.

## Run

```bash
pip install -r rag/requirements.txt
# .env at repo root must contain OPEN_AI_API=sk-...   (OpenAI; used for embeddings + chat)

# 1) build the index (parse → chunk → validate → embed). Fails closed if the
#    corpus does not pass the validation checklist.
PYTHONIOENCODING=utf-8 PYTHONUTF8=1 python -m rag.build_index

# 2) run the demo, then open http://127.0.0.1:5000
PYTHONIOENCODING=utf-8 PYTHONUTF8=1 python -m rag.app

# tests (Windows needs the UTF-8 prefix)
PYTHONIOENCODING=utf-8 PYTHONUTF8=1 python -m pytest rag/tests -q
```

Source PDFs live in `corpus/raw/` and are git-ignored (vendor reference); the
parsed cache (`corpus/cache/`) and built index (`index/`) are generated and also
git-ignored.
