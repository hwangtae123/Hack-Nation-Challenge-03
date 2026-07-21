# RealDoor — Application-Readiness Copilot

RealDoor helps a renter prepare a LIHTC (Section 42) affordable-housing application:
it extracts data from their documents, explains what the program's rules require with
citations, runs the deterministic income math, and assembles a renter-controlled
readiness packet. It never decides eligibility — that stays with a human reviewer.

> AI extracts, explains, retrieves, calculates, and prepares. The renter confirms.
> A qualified human decides.

**Live demo:** http://34.173.28.107:8000
**Scope (frozen):** LIHTC (Section 42) · Boston-Cambridge-Quincy, MA-NH HMFA · FY2026
(effective 2026-05-01)

## The three-stage flow

| Stage | What it does | Code |
| --- | --- | --- |
| ① Profile | Extracts allowlisted fields from a synthetic pay stub / benefit letter / etc., with a source box and confidence on every value. Nothing downstream ever sees a value until the renter confirms or corrects it. | `src/` |
| ② Understand | Answers rule questions from a versioned, citation-only corpus (HUD Handbook 4350.3, IRS Pub 5913, 26 U.S.C. 42(g), 26 CFR 1.42-5) via hybrid retrieval, and runs deterministic income annualization against the frozen FY2026 MTSP limits. Abstains rather than guessing. | `rag/understand.py`, `rag/calculate.py`, `rag/retrieve.py` |
| ③ Prepare | Flags missing or expired documents against a gold checklist, and assembles a renter-controlled packet (preview, edit, download as JSON/PDF, delete) that is never auto-sent anywhere. | `rag/prepare.py` |

Plus two additions beyond the required build:

- **Discover (stretch goal)** — a transparent LIHTC property directory built from the
  organizer's public HUD data. Availability is always `"unknown"`, filters are
  renter-selected only, and order is a fixed alphabetical sort — never a ranking.
  (`rag/discover.py`)
- **AI review notes (optional)** — a non-authoritative LLM commentary layer on top of
  the already-deterministic Profile/Prepare output ("gross pay doesn't reconcile with
  hourly rate × hours"), never replacing it, and hardened against prompt injection.
  (`rag/copilot_notes.py`)

## Non-negotiables this project holds to

- **No decisioning.** Nothing here ever outputs "eligible," "approved," "denied," or a
  ranking. Deterministic regex safety intercepts refuse "decide for me" requests
  *before* any LLM call runs (`rag/understand.py`).
- **Numbers are a lookup, never retrieval.** Income limits come from
  `rag/boston_mtsp_thresholds_fy2026.json` via `rag/calculate.py` — never a vector
  search result.
- **Untrusted input, everywhere.** Instructions injected inside an uploaded document
  are quarantined as inert data at extraction time (`src/quarantine.py`); the
  Understand and AI-notes system prompts explicitly forbid following instructions
  found in retrieved chunks or renter-entered field values, and every generated note
  is scanned afterward for banned/injection-compliance language before it's shown.
  See `rag/tests/test_app.py` for the HTTP-level refusal / prompt-injection /
  session-deletion test suite (there is no SQL or database anywhere in this codebase,
  so the analogous risk covered instead is path/directory traversal via a file name or
  session id).
- **Privacy by construction.** Uploads are encrypted at rest (`rag/crypto.py`) and
  live only under a session-scoped, renter-deletable folder; nothing is ever used to
  train a model; a consent/action audit log records that something happened and which
  rule version applied — never document content (`rag/audit_log.py`).
- **Accessible.** A keyboard-complete wizard, `label[for]` associations, ARIA live
  regions for errors and status, and focus preserved across step transitions.

## Layout

| path | role |
| --- | --- |
| `src/` | Stage 1: allowlisted field extraction, geometry-based watermark/injection handling, OCR fallback, the renter confirmation gate |
| `rag/` | Stages 2–3, Discover, AI notes, and the Flask app (`rag/app.py`) tying it all together — see `rag/README.md` for the retrieval pipeline in depth |
| `rules/rule_corpus.jsonl` | The frozen, citation-only rule corpus behind Prepare's and Discover's citations |
| `data/`, `realdoor-hackathon-starter-pack/` | Organizer-provided HUD MTSP limits, LIHTC property subset, synthetic documents, and gold/evaluation fixtures |

## Running locally

```bash
pip install -r requirements.txt -r rag/requirements.txt
# .env at repo root: OPENAI_API_KEY=sk-...   (embeddings, chat synthesis, OCR vision, AI notes)

# On Windows, force UTF-8 so the console can print non-ASCII document text:
PYTHONIOENCODING=utf-8 PYTHONUTF8=1 python -m rag.build_index   # builds the rule index once
PYTHONIOENCODING=utf-8 PYTHONUTF8=1 python -m rag.app           # http://127.0.0.1:5000

PYTHONIOENCODING=utf-8 PYTHONUTF8=1 python -m pytest tests rag/tests -q
```

## Why Stage 1 extraction is not a plain PDF text dump

The synthetic fixtures carry three deliberate traps, all handled from geometry (never
from storage order or the manifest's flags):

1. **Diagonal watermark.** A large "DOCUMENT / TRAINING FIXTURE" watermark is
   spatially interleaved *inside* value tokens (`2026-0T6-20`, `$28.50A`). pdfplumber
   merges the tall watermark glyph into the value word, so filtering at the word
   level would drop the value. We filter at the **character** level (drop glyphs
   taller than ~20pt) and re-group survivors into words.
2. **Reverse storage.** `extract_text()` returns scrambled order. We ignore storage
   order entirely and re-assemble words into visual lines by geometry.
3. **Coordinate system.** pdfplumber uses a top-left origin; gold uses
   `pdf_points_bottom_left_origin`. Every emitted box is y-flipped
   (`y = page_height - y_topleft`).

Rasterized/image documents are detected by us (never trusted from the manifest) and
routed to an OCR path (`src/extract_ocr.py`, OpenAI vision) that extracts the same
allowlisted fields from a page image when no text layer exists.

## Deployment

Runs as a systemd-managed gunicorn service on a GCP Compute Engine VM. See the
"Deployment (GCP Compute Engine)" section in `rag/README.md` for the exact build/
transfer/restart commands.
