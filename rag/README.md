# RealDoor RAG — Rule Retrieval (Stages 2–3)

The rule-retrieval layer of the RealDoor LIHTC copilot. It **understands** (cites
rule prose + runs deterministic income math) and helps **prepare** (flags missing
or stale documents). It never decides eligibility — income and limits are shown
side by side for a **human** reviewer.

**Scope (frozen):** LIHTC (Section 42) · Boston-Cambridge-Quincy, MA-NH HMFA ·
FY2026 (effective 2026-05-01).

## Absolute rules (see `claude.md`)

1. **Numbers are a lookup, not retrieval.** Income limits come from
   `boston_mtsp_thresholds_fy2026.json` via `calculate.py`, never from a chunk.
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

## Privacy & safety controls (the app, `app.py` + `templates/`)

- **Session-scoped uploads.** Each browser session gets an opaque client-generated
  id; uploaded PDFs live under `uploads/<session_id>/` and nowhere else.
  `POST /api/session/delete` wipes that directory on request (the "Delete this
  session" button in the Prepare step).
- **Uploads are encrypted at rest.** `crypto.py` encrypts an uploaded PDF's
  bytes (Fernet/AES) before it is ever written to `uploads/<session_id>/`.
  Plaintext exists only in a short-lived temp file for the duration of a single
  extraction or page-render request, then is deleted. Bundled demo documents
  (public synthetic fixtures already in the repo) are not encrypted -- there is
  nothing to protect by re-encrypting already-public content. Set
  `UPLOAD_ENCRYPTION_KEY` (a `Fernet.generate_key()` value) as an environment
  variable in production so encrypted uploads stay readable across restarts;
  without it a key is generated per-process, which just means an in-flight
  upload becomes unreadable after a restart -- the same practical effect as
  the renter deleting the session.
- **Never train on uploads.** Uploaded documents and extracted values are used
  only to answer the current session's requests; they are never used to train
  or fine-tune any model.
- **No server-side packet storage.** The assembled packet (confirmed fields +
  checklist + citations) is built and held in the browser only; downloading or
  printing it happens client-side, with no server round-trip.
- **Consent/action audit log, not a content log.** `audit_log.py` appends one
  JSON line per action (document extracted, prepare run, rule question asked,
  session deleted) plus the rule year/effective date in force at the time —
  never the uploaded document text, extracted field values, or the renter's
  question text. See `uploads/audit.jsonl` (git-ignored).

## Optional AI review notes (`copilot_notes.py`)

`POST /api/profile_notes` (one document's extracted fields) and
`POST /api/prepare_notes` (the checklist grid + flags) ask an LLM for a short,
plain-language completeness/consistency note -- "this gross pay doesn't
reconcile with hourly rate x hours", "no benefit letter attached" -- layered
*on top of* the already-deterministic Profile/Prepare output, never replacing
it. These are optional (a button, not automatic) and always rendered as a
clearly secondary "AI suggestion -- not a decision" panel.

Two things keep this from becoming a decisioning surface:
- `prepare_notes`'s signature has no `income_comparison`/threshold/AMI
  parameter at all -- it structurally cannot leak that data because it is
  never given it.
- Every generated note is scanned for eligibility/threshold/AMI/percentage
  language after the fact (`copilot_notes._to_review_notes`); if anything
  slips through, the note is discarded and the caller sees an abstain
  message instead of the tainted text -- the same fail-closed pattern as
  `understand.py`'s retrieval abstention. See `rag/tests/test_copilot_notes.py`.

## Discover (stretch goal, `discover.py`)

`GET /api/properties` serves the organizer-provided one-metro LIHTC property
subset (public HUD project-location data; no household/income/demographic
fields exist in that file at all). The full, unfiltered set is always what
you get with no query params; `city` and `bedroom_type` (repeatable) are
renter-selected filters only, never inferred. Every listing is stamped
`"availability": "unknown"` (a constant, never computed) and cites
`HUD-DATA-001` / `HUD-GEO-001`. Order is a fixed alphabetical sort by project
name -- never a relevance/acceptance ranking.

## Deployment (GCP Compute Engine)

The demo runs as a systemd-managed gunicorn service on a single e2-micro VM
(no GitHub involved in the deploy path, so the organizer starter pack never
has to be pushed to a public repo):

```bash
# from the repo root, package everything the app needs at runtime (excludes
# .git, __pycache__, rag/uploads, .env, and corpus/raw -- the source PDFs are
# only needed to rebuild the index, not to serve it)
tar --exclude='./.git' --exclude='__pycache__' --exclude='.pytest_cache' \
    --exclude='./rag/uploads' --exclude='./.env' --exclude='./rag/corpus/raw' \
    -czf /tmp/realdoor-deploy.tar.gz .

gcloud compute scp /tmp/realdoor-deploy.tar.gz realdoor-demo:realdoor-deploy.tar.gz --zone=us-central1-a
gcloud compute ssh realdoor-demo --zone=us-central1-a --command="
  mkdir -p ~/app && tar -xzf ~/realdoor-deploy.tar.gz -C ~/app
  cd ~/app && python3 -m venv venv && source venv/bin/activate
  pip install -r requirements.txt -r rag/requirements.txt
  sudo systemctl restart realdoor.service
"
```

`/etc/realdoor.env` on the VM holds `OPENAI_API_KEY` and a persistent
`UPLOAD_ENCRYPTION_KEY` (systemd `EnvironmentFile=`); `/etc/systemd/system/realdoor.service`
runs `gunicorn --bind 0.0.0.0:8000 rag.app:app` and restarts on failure or
reboot (`systemctl enable`d). A firewall rule (`realdoor-demo-http`) opens
`tcp:8000` from `0.0.0.0/0`. The pre-built `rag/index/` + `rag/corpus/cache/`
were shipped as part of the tarball, so no OpenAI-dependent build step runs on
the server itself.
