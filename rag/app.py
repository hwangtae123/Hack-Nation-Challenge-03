"""Unified RealDoor demo (Flask): Profile (extraction) + Understand + Prepare
(+ the optional Discover stretch goal).

This wires the Stage 1 extraction pipeline (``src/``) into the RAG demo so a user
can pick or upload a synthetic income document, see the extracted fields with
their source boxes and confidence, confirm them (nothing flows downstream until
confirmed), and then run the neutral income comparison and readiness check.
Discover (``/api/properties``) is unrelated to a renter's own documents: it is
a read-only, public-data property directory (see ``rag/discover.py``).

Endpoints (all English output, no eligibility verdict anywhere):
  GET    /                    the single-page UI
  GET    /api/documents       list the bundled synthetic documents
  POST   /api/extract         run extraction on a bundled document
  POST   /api/extract_batch   run extraction on several bundled documents
  POST   /api/upload          run extraction on one or more uploaded PDFs
  GET    /api/page.png        page 1 rendered with extracted/quarantined boxes
  POST   /api/understand      cited rule answer (or safe/abstain)
  POST   /api/income          neutral annualized-income vs. threshold comparison
  POST   /api/prepare         deterministic document-readiness + checklist grid
  POST   /api/profile_notes   optional, non-authoritative AI consistency notes on one document
  POST   /api/prepare_notes   optional, non-authoritative AI notes on the checklist grid
  POST   /api/session/delete  wipe an upload session's files from disk
  GET    /api/properties      (stretch: Discover) transparent LIHTC property directory

Data-handling guardrail: nothing in this module ever calls out to a property
management system, landlord, or other third party -- there is no outbound
integration of any kind. Uploaded files and extracted values exist only for
this process's lifetime (or until the renter deletes the session below); the
extracted "packet" the front end assembles lives in the browser only and is
never written to a database here.

Run:  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 python -m rag.app   (from the repo root)
"""
from __future__ import annotations

import io
import re
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pdfplumber
from flask import Flask, abort, jsonify, render_template, request, send_file
from PIL import ImageDraw
from werkzeug.utils import secure_filename

from rag import crypto
from rag.audit_log import log_event
from rag.calculate import IncomeSource, summarize_income
from rag.copilot_notes import prepare_notes, profile_notes
from rag.discover import available_cities, discover_properties
from rag.index import index_exists
from rag.prepare import assess_readiness
from rag.understand import answer_question
from src import allowlist as doc_allowlist
from src import config as doc_config
from src.profile import build_profile

app = Flask(__name__)

UPLOADS_DIR = Path(__file__).resolve().parent / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

# Box colors on the rendered page.
_QUARANTINE_COLOR = (220, 40, 40)
_FIELD_COLOR = (30, 160, 90)

# Uploads are scoped under UPLOADS_DIR/<session_id>/ so a renter can wipe
# everything they uploaded with one call (see /api/session/delete). The id is
# opaque and client-generated (e.g. crypto.randomUUID()); this pattern is just
# a path-safety allowlist, not an auth mechanism.
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _household_id(file_name: str) -> str:
    """hh-001_d03_pay_stub.pdf -> HH-001."""
    return file_name.split("_", 1)[0].upper()


def _session_dir(session_id: str) -> Path:
    if not _SESSION_ID_RE.match(session_id or ""):
        abort(400, description="Invalid session id.")
    path = (UPLOADS_DIR / session_id).resolve()
    if not str(path).startswith(str(UPLOADS_DIR.resolve())):
        abort(400, description="Invalid session id.")
    return path


def _safe_doc_path(file_name: str) -> Path:
    """Resolve a bundled dataset document path, refusing anything outside it.

    Dataset documents are public synthetic fixtures shipped with the repo;
    they are never encrypted (see ``crypto.py`` for why only uploads are).
    """
    name = secure_filename(file_name)
    path = (doc_config.DOCUMENTS_DIR / name).resolve()
    if not str(path).startswith(str(doc_config.DOCUMENTS_DIR.resolve())) or not path.exists():
        abort(404, description="Document not found.")
    return path


def _uploaded_enc_path(file_name: str, session_id: str) -> Path:
    """Resolve an uploaded (encrypted-at-rest) document's on-disk path."""
    name = secure_filename(file_name)
    path = (_session_dir(session_id) / (name + ".enc")).resolve()
    if not str(path).startswith(str(UPLOADS_DIR.resolve())) or not path.exists():
        abort(404, description="Document not found.")
    return path


@contextmanager
def _resolve_for_processing(file_name: str, source: str, session_id: str = "") -> Iterator[Path]:
    """Yield a plaintext path to process, for the duration of the ``with`` block.

    Dataset documents are bundled plaintext fixtures and are opened directly.
    Uploaded documents are encrypted at rest; this decrypts to a short-lived
    temp file that is deleted as soon as the block exits.
    """
    if source == "upload":
        name = secure_filename(file_name)
        enc_path = _uploaded_enc_path(file_name, session_id)
        try:
            with crypto.decrypted_copy(enc_path, name) as plain_path:
                yield plain_path
        except crypto.DecryptionError as exc:
            abort(410, description=str(exc))
    else:
        yield _safe_doc_path(file_name)


def _income_sources(raw: list[dict[str, Any]]) -> list[IncomeSource]:
    return [
        IncomeSource(
            amount=float(s["amount"]),
            frequency=str(s["frequency"]),
            source_document_id=s.get("source_document_id"),
        )
        for s in raw
    ]


@app.get("/")
def home() -> str:
    return render_template("index.html", index_ready=index_exists())


@app.get("/api/documents")
def api_documents():
    docs = []
    for p in sorted(doc_config.DOCUMENTS_DIR.glob("*.pdf")):
        docs.append(
            {
                "file_name": p.name,
                "document_type": doc_allowlist.infer_document_type(p.name),
                "household_id": _household_id(p.name),
            }
        )
    return jsonify(documents=docs)


def _extract_response(path: Path, document_type: str | None = None) -> dict[str, Any]:
    profile = build_profile(path, document_type=document_type)
    data = profile.to_dict()
    data["household_id"] = _household_id(path.name)
    return data


@app.post("/api/extract")
def api_extract():
    file_name = (request.get_json(silent=True) or {}).get("file_name", "")
    if not file_name:
        return jsonify(error="file_name is required."), 400
    path = _safe_doc_path(file_name)
    data = _extract_response(path)
    log_event("document_extract", source="dataset", document_type=data.get("document_type"), file_count=1)
    return jsonify(data)


@app.post("/api/extract_batch")
def api_extract_batch():
    """Extract several bundled documents at once (e.g. a whole household)."""
    names = (request.get_json(silent=True) or {}).get("file_names", [])
    if not names:
        return jsonify(error="file_names is required."), 400
    documents = [_extract_response(_safe_doc_path(n)) for n in names]
    log_event(
        "document_extract",
        source="dataset",
        file_count=len(documents),
        document_types=[d.get("document_type") for d in documents],
    )
    return jsonify(documents=documents)


@app.post("/api/upload")
def api_upload():
    """Accept one or many uploaded PDFs; returns one extraction result each.

    Multi-file uploads use the form field ``files``; a single upload may use
    ``file`` with an optional ``document_type`` override for arbitrary names.
    Requires a client-generated ``session_id`` so the renter can later wipe
    everything they uploaded in one call (see /api/session/delete).
    """
    session_id = request.form.get("session_id", "")
    session_dir = _session_dir(session_id)  # 400s on a missing/invalid id

    files = request.files.getlist("files")
    single = request.files.get("file")
    if not files and single is not None:
        files = [single]
    if not files:
        return jsonify(error="No files uploaded."), 400

    override = request.form.get("document_type") or None
    if override and override not in doc_allowlist.ALLOWLIST:
        return jsonify(error=f"Unknown document_type: {override}"), 400

    session_dir.mkdir(parents=True, exist_ok=True)
    documents: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for file in files:
        if not file.filename:
            continue
        if not file.filename.lower().endswith(".pdf"):
            errors.append({"file": file.filename, "error": "Only PDF documents are supported."})
            continue
        name = secure_filename(file.filename)
        dest = session_dir / (name + ".enc")
        dest.write_bytes(crypto.encrypt_bytes(file.read()))  # encrypted at rest; see crypto.py
        # A per-file type override only applies to a single upload. The decrypted
        # plaintext exists only for this block, in a temp file, then is removed.
        with crypto.decrypted_copy(dest, name) as plain_path:
            data = _extract_response(plain_path, document_type=override if len(files) == 1 else None)
        data["source"] = "upload"
        documents.append(data)
    log_event(
        "document_extract",
        session_id=session_id,
        source="upload",
        file_count=len(documents),
        document_types=[d.get("document_type") for d in documents],
        error_count=len(errors),
    )
    return jsonify(documents=documents, errors=errors)


@app.post("/api/session/delete")
def api_session_delete():
    """Wipe every file uploaded under a session id.

    This is the renter's "delete my data" control: it removes the uploaded
    PDFs from disk. Extracted fields and any assembled packet live only in the
    browser and are never stored server-side, so deleting the upload
    directory is the complete server-side cleanup needed.
    """
    session_id = (request.get_json(silent=True) or {}).get("session_id", "")
    session_dir = _session_dir(session_id)
    if session_dir.exists():
        shutil.rmtree(session_dir)
    log_event("session_delete", session_id=session_id)
    return jsonify(deleted=True, session_id=session_id)


@app.get("/api/page.png")
def api_page_png():
    file_name = request.args.get("file", "")
    source = request.args.get("source", "dataset")
    session_id = request.args.get("session_id", "")
    buf = io.BytesIO()
    with _resolve_for_processing(file_name, source, session_id) as path:
        profile = build_profile(path)
        with pdfplumber.open(str(path)) as pdf:
            page = pdf.pages[0]
            height = page.height
            rendered = page.to_image(resolution=110)
            img = rendered.original.convert("RGB")
            scale = img.width / page.width
            draw = ImageDraw.Draw(img)
            for f in profile.fields:
                if not f.bbox or len(f.bbox) != 4:
                    continue  # OCR-extracted fields have no per-field box to draw
                is_q = f.field == "untrusted_instruction_text"
                color = _QUARANTINE_COLOR if is_q else _FIELD_COLOR
                x0, y0, x1, y1 = f.bbox  # bottom-left origin
                rect = [x0 * scale, (height - y1) * scale, x1 * scale, (height - y0) * scale]
                draw.rectangle(rect, outline=color, width=2)
                label = "INJECTION" if is_q else f.field
                draw.text((rect[0], max(0, rect[1] - 11)), label, fill=color)
        img.save(buf, "PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


@app.post("/api/understand")
def api_understand():
    # The question text itself is never logged (it is renter-entered free
    # text and may describe personal circumstances); only the outcome shape is.
    question = (request.get_json(silent=True) or {}).get("question", "").strip()
    if not question:
        return jsonify(error="A question is required."), 400
    try:
        answer = answer_question(question)
        log_event(
            "understand_query",
            abstained=answer.abstained,
            safety_intercept=answer.safety_intercept,
            citation_count=len(answer.citations),
        )
        return jsonify(answer.to_dict())
    except FileNotFoundError:
        log_event("understand_query", abstained=True, safety_intercept=None, citation_count=0, index_missing=True)
        return jsonify(
            question=question,
            answer="The rule index has not been built yet. Run "
            "`python -m rag.build_index`, then try again.",
            abstained=True,
            safety_intercept=None,
            citations=[],
        )


@app.post("/api/income")
def api_income():
    data = request.get_json(silent=True) or {}
    try:
        result = summarize_income(
            _income_sources(data.get("income_sources", [])),
            household_size=int(data.get("household_size", 1)),
            threshold_pct=int(data.get("threshold_pct", 60)),
        )
    except (KeyError, ValueError) as exc:
        return jsonify(error=f"Invalid input: {exc}"), 400
    return jsonify(result.to_dict())


@app.post("/api/prepare")
def api_prepare():
    data = request.get_json(silent=True) or {}
    try:
        assessment = assess_readiness(
            household_id=str(data.get("household_id", "HH")),
            household_size=int(data.get("household_size", 1)),
            required_document_types=list(data.get("required_document_types", [])),
            present_document_types=list(data.get("present_document_types", [])),
            income_sources=_income_sources(data.get("income_sources", [])),
            document_dates=data.get("document_dates") or None,
            known_conflicts=data.get("known_conflicts") or None,
        )
    except (KeyError, ValueError) as exc:
        return jsonify(error=f"Invalid input: {exc}"), 400
    log_event(
        "prepare_run",
        household_id=assessment.household_id,
        readiness_status=assessment.readiness_status,
        flag_codes=[f.code for f in assessment.flags],
    )
    return jsonify(assessment.to_dict())


@app.post("/api/profile_notes")
def api_profile_notes():
    """Optional AI commentary on one document's extracted fields -- never
    authoritative, never an eligibility signal. See rag/copilot_notes.py.
    """
    data = request.get_json(silent=True) or {}
    document_type = str(data.get("document_type") or "")
    fields = list(data.get("fields") or [])
    try:
        result = profile_notes(document_type, fields)
    except Exception as exc:  # LLM call is best-effort; never break the Profile step over it.
        result = None
        error = str(exc)
    else:
        error = None
    log_event("profile_notes", document_type=document_type, abstained=(result.abstained if result else True))
    if result is None:
        return jsonify(notes="", abstained=True, abstain_reason=f"Could not generate notes: {error}")
    return jsonify(result.to_dict())


@app.post("/api/prepare_notes")
def api_prepare_notes():
    """Optional AI commentary on the document checklist grid -- structurally
    excludes income/threshold/AMI data (see prepare_notes's signature).
    """
    data = request.get_json(silent=True) or {}
    document_status = list(data.get("document_status") or [])
    flags = list(data.get("flags") or [])
    try:
        result = prepare_notes(document_status, flags)
    except Exception as exc:  # LLM call is best-effort; never break the Prepare step over it.
        result = None
        error = str(exc)
    else:
        error = None
    log_event(
        "prepare_notes",
        household_id=data.get("household_id"),
        abstained=(result.abstained if result else True),
    )
    if result is None:
        return jsonify(notes="", abstained=True, abstain_reason=f"Could not generate notes: {error}")
    return jsonify(result.to_dict())


@app.get("/api/properties")
def api_properties():
    """Stretch goal: Discover. Renter-selected filters only; the unfiltered
    set is always available (no filters = everything); availability is always
    "unknown"; order is a fixed alphabetical sort, never a ranking or a
    prediction of acceptance/fit. See rag/discover.py for the boundaries.
    """
    city = request.args.get("city") or None
    bedroom_types = [b for b in request.args.getlist("bedroom_type") if b]
    result = discover_properties(city=city, bedroom_types=bedroom_types)
    log_event(
        "discover_query",
        city=city,
        bedroom_types=bedroom_types,
        result_count=len(result.properties),
    )
    return jsonify(result.to_dict())


@app.get("/api/properties/cities")
def api_property_cities():
    return jsonify(cities=available_cities())


if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
