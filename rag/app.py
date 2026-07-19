"""Unified RealDoor demo (Flask): Profile (extraction) + Understand + Prepare.

This wires the Stage 1 extraction pipeline (``src/``) into the RAG demo so a user
can pick or upload a synthetic income document, see the extracted fields with
their source boxes and confidence, confirm them (nothing flows downstream until
confirmed), and then run the neutral income comparison and readiness check.

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
  POST   /api/session/delete  wipe an upload session's files from disk

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
from pathlib import Path
from typing import Any

import pdfplumber
from flask import Flask, abort, jsonify, render_template, request, send_file
from PIL import ImageDraw
from werkzeug.utils import secure_filename

from rag.calculate import IncomeSource, summarize_income
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


def _safe_doc_path(file_name: str, source: str, session_id: str = "") -> Path:
    """Resolve a document path, refusing anything outside the allowed folders."""
    name = secure_filename(file_name)
    base = _session_dir(session_id) if source == "upload" else doc_config.DOCUMENTS_DIR
    path = (base / name).resolve()
    if not str(path).startswith(str(Path(base).resolve())) or not path.exists():
        abort(404, description="Document not found.")
    return path


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
    path = _safe_doc_path(file_name, "dataset")
    return jsonify(_extract_response(path))


@app.post("/api/extract_batch")
def api_extract_batch():
    """Extract several bundled documents at once (e.g. a whole household)."""
    names = (request.get_json(silent=True) or {}).get("file_names", [])
    if not names:
        return jsonify(error="file_names is required."), 400
    documents = [_extract_response(_safe_doc_path(n, "dataset")) for n in names]
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
        dest = session_dir / name
        file.save(str(dest))
        # A per-file type override only applies to a single upload.
        data = _extract_response(dest, document_type=override if len(files) == 1 else None)
        data["source"] = "upload"
        documents.append(data)
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
    return jsonify(deleted=True, session_id=session_id)


@app.get("/api/page.png")
def api_page_png():
    file_name = request.args.get("file", "")
    source = request.args.get("source", "dataset")
    session_id = request.args.get("session_id", "")
    path = _safe_doc_path(file_name, source, session_id)
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
    buf = io.BytesIO()
    img.save(buf, "PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


@app.post("/api/understand")
def api_understand():
    question = (request.get_json(silent=True) or {}).get("question", "").strip()
    if not question:
        return jsonify(error="A question is required."), 400
    try:
        return jsonify(answer_question(question).to_dict())
    except FileNotFoundError:
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
    return jsonify(assessment.to_dict())


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
