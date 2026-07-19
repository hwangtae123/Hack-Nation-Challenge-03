"""Assemble a single document into an unconfirmed extraction profile.

This is the hand-off boundary of Stage 1. It ties the pieces together:
    detect mode -> text path: extract allowlisted fields + quarantine injections
                -> image path: OCR the allowlisted fields via vision
                -> load everything into a renter ConfirmationGate, LOCKED.

Hard rule enforced here: a value whose ``confirmed`` flag is False must never
reach any downstream calculation or aggregation. ``DocumentProfile.to_downstream``
is the only exit, and it delegates to the gate, which raises while anything is
still pending. Quarantined instruction text is never releasable at all. This
module makes no eligibility, approval, or scoring decision.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field as dc_field
from pathlib import Path

from src import allowlist, detect, extract_ocr
from src.confirm import ConfirmationGate, ConfirmItem, DownstreamLockedError
from src.extract_text import Word
from src.fields import ExtractedField, extract_fields
from src.quarantine import QuarantinedText, scan_document

logger = logging.getLogger(__name__)

UNTRUSTED_FIELD = "untrusted_instruction_text"


@dataclass
class DocumentProfile:
    """The unconfirmed result of processing one document.

    ``status`` is a coarse readiness label, never an eligibility verdict:
      - ``"NEEDS_CONFIRMATION"``: fields extracted, awaiting renter confirmation.
      - ``"NEEDS_OCR"``: page is a raster image and OCR produced nothing usable.
    ``extraction_method`` is ``"text"`` or ``"ocr"``.
    """

    file_name: str
    document_type: str | None
    mode: str  # "text" | "image"
    status: str
    fields: list[ExtractedField] = dc_field(default_factory=list)
    quarantine: list[QuarantinedText] = dc_field(default_factory=list)
    gate: ConfirmationGate | None = None
    extraction_method: str = "text"

    def to_dict(self) -> dict:
        return {
            "file_name": self.file_name,
            "document_type": self.document_type,
            "mode": self.mode,
            "status": self.status,
            "extraction_method": self.extraction_method,
            "fields": [f.to_dict() for f in self.fields],
            "quarantine": [q.to_dict() for q in self.quarantine],
        }

    def to_downstream(self) -> list[dict]:
        """Return confirmed, releasable values for downstream use.

        Raises ``DownstreamLockedError`` unless the document has a confirmation
        gate and every value in it has been confirmed. Quarantined instruction
        text is never included.
        """
        if self.gate is None:
            raise DownstreamLockedError(
                f"{self.file_name}: extraction incomplete (status={self.status}); "
                "nothing to release."
            )
        return self.gate.release()


def _locked_gate(fields: list[ExtractedField]) -> ConfirmationGate:
    """Load extracted fields into a gate, all unconfirmed (downstream locked)."""
    gate = ConfirmationGate()
    for f in fields:
        gate.add(
            ConfirmItem(
                field=f.field,
                value=f.value,
                bbox=f.bbox,
                confidence=f.confidence,
                confirmed=False,
                quarantined=(f.field == UNTRUSTED_FIELD),
            )
        )
    return gate


def build_profile(
    pdf_path: str | Path,
    document_type: str | None = None,
    page_number: int = 1,
) -> DocumentProfile:
    """Process one document into a locked, unconfirmed ``DocumentProfile``.

    ``document_type`` is inferred from the file name when not supplied.
    """
    pdf_path = Path(pdf_path)
    file_name = pdf_path.name
    if document_type is None:
        document_type = allowlist.infer_document_type(file_name)

    mode = detect.detect_mode(pdf_path, page_number)

    if mode != "text":
        # Rasterized document: OCR the allowlisted fields off the page image.
        fields = []
        if document_type is not None and extract_ocr.is_available():
            try:
                fields = extract_ocr.extract_fields_ocr(pdf_path, document_type, page_number)
            except Exception as exc:  # OCR is best-effort; fall back to NEEDS_OCR.
                logger.warning("OCR failed for %s: %s", file_name, exc)
                fields = []
        if not fields:
            return DocumentProfile(
                file_name=file_name,
                document_type=document_type,
                mode=mode,
                status="NEEDS_OCR",
                extraction_method="ocr",
            )
        return DocumentProfile(
            file_name=file_name,
            document_type=document_type,
            mode=mode,
            status="NEEDS_CONFIRMATION",
            fields=fields,
            gate=_locked_gate(fields),
            extraction_method="ocr",
        )

    # Text path: allowlisted field extraction (never untrusted_instruction_text).
    fields = []
    if document_type is not None:
        fields = extract_fields(pdf_path, document_type, page_number)

    # Quarantine any injected instructions as inert data (audit trail always;
    # surfaced as an allowlisted field only where permitted).
    quarantined = scan_document(pdf_path, page_number)
    if document_type is not None and allowlist.is_allowed(document_type, UNTRUSTED_FIELD):
        for q in quarantined:
            fields.append(
                ExtractedField(
                    field=UNTRUSTED_FIELD,
                    value=q.text,
                    page=q.page,
                    bbox=q.bbox,
                    confidence=1.0,
                    confirmed=False,
                )
            )

    return DocumentProfile(
        file_name=file_name,
        document_type=document_type,
        mode=mode,
        status="NEEDS_CONFIRMATION",
        fields=fields,
        quarantine=quarantined,
        gate=_locked_gate(fields),
        extraction_method="text",
    )


# Re-export for convenience so callers can catch the lock from one place.
__all__ = ["DocumentProfile", "build_profile", "DownstreamLockedError", "Word"]
