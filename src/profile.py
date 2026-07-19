"""Assemble a single document into an unconfirmed extraction profile.

This is the hand-off boundary of Stage 1. It ties the pieces together:
    detect mode -> (text path) extract allowlisted fields
                -> quarantine any injected instructions as inert data
                -> load everything into a renter ConfirmationGate, LOCKED.

Hard rule enforced here: a value whose ``confirmed`` flag is False must never
reach any downstream calculation or aggregation. ``DocumentProfile.to_downstream``
is the only exit, and it delegates to the gate, which raises while anything is
still pending. Quarantined instruction text is never releasable at all. This
module makes no eligibility, approval, or scoring decision.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any

from src import allowlist, detect
from src.confirm import ConfirmationGate, ConfirmItem, DownstreamLockedError
from src.extract_text import Word
from src.fields import ExtractedField, extract_fields
from src.quarantine import QuarantinedText, scan_document

UNTRUSTED_FIELD = "untrusted_instruction_text"


@dataclass
class DocumentProfile:
    """The unconfirmed result of processing one document.

    ``status`` is a coarse readiness label, never an eligibility verdict:
      - ``"NEEDS_CONFIRMATION"``: text extracted, awaiting renter confirmation.
      - ``"NEEDS_OCR"``: page is a raster image; the OCR path is not wired up yet.
    """

    file_name: str
    document_type: str | None
    mode: str  # "text" | "image"
    status: str
    fields: list[ExtractedField] = dc_field(default_factory=list)
    quarantine: list[QuarantinedText] = dc_field(default_factory=list)
    gate: ConfirmationGate | None = None

    def to_dict(self) -> dict:
        return {
            "file_name": self.file_name,
            "document_type": self.document_type,
            "mode": self.mode,
            "status": self.status,
            "fields": [f.to_dict() for f in self.fields],
            "quarantine": [q.to_dict() for q in self.quarantine],
        }

    def to_downstream(self) -> list[dict]:
        """Return confirmed, releasable values for downstream use.

        Raises ``DownstreamLockedError`` unless the document has been fully
        confirmed by the renter. Quarantined instruction text is never included.
        """
        if self.mode != "text":
            raise DownstreamLockedError(
                f"{self.file_name}: extraction incomplete (mode={self.mode}); "
                "cannot release to downstream."
            )
        if self.gate is None:
            raise DownstreamLockedError(f"{self.file_name}: no confirmation gate.")
        return self.gate.release()


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
        # Stage 1 only wires up the text path; raster docs need OCR.
        return DocumentProfile(
            file_name=file_name,
            document_type=document_type,
            mode=mode,
            status="NEEDS_OCR",
        )

    # 1) Allowlisted field extraction (never includes untrusted_instruction_text).
    fields: list[ExtractedField] = []
    if document_type is not None:
        fields = extract_fields(pdf_path, document_type, page_number)

    # 2) Quarantine any injected instructions as inert data (audit trail always;
    #    surfaced as an allowlisted field only where permitted).
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

    # 3) Load everything into the renter gate, LOCKED (confirmed=False).
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

    return DocumentProfile(
        file_name=file_name,
        document_type=document_type,
        mode=mode,
        status="NEEDS_CONFIRMATION",
        fields=fields,
        quarantine=quarantined,
        gate=gate,
    )


# Re-export for convenience so callers can catch the lock from one place.
__all__ = ["DocumentProfile", "build_profile", "DownstreamLockedError", "Word"]
