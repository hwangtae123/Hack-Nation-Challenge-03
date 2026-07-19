"""Per-document-type field allowlists.

These are the ONLY fields the pipeline is permitted to emit for a given document
type. Anything a document happens to contain that is not on this list -- extra
labels, marketing copy, and especially injected instructions -- must never be
surfaced as an extracted field. `untrusted_instruction_text` is intentionally
allowlisted for the two document types that can carry adversarial text so that
such text is *quarantined as data*, never executed.
"""
from __future__ import annotations

# Document type -> ordered list of allowed field names.
ALLOWLIST: dict[str, list[str]] = {
    "application_summary": [
        "person_name",
        "household_size",
        "address",
        "application_date",
    ],
    "pay_stub": [
        "person_name",
        "pay_date",
        "pay_period_start",
        "pay_period_end",
        "pay_frequency",
        "gross_pay",
        "net_pay",
        "hourly_rate",
        "regular_hours",
        "untrusted_instruction_text",
    ],
    "employment_letter": [
        "person_name",
        "document_date",
        "hourly_rate",
        "weekly_hours",
    ],
    "benefit_letter": [
        "person_name",
        "document_date",
        "monthly_benefit",
        "benefit_frequency",
    ],
    "gig_statement": [
        "person_name",
        "statement_month",
        "gross_receipts",
        "platform_fees",
        "untrusted_instruction_text",
    ],
}

DOCUMENT_TYPES = tuple(ALLOWLIST.keys())


def allowed_fields(document_type: str) -> list[str]:
    """Return the allowed field names for a document type (empty if unknown)."""
    return list(ALLOWLIST.get(document_type, []))


def is_allowed(document_type: str, field: str) -> bool:
    """Return True iff `field` may be emitted for `document_type`."""
    return field in ALLOWLIST.get(document_type, ())


def infer_document_type(file_name: str) -> str | None:
    """Best-effort document-type inference from a file name.

    File names follow ``hh-001_d03_pay_stub.pdf``; the type is the suffix after
    the ``dNN_`` segment. Returns None when no known type matches.
    """
    stem = file_name.lower()
    if stem.endswith(".pdf"):
        stem = stem[:-4]
    for doc_type in ALLOWLIST:
        if stem.endswith(doc_type):
            return doc_type
    return None
