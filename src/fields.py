"""Allowlisted field extraction for text-layer RealDoor documents.

The synthetic documents share one template: an all-uppercase LABEL row
followed directly by a VALUE row, with values left-aligned to their label and
laid out in columns. A value's column spans from its label's ``x0`` to the
next label's ``x0`` on the same label row (or to the page edge when the
field owns the last column on that row).

This module locates each allowlisted field's label, carves out its value
column, converts whatever token(s) land in that column to the field's
expected type, and reports a source bounding box (bottom-left origin,
matching :mod:`extract_text`) alongside a deterministic confidence score.
Every emitted field is unconfirmed (``confirmed=False``) -- confirmation is a
later, human-in-the-loop step this module has no opinion about.

Nothing here trusts the manifest's ``rasterized`` flag, and nothing here
reads gold values -- gold is only ever consumed by the test suite as a
scoring reference. Callers are expected to route image-mode pages to a
separate OCR path (see :func:`detect.detect_mode`) before calling
:func:`extract_fields`.

``untrusted_instruction_text`` is deliberately never produced here, even for
document types that allowlist it -- injected/adversarial text is quarantined
by a separate module so it is only ever handled as inert data, never as an
instruction.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Callable

import pdfplumber

from . import allowlist
from .extract_text import Word, extract_words, reconstruct_lines

logger = logging.getLogger(__name__)

# US Letter fallback width in points, used only if the page width cannot be
# read from the PDF itself.
_DEFAULT_PAGE_WIDTH = 612.0

# Horizontal slack (points) applied to a value column's search window: the
# label's left edge is nudged left and the boundary's left edge nudged right
# so slightly indented or kerned value tokens are still captured without
# pulling in the *next* column's tokens.
_LEFT_SLACK = 6.0
_RIGHT_SLACK = 3.0

# Token patterns used both to pick the right token out of a candidate list
# and to score extraction confidence.
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_MONTH_RE = re.compile(r"^\d{4}-\d{2}$")
_INT_RE = re.compile(r"^\d+$")
_MONEY_RE = re.compile(r"^\$?[\d,]+(\.\d+)?$")
_WORD_RE = re.compile(r"^[A-Za-z]+$")

# Deterministic confidence scores. Tests only assert 0 < confidence <= 1, but
# these are kept distinct so a downstream reviewer can tell a clean parse
# from a best-effort fallback.
_CONF_CLEAN = 0.98  # a single token matched the expected format exactly
_CONF_MULTIWORD_TEXT = 0.95  # a multi-token free-text value (name/address)
_CONF_DIRTY = 0.6  # a token was taken but did not match the expected format


@dataclass
class ExtractedField:
    """One field value located on a page, with its source box and confidence."""

    field: str
    value: Any
    page: int
    bbox: list[float]  # [x0, y0, x1, y1], bottom-left origin, PDF points
    confidence: float
    confirmed: bool = False

    def to_dict(self) -> dict:
        """Serialize to the plain-dict shape used by the rest of the pipeline."""
        return {
            "field": self.field,
            "value": self.value,
            "page": self.page,
            "bbox": self.bbox,
            "confidence": self.confidence,
            "confirmed": self.confirmed,
        }


# (field, anchor_label, boundary_label_or_None, kind) per document type.
# `boundary_label` is another label on the *same* label row that bounds the
# value column on the right; None means "to the page edge" (the field owns
# the last column on its label row).
FIELD_LAYOUT: dict[str, list[tuple[str, str, str | None, str]]] = {
    "application_summary": [
        ("person_name", "APPLICANT", "HOUSEHOLD SIZE", "text"),
        ("household_size", "HOUSEHOLD SIZE", None, "int"),
        ("address", "MAILING ADDRESS", None, "text"),
        ("application_date", "APPLICATION DATE", None, "date"),
    ],
    "pay_stub": [
        ("person_name", "EMPLOYEE", "PAY DATE", "text"),
        ("pay_date", "PAY DATE", None, "date"),
        ("pay_period_start", "PAY PERIOD", "PAY FREQUENCY", "date_first"),
        ("pay_period_end", "PAY PERIOD", "PAY FREQUENCY", "date_second"),
        ("pay_frequency", "PAY FREQUENCY", None, "word"),
        ("regular_hours", "REGULAR HOURS", "HOURLY RATE", "int"),
        ("hourly_rate", "HOURLY RATE", "GROSS PAY", "money"),
        ("gross_pay", "GROSS PAY", "NET PAY", "money"),
        ("net_pay", "NET PAY", None, "money"),
    ],
    "employment_letter": [
        ("person_name", "EMPLOYEE", "LETTER DATE", "text"),
        ("document_date", "LETTER DATE", None, "date"),
        ("weekly_hours", "HOURS PER WEEK", "HOURLY RATE", "int"),
        ("hourly_rate", "HOURLY RATE", None, "money"),
    ],
    "benefit_letter": [
        ("person_name", "RECIPIENT", "LETTER DATE", "text"),
        ("document_date", "LETTER DATE", None, "date"),
        ("monthly_benefit", "MONTHLY AMOUNT", "FREQUENCY", "money"),
        ("benefit_frequency", "FREQUENCY", None, "word"),
    ],
    "gig_statement": [
        ("person_name", "WORKER", "STATEMENT MONTH", "text"),
        ("statement_month", "STATEMENT MONTH", None, "month"),
        ("gross_receipts", "RECEIPTS", "PLATFORM FEES", "money"),
        ("platform_fees", "PLATFORM FEES", None, "money"),
    ],
}


# --------------------------------------------------------------------------
# Layout helpers: label rows, label lookup, column carving.
# --------------------------------------------------------------------------


def _norm(s: str) -> str:
    """Collapse whitespace and upper-case a string, for label matching."""
    return re.sub(r"\s+", "", s).upper()


def _is_label_row(line: list[Word]) -> bool:
    """Return True iff `line` is an all-uppercase template LABEL row.

    A label row's alphabetic words are all upper-case. This distinguishes
    template rows (e.g. "PAY PERIOD PAY FREQUENCY") from prose and value
    rows (e.g. "Mara North"), which mix case. Tokens with no alphabetic
    characters (numbers, punctuation, the ``(cid:127)`` glyph placeholder)
    are ignored for the case check but do not by themselves make a row a
    label row.
    """
    has_alpha = False
    for w in line:
        letters = [c for c in w.text if c.isalpha()]
        if letters:
            has_alpha = True
            if not all(c.isupper() for c in letters):
                return False
    return has_alpha


def _find_label_x0_on_line(line: list[Word], label: str) -> float | None:
    """Locate `label` on one specific line; return its left edge (x0) or None.

    Matching is done on the whitespace-collapsed, upper-cased concatenation
    of the line's words, so a multi-word label (e.g. "PAY FREQUENCY") is
    found even though it is split across separate `Word` tokens.
    """
    target = _norm(label)
    concat = ""
    spans: list[tuple[int, Word]] = []
    for w in line:
        spans.append((len(concat), w))
        concat += _norm(w.text)
    pos = concat.find(target)
    if pos == -1:
        return None
    for start, w in spans:
        end = start + len(_norm(w.text))
        if start <= pos < end:
            return w.x0
    return line[0].x0 if line else None  # defensive; spans always cover pos


def _find_label(lines: list[list[Word]], label: str) -> tuple[int | None, float | None]:
    """Find `label` on the first label row that contains it.

    Search is restricted to label rows so prose that happens to echo a
    label word (unlikely, but possible) can never be mistaken for the
    template's own label row. Returns (line_index, x0_of_the_label) or
    (None, None) if no label row contains it.
    """
    for i, ln in enumerate(lines):
        if not _is_label_row(ln):
            continue
        x0 = _find_label_x0_on_line(ln, label)
        if x0 is not None:
            return i, x0
    return None, None


def _words_in_x(line: list[Word], x_lo: float, x_hi: float) -> list[Word]:
    """Return words on `line` whose horizontal center falls in [x_lo, x_hi)."""
    return [w for w in line if x_lo <= (w.x0 + w.x1) / 2 < x_hi]


def _union_bbox(words: list[Word]) -> list[float]:
    """Bounding box (bottom-left origin) enclosing every word in `words`."""
    return [
        min(w.x0 for w in words),
        min(w.y0 for w in words),
        max(w.x1 for w in words),
        max(w.y1 for w in words),
    ]


def _page_width(pdf_path, page_number: int) -> float:
    """Read the page width in points, falling back to US Letter on failure."""
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            return float(pdf.pages[page_number - 1].width)
    except Exception:  # pragma: no cover - defensive; fixtures are well-formed
        logger.debug("Could not read page width for %s; using default", pdf_path)
        return _DEFAULT_PAGE_WIDTH


# --------------------------------------------------------------------------
# Value selection per field "kind".
#
# A selector takes the candidate word list for a value column and returns
# (value, words_used_for_bbox, confidence), or None if no value could be
# reliably extracted from those candidates -- callers treat None as "skip
# this field".
# --------------------------------------------------------------------------

_Selection = tuple[Any, list[Word], float]


def _select_text(cand: list[Word]) -> _Selection | None:
    """Join every candidate token with a space (person_name, address)."""
    if not cand:
        return None
    value = " ".join(w.text for w in cand)
    return value, cand, _CONF_MULTIWORD_TEXT


def _select_word(cand: list[Word]) -> _Selection | None:
    """Take the first token as-is (pay_frequency, benefit_frequency)."""
    if not cand:
        return None
    token = cand[0]
    confidence = _CONF_CLEAN if _WORD_RE.match(token.text) else _CONF_DIRTY
    return token.text, [token], confidence


def _first_matching(cand: list[Word], pattern: re.Pattern) -> tuple[Word, bool] | None:
    """First token matching `pattern`; else the first token as a dirty fallback."""
    for w in cand:
        if pattern.match(w.text):
            return w, True
    if cand:
        return cand[0], False
    return None


def _select_date(cand: list[Word]) -> _Selection | None:
    """First token matching YYYY-MM-DD, falling back to the first token."""
    picked = _first_matching(cand, _DATE_RE)
    if picked is None:
        return None
    w, matched = picked
    return w.text, [w], (_CONF_CLEAN if matched else _CONF_DIRTY)


def _select_month(cand: list[Word]) -> _Selection | None:
    """First token matching YYYY-MM, falling back to the first token."""
    picked = _first_matching(cand, _MONTH_RE)
    if picked is None:
        return None
    w, matched = picked
    return w.text, [w], (_CONF_CLEAN if matched else _CONF_DIRTY)


def _select_date_nth(cand: list[Word], n: int) -> _Selection | None:
    """The (n+1)-th token matching YYYY-MM-DD (n=0 first, n=1 second).

    Used to split a "start end" date pair sharing one value column
    (pay_period_start / pay_period_end). Unlike `_select_date`, there is no
    single-token fallback: if fewer than n+1 dates are present the field is
    skipped rather than guessing.
    """
    matches = [w for w in cand if _DATE_RE.match(w.text)]
    if len(matches) <= n:
        return None
    w = matches[n]
    return w.text, [w], _CONF_CLEAN


def _select_int(cand: list[Word]) -> _Selection | None:
    """First token, parsed as an integer (regular_hours, household_size, ...)."""
    if not cand:
        return None
    token = cand[0]
    matched = bool(_INT_RE.match(token.text))
    try:
        value = int(token.text)
    except ValueError:
        return None
    return value, [token], (_CONF_CLEAN if matched else _CONF_DIRTY)


def _select_money(cand: list[Word]) -> _Selection | None:
    """First token, parsed as a dollar amount (hourly_rate, gross_pay, ...)."""
    if not cand:
        return None
    token = cand[0]
    matched = bool(_MONEY_RE.match(token.text))
    cleaned = token.text.replace("$", "").replace(",", "")
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return value, [token], (_CONF_CLEAN if matched else _CONF_DIRTY)


_SELECTORS: dict[str, Callable[[list[Word]], _Selection | None]] = {
    "text": _select_text,
    "word": _select_word,
    "date": _select_date,
    "month": _select_month,
    "date_first": lambda cand: _select_date_nth(cand, 0),
    "date_second": lambda cand: _select_date_nth(cand, 1),
    "int": _select_int,
    "money": _select_money,
}


# --------------------------------------------------------------------------
# Public entry point.
# --------------------------------------------------------------------------


def extract_fields(pdf_path, document_type: str, page_number: int = 1) -> list[ExtractedField]:
    """Extract every allowlisted, locatable field for `document_type`.

    Walks the field layout registered for `document_type`, finds each
    field's label on a label row, carves out the value column below it
    (bounded by the next label on that row, or the page edge), converts
    whatever token(s) land in that column to the field's expected type, and
    returns one `ExtractedField` per field that was successfully located.

    A field whose label cannot be found, whose value row is missing, whose
    value column is empty, or whose token(s) cannot be parsed into the
    expected type is silently skipped (logged at debug level) -- this
    function never raises for a missing field.

    Only fields on `allowlist.allowed_fields(document_type)` are ever
    considered, and `untrusted_instruction_text` is never emitted here even
    if it is on that allowlist -- it is quarantined by a separate module.
    """
    layout = FIELD_LAYOUT.get(document_type, [])
    if not layout:
        logger.debug("No field layout registered for document_type=%r", document_type)
        return []

    allowed = set(allowlist.allowed_fields(document_type))

    words = extract_words(pdf_path, page_number)
    if not words:
        logger.debug("No text-layer words extracted from %s page %d", pdf_path, page_number)
        return []
    lines = reconstruct_lines(words)
    page_width = _page_width(pdf_path, page_number)

    results: list[ExtractedField] = []
    for field_name, anchor_label, boundary_label, kind in layout:
        if field_name == "untrusted_instruction_text":
            # Quarantined by a separate module; never extracted here even if
            # it is technically on the allowlist for this document type.
            continue
        if field_name not in allowed:
            # Defensive: FIELD_LAYOUT should already mirror the allowlist,
            # but never emit a field the allowlist does not permit.
            logger.debug("Field %r not allowlisted for %r; skipping", field_name, document_type)
            continue

        line_idx, label_x0 = _find_label(lines, anchor_label)
        if line_idx is None:
            logger.debug("Anchor label %r not found for field %r", anchor_label, field_name)
            continue
        if line_idx + 1 >= len(lines):
            logger.debug("No value row below anchor label for field %r", field_name)
            continue

        x_hi = page_width
        if boundary_label is not None:
            boundary_x0 = _find_label_x0_on_line(lines[line_idx], boundary_label)
            if boundary_x0 is not None:
                x_hi = boundary_x0
            else:
                logger.debug(
                    "Boundary label %r not found on label row for field %r; "
                    "using page width",
                    boundary_label, field_name,
                )

        value_line = lines[line_idx + 1]
        candidates = _words_in_x(value_line, label_x0 - _LEFT_SLACK, x_hi - _RIGHT_SLACK)
        if not candidates:
            logger.debug("No value tokens found in column for field %r", field_name)
            continue

        selection = _SELECTORS[kind](candidates)
        if selection is None:
            logger.debug(
                "Could not parse a %r value for field %r from tokens %r",
                kind, field_name, [w.text for w in candidates],
            )
            continue

        value, chosen_words, confidence = selection
        results.append(
            ExtractedField(
                field=field_name,
                value=value,
                page=page_number,
                bbox=_union_bbox(chosen_words),
                confidence=confidence,
                confirmed=False,
            )
        )

    return results
