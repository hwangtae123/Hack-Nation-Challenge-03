"""End-to-end tests for `src.fields.extract_fields` against the gold fixtures.

Gold values are read here, and ONLY here, as a scoring reference -- `src/`
never reads gold. The manifest's `rasterized` column is deliberately not
trusted for routing (see `src/detect.py`); every test below re-derives text
vs. image mode itself via `detect.detect_mode` before deciding whether a
document is in scope for text-layer extraction.

Two independent checks are applied to every extracted field that has a gold
counterpart (except the quarantined `untrusted_instruction_text`, which this
module never extracts):

* Value equality -- numbers (gold value is int/float) are compared with
  `math.isclose(extracted, gold, abs_tol=0.01)` after coercing both sides to
  float; everything else (strings) is compared with exact `==`.
* Bbox correctness -- the CENTER of the extracted bbox must fall inside the
  gold bbox expanded by 2.0 points on every side. Gold boxes are slightly
  padded supersets of the tight glyph boxes, so exact coordinate comparison
  would be too strict.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

# Make the repo root importable regardless of the current working directory
# pytest was invoked from.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import allowlist, config, detect, fields  # noqa: E402  (path setup above)

# Fields that are intentionally quarantined, never produced by extract_fields,
# and therefore excluded from every value/bbox comparison below.
_QUARANTINED_FIELDS = {"untrusted_instruction_text"}

# Bbox-center-in-gold check tolerance, in PDF points.
_BBOX_PAD = 2.0

# Numeric tolerance for value comparisons (dollars/hours are given to at most
# two decimal places in the fixtures).
_VALUE_ABS_TOL = 0.01


def _load_gold() -> dict[str, dict[str, dict]]:
    """Load `document_gold.jsonl` into {file_name: {field_name: gold_record}}.

    This is the ONLY place gold is read in this test suite; it is used
    purely as a scoring reference for the assertions below.
    """
    by_file: dict[str, dict[str, dict]] = {}
    with open(config.GOLD_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            by_file[record["file_name"]] = {gf["field"]: gf for gf in record["fields"]}
    return by_file


def _load_manifest() -> list[dict]:
    """Load `document_manifest.csv` rows as plain dicts."""
    with open(config.MANIFEST_PATH, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


GOLD_BY_FILE = _load_gold()
MANIFEST_ROWS = _load_manifest()
MANIFEST_BY_FILE = {row["file_name"]: row for row in MANIFEST_ROWS}


def _values_match(extracted: object, gold_value: object) -> bool:
    """Compare an extracted value to its gold value.

    Numbers are compared with `math.isclose` (abs_tol=0.01) after coercing
    both sides to float; everything else uses exact equality.
    """
    if isinstance(gold_value, (int, float)) and not isinstance(gold_value, bool):
        try:
            return math.isclose(float(extracted), float(gold_value), abs_tol=_VALUE_ABS_TOL)
        except (TypeError, ValueError):
            return False
    return extracted == gold_value


def _bbox_center_in_gold(extracted_bbox: list[float], gold_bbox: list[float]) -> bool:
    """True iff the center of `extracted_bbox` lies inside `gold_bbox` + pad."""
    cx = (extracted_bbox[0] + extracted_bbox[2]) / 2.0
    cy = (extracted_bbox[1] + extracted_bbox[3]) / 2.0
    gx0, gy0, gx1, gy1 = gold_bbox
    return (gx0 - _BBOX_PAD) <= cx <= (gx1 + _BBOX_PAD) and (gy0 - _BBOX_PAD) <= cy <= (gy1 + _BBOX_PAD)


def _assert_matches_gold(file_name: str, document_type: str) -> int:
    """Extract `file_name` and assert every gold field matches value + bbox.

    Returns the number of (non-quarantined) gold fields checked, so callers
    iterating many documents can assert the loop wasn't accidentally a no-op.
    """
    gold_fields = GOLD_BY_FILE[file_name]
    pdf_path = config.document_path(file_name)
    extracted_by_field = {ef.field: ef for ef in fields.extract_fields(pdf_path, document_type)}

    checked = 0
    for name, gold_field in gold_fields.items():
        if name in _QUARANTINED_FIELDS:
            continue
        checked += 1
        assert name in extracted_by_field, f"{file_name}: missing extracted field {name!r}"
        ef = extracted_by_field[name]
        assert _values_match(ef.value, gold_field["value"]), (
            f"{file_name}.{name}: extracted value {ef.value!r} != gold value "
            f"{gold_field['value']!r}"
        )
        assert _bbox_center_in_gold(ef.bbox, gold_field["bbox"]), (
            f"{file_name}.{name}: extracted bbox {ef.bbox} center is not inside "
            f"gold bbox {gold_field['bbox']} (+/- {_BBOX_PAD}pt)"
        )
    return checked


def test_target_pay_stub_matches_gold():
    """Required end-to-end check: every gold field for the target pay stub
    (hh-001_d03_pay_stub.pdf) must be extracted with the correct value and a
    bbox whose center lands inside the (padded) gold bbox.
    """
    file_name = "hh-001_d03_pay_stub.pdf"
    document_type = MANIFEST_BY_FILE[file_name]["document_type"]
    pdf_path = config.document_path(file_name)

    # Sanity check on the routing assumption: this fixture must be text-layer
    # for extract_fields (the text-layer path) to be the right tool at all.
    assert detect.detect_mode(pdf_path) == "text"

    checked = _assert_matches_gold(file_name, document_type)
    assert checked > 0, "expected at least one gold field for the target pay stub"


def test_all_text_layer_docs():
    """For every manifest document that is genuinely text-layer (per
    `detect.detect_mode`, not the untrusted manifest `rasterized` flag), every
    gold field except `untrusted_instruction_text` must match on value and
    bbox center.
    """
    total_checked = 0
    for row in MANIFEST_ROWS:
        file_name = row["file_name"]
        document_type = row["document_type"]
        if file_name not in GOLD_BY_FILE:
            continue  # no scoring reference available for this document
        pdf_path = config.document_path(file_name)
        if detect.detect_mode(pdf_path) != "text":
            continue  # image-mode page; out of scope for this extractor
        total_checked += _assert_matches_gold(file_name, document_type)

    assert total_checked > 0, "expected at least one text-layer document with gold fields"


def test_no_disallowed_fields():
    """extract_fields must never emit a field outside the document type's
    allowlist, and must never emit `untrusted_instruction_text`.
    """
    total_docs = 0
    for row in MANIFEST_ROWS:
        file_name = row["file_name"]
        document_type = row["document_type"]
        pdf_path = config.document_path(file_name)
        if detect.detect_mode(pdf_path) != "text":
            continue
        total_docs += 1
        allowed = set(allowlist.allowed_fields(document_type))
        for ef in fields.extract_fields(pdf_path, document_type):
            assert ef.field != "untrusted_instruction_text", (
                f"{file_name}: extract_fields must never emit untrusted_instruction_text"
            )
            assert ef.field in allowed, (
                f"{file_name}: extracted field {ef.field!r} is not allowlisted for "
                f"{document_type!r}"
            )

    assert total_docs > 0, "expected at least one text-layer document in the manifest"


def test_all_unconfirmed():
    """Every ExtractedField must be unconfirmed with a valid confidence score."""
    total_fields = 0
    for row in MANIFEST_ROWS:
        file_name = row["file_name"]
        document_type = row["document_type"]
        pdf_path = config.document_path(file_name)
        if detect.detect_mode(pdf_path) != "text":
            continue
        for ef in fields.extract_fields(pdf_path, document_type):
            total_fields += 1
            assert ef.confirmed is False
            assert 0 < ef.confidence <= 1

    assert total_fields > 0, "expected at least one extracted field across all text-layer docs"
