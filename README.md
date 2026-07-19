# RealDoor — Stage 1: Document Extraction Pipeline

Extract an allowlisted set of fields from synthetic income documents (PDF),
attach a **source box** and **confidence** to every value, quarantine any
injected instructions as inert data, and keep every value **locked** until a
renter confirms it.

This pipeline stops at the extraction → confirmation hand-off. It makes **no**
eligibility, approval, or scoring decision.

## Why this is not a plain PDF text dump

The fixtures carry three deliberate traps, all handled from geometry (never from
storage order or the manifest flags):

1. **Diagonal watermark.** A large "DOCUMENT / TRAINING FIXTURE" watermark is
   spatially interleaved *inside* value tokens (`2026-0T6-20`, `$28.50A`).
   pdfplumber merges the tall watermark glyph into the value word, so filtering
   at the word level would drop the value. We filter at the **character** level
   (drop glyphs taller than ~20pt) and re-group survivors into words.
2. **Reverse storage.** `extract_text()` returns scrambled order. We ignore
   storage order entirely and re-assemble words into visual lines by geometry.
3. **Coordinate system.** pdfplumber uses a top-left origin; gold uses
   `pdf_points_bottom_left_origin`. Every emitted box is y-flipped
   (`y = page_height - y_topleft`).

Rasterized/image documents are detected by us (not trusted from the manifest)
and routed to an OCR path that is interface-only in Stage 1.

## Layout of `src/`

| module | responsibility |
| --- | --- |
| `config.py` | filesystem locations for documents / gold / manifest |
| `detect.py` | text-layer vs image detection (self-decided, flag not trusted) |
| `extract_text.py` | char-level watermark filter, geometry reassembly, y-flip |
| `extract_ocr.py` | OCR path for raster docs — **interface only** for now |
| `allowlist.py` | per-document-type allowed fields (the only fields we may emit) |
| `fields.py` | label→column extraction → `{field, value, page, bbox, confidence, confirmed:false}` |
| `quarantine.py` | detect injected instructions → `untrusted_instruction_text` (inert data) |
| `confirm.py` | renter confirmation gate; downstream stays locked until confirmed |
| `profile.py` | glue: one document → one locked, unconfirmed `DocumentProfile` |

## Safety properties

- **Nothing hardcoded from gold.** Gold is read only by the test suite, as a
  scoring reference.
- **Allowlist enforced.** Only fields on a document type's allowlist are emitted.
- **Injections are data, never instructions.** Matched injection text is returned
  as inert `untrusted_instruction_text` and is *never* releasable downstream.
- **Locked until confirmed.** `DocumentProfile.to_downstream()` raises
  `DownstreamLockedError` until the renter confirms every value.

## Running

```bash
pip install -r requirements.txt
# On Windows, force UTF-8 so the console can print document text:
PYTHONIOENCODING=utf-8 PYTHONUTF8=1 python -m pytest tests -q
```

The end-to-end gold check for `hh-001_d03_pay_stub.pdf` lives in
`tests/test_extract.py`; injection quarantine is covered by
`tests/test_quarantine.py`.
