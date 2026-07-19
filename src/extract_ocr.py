"""OCR extraction path for rasterized (image-only) documents.

INTERFACE ONLY for now. Roughly a third of the fixtures are rasterized: the page
is a single baked image with no usable text layer, so ``detect.detect_mode``
routes them here instead of to ``extract_text``. The public surface mirrors
``extract_text`` so ``profile`` can treat the two paths interchangeably once this
is implemented.

Implementation plan (not yet wired up):
  1. Render the page to a raster with pdfplumber's ``page.to_image(resolution=...)``.
  2. Run an OCR engine (e.g. Tesseract via ``pytesseract.image_to_data``) to get
     word-level text with pixel bounding boxes and per-word confidences.
  3. Convert pixel boxes back to PDF points and y-flip to the bottom-left origin
     so the boxes are interchangeable with the text-layer path
     (``x_pt = x_px * page_width / image_width``; ``y_bottom = page_height - ...``).
  4. Reuse the SAME watermark/geometry reassembly ideas as ``extract_text`` and
     feed results into the same ``fields`` label->column logic.
"""
from __future__ import annotations

from src.extract_text import Word


class OCRNotAvailableError(NotImplementedError):
    """Raised when the OCR path is invoked before it has been implemented."""


def extract_words(pdf_path, page_number: int = 1) -> list[Word]:
    """Return OCR'd words as bottom-left-origin ``Word`` objects.

    Mirrors ``extract_text.extract_words`` so downstream code is path-agnostic.
    Not implemented yet.
    """
    raise OCRNotAvailableError(
        "OCR path not implemented. Rasterized documents require an OCR backend "
        "(see module docstring). Only the text-layer path is wired up in Stage 1."
    )


def is_available() -> bool:
    """Return True once an OCR backend is wired up. Currently always False."""
    return False
