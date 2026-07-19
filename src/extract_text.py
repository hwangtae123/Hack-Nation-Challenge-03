"""Text-layer extraction for the RealDoor synthetic documents.

Three data traps are handled here, all validated against the real fixtures:

1. Diagonal "DOCUMENT / TRAINING FIXTURE" watermark. Its glyphs are ~32-40pt
   tall and are spatially interleaved *inside* real value tokens, e.g. the pay
   date renders as ``2026-0T6-20`` and the hourly rate as ``$28.50A``. Filtering
   at the *word* level is too coarse -- pdfplumber merges the tall watermark
   glyph into the value word, so the whole word gets a large height and would be
   dropped along with the value. We therefore filter at the *character* level
   (drop chars taller than ``WATERMARK_HEIGHT_MAX``) and re-group the survivors
   into words. Content glyphs are <= ~18pt, so the split is clean.

2. Reverse character/word storage. ``extract_text()`` returns glyphs in storage
   order, which is scrambled. We ignore storage order entirely and re-assemble
   purely from geometry: cluster words into visual lines by their ``top``
   coordinate, then sort each line left-to-right by ``x0``.

3. Coordinate system. pdfplumber uses a top-left origin; the gold boxes use
   ``pdf_points_bottom_left_origin``. Every emitted box is y-flipped via
   ``y = page_height - y_topleft`` so it lines up with gold.
"""
from __future__ import annotations

from dataclasses import dataclass

import pdfplumber

# Glyphs strictly taller than this are treated as watermark and dropped.
# Content text tops out around 18pt; the watermark starts around 26pt.
WATERMARK_HEIGHT_MAX = 20.0

# Words whose ``top`` differ by no more than this belong to the same visual line.
LINE_TOLERANCE = 6.0

# Horizontal gap (points) below which adjacent glyphs are joined into one word.
# Kept small so column-separated tokens ("Mara North", "76 $28.50") stay split.
WORD_X_TOLERANCE = 1.5


@dataclass
class Word:
    """A single reconstructed word with a bottom-left-origin bounding box.

    ``bbox`` is ``[x0, y0, x1, y1]`` in PDF points with a bottom-left origin,
    matching the gold ``pdf_points_bottom_left_origin`` convention (``y0`` is the
    bottom edge, ``y1`` the top edge, ``y1 > y0``).
    """

    text: str
    x0: float
    y0: float  # bottom edge, bottom-left origin
    x1: float
    y1: float  # top edge, bottom-left origin
    page: int  # 1-based

    @property
    def bbox(self) -> list[float]:
        return [self.x0, self.y0, self.x1, self.y1]


def _keep_char(obj: dict, width: float, height: float) -> bool:
    """Return True for a content character (not watermark, in-bounds, upright)."""
    if obj.get("object_type") != "char":
        # Non-char objects are irrelevant to word extraction; drop them so the
        # filtered page only carries the glyphs we want to re-group.
        return False
    if obj.get("height", 0.0) > WATERMARK_HEIGHT_MAX:
        return False
    if not obj.get("upright", True):
        return False
    if not (0 <= obj["x0"] and obj["x1"] <= width and 0 <= obj["top"] and obj["bottom"] <= height):
        return False
    return True


def extract_words(pdf_path, page_number: int = 1) -> list[Word]:
    """Extract cleaned, watermark-free words with bottom-left-origin boxes.

    Returns words sorted top-to-bottom then left-to-right. Storage order is
    ignored; every position comes from geometry.
    """
    words: list[Word] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        page = pdf.pages[page_number - 1]
        h = page.height
        w = page.width
        content = page.filter(lambda o: _keep_char(o, w, h))
        for raw in content.extract_words(use_text_flow=False, x_tolerance=WORD_X_TOLERANCE):
            words.append(
                Word(
                    text=raw["text"],
                    x0=round(raw["x0"], 2),
                    y0=round(h - raw["bottom"], 2),  # y-flip: bottom edge
                    x1=round(raw["x1"], 2),
                    y1=round(h - raw["top"], 2),  # y-flip: top edge
                    page=page_number,
                )
            )
    # Top-to-bottom (largest y1 first), then left-to-right.
    words.sort(key=lambda wd: (-wd.y1, wd.x0))
    return words


def reconstruct_lines(words: list[Word]) -> list[list[Word]]:
    """Group words into visual lines from geometry alone.

    Words are clustered by their top edge (``y1``) within ``LINE_TOLERANCE`` and
    each resulting line is sorted left-to-right. Lines are returned top-to-bottom.
    """
    if not words:
        return []
    ordered = sorted(words, key=lambda wd: -wd.y1)
    lines: list[list[Word]] = []
    current: list[Word] = []
    ref_top: float | None = None
    for wd in ordered:
        if ref_top is None or abs(wd.y1 - ref_top) <= LINE_TOLERANCE:
            current.append(wd)
            # Anchor the line on its first (highest) word's top edge.
            ref_top = current[0].y1 if ref_top is None else ref_top
        else:
            lines.append(sorted(current, key=lambda x: x.x0))
            current = [wd]
            ref_top = wd.y1
    if current:
        lines.append(sorted(current, key=lambda x: x.x0))
    return lines


def line_text(line: list[Word]) -> str:
    """Join a reconstructed line's words left-to-right into a single string."""
    return " ".join(wd.text for wd in line)


def page_text(pdf_path, page_number: int = 1) -> str:
    """Full page text, position-reassembled, one visual line per output line."""
    lines = reconstruct_lines(extract_words(pdf_path, page_number))
    return "\n".join(line_text(ln) for ln in lines)
