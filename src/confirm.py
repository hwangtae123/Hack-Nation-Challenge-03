"""Renter-confirmation gate for extracted document values.

Every value this pipeline extracts (a name, a pay amount, a bbox-anchored
number -- anything) is held behind a `ConfirmationGate` until a renter
explicitly confirms it (or edits it, which also confirms it). Nothing may be
treated as "downstream data" -- used for eligibility, scoring, or any other
purpose -- until it has passed through this gate. `release()` is the single
choke point: it raises `DownstreamLockedError` unless every relevant item has
been confirmed.

This module is storage-agnostic: `ConfirmationGate` holds its items in memory
only (a plain list). Persisting confirmations (e.g. to a database or a review
UI's backing store) is a concern for a caller, not for this module.

Quarantined items (`quarantined=True`) are how `untrusted_instruction_text`
values flow through this gate. They are always injected-text candidates
identified by `src/quarantine.py`, and per that module's contract they must
never be able to influence pipeline behavior. This gate enforces the other
half of that contract: a quarantined item can be shown to a renter and even
"confirmed" (acknowledged as seen), but `release()` unconditionally excludes
quarantined items from its output -- confirming one never unlocks it into
downstream data. There is no code path in this file by which a quarantined
item's `value` (the untrusted instruction text) reaches `release()`'s return
value.

Design decision -- do quarantined items count toward `is_ready()`?
No. `pending()` / `is_ready()` only consider non-quarantined items. A
quarantined item is, by definition, never going to be released regardless of
its `confirmed` flag, so making it a gatekeeper for the *other* items would
serve no protective purpose -- it would only force a renter to interact with
untrusted instruction text before they can confirm their own legitimate
values. Quarantined items may still be confirmed/edited/rejected for
bookkeeping (e.g. "renter acknowledged this was flagged"), but that state
never appears in `release()` output and never affects whether the gate is
"ready".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class DownstreamLockedError(Exception):
    """Raised when `release()` is called while non-quarantined items are pending."""


@dataclass
class ConfirmItem:
    """A single extracted value awaiting (or having received) renter confirmation.

    `quarantined=True` marks this as untrusted document text (surfaced under
    the `untrusted_instruction_text` field) rather than a genuine extracted
    value -- see the module docstring for how that flag is enforced.
    """

    field: str
    value: Any
    bbox: list[float]
    confidence: float
    confirmed: bool = False
    quarantined: bool = False


class ConfirmationGate:
    """In-memory holding area that locks extracted values until confirmed.

    Downstream code must never read extracted values directly -- it must go
    through `release()`, which enforces that (a) nothing non-quarantined is
    still pending and (b) quarantined items never appear in the output.
    """

    def __init__(self, items: list[ConfirmItem] | None = None) -> None:
        self._items: list[ConfirmItem] = list(items) if items else []

    def add(self, item: ConfirmItem) -> None:
        """Add a new item to the gate."""
        self._items.append(item)

    def _get(self, field: str) -> ConfirmItem:
        for item in self._items:
            if item.field == field:
                return item
        raise KeyError(f"No item with field {field!r} in this gate")

    def confirm(self, field: str) -> None:
        """Mark the item for `field` as confirmed, unchanged value."""
        self._get(field).confirmed = True

    def edit(self, field: str, new_value: Any) -> None:
        """Overwrite the value for `field` and mark it confirmed.

        Editing is itself a form of renter confirmation: the renter looked at
        the extracted value, corrected it, and is asserting the new value is
        right.
        """
        item = self._get(field)
        item.value = new_value
        item.confirmed = True

    def reject(self, field: str) -> None:
        """Drop the item for `field` from the gate entirely.

        A rejected item is removed outright rather than merely flagged, so it
        can never appear in `pending()`, `confirmed_items()`, or `release()`.
        """
        self._items = [item for item in self._items if item.field != field]

    def pending(self) -> list[ConfirmItem]:
        """Return not-yet-confirmed items that block readiness.

        Quarantined items are excluded here by design -- see module
        docstring for the rationale. An unconfirmed quarantined item is never
        "pending" in the sense of blocking `release()`.
        """
        return [item for item in self._items if not item.confirmed and not item.quarantined]

    def is_ready(self) -> bool:
        """True iff there is nothing (non-quarantined) left pending."""
        return len(self.pending()) == 0

    def confirmed_items(self) -> list[ConfirmItem]:
        """Return all confirmed items, quarantined or not."""
        return [item for item in self._items if item.confirmed]

    def release(self) -> list[dict]:
        """Return confirmed, non-quarantined values for downstream use.

        Raises `DownstreamLockedError` if any non-quarantined item is still
        pending (`is_ready()` is False). Quarantined items are always
        excluded from the returned list, even if `confirmed=True` -- they are
        never allowed to become downstream data.
        """
        if not self.is_ready():
            raise DownstreamLockedError(
                "Cannot release: one or more non-quarantined items are not yet confirmed."
            )
        return [
            {
                "field": item.field,
                "value": item.value,
                "bbox": list(item.bbox),
                "confidence": item.confidence,
            }
            for item in self._items
            if item.confirmed and not item.quarantined
        ]
