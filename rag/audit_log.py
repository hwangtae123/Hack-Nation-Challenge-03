"""Privacy-safe audit log: consent, actions, and rule versions only.

The challenge's CONSENT AND CORRECTION requirement is to log *that* an action
happened and *which* rule version applied -- never raw document contents,
extracted field values, or renter-entered free text. Every call site in this
codebase must pass only identifiers, counts, and codes to `log_event`.

The log is an append-only JSON-lines file under ``uploads/`` (the same
ephemeral, git-ignored area as session uploads). Session deletion
(`/api/session/delete`) removes uploaded documents, not this log -- the log
never contains document content, so retaining "an action occurred" records
across a content deletion is intentional, not a leak.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from rag import config

LOG_PATH = config.RAG_DIR / "uploads" / "audit.jsonl"

_logger = logging.getLogger("realdoor.audit")
_logger.setLevel(logging.INFO)
if not _logger.handlers:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    _handler.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(_handler)
    _logger.propagate = False


def log_event(event: str, session_id: str | None = None, **fields: Any) -> None:
    """Append one audit record: an action, its session, and the rule version in
    force. ``fields`` must be identifiers, counts, or codes -- never raw
    document text, extracted values, or user-entered free text.
    """
    record = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "event": event,
        "session_id": session_id,
        "rule_year": config.RULE_YEAR,
        "rule_effective_date": config.EFFECTIVE_DATE,
        **fields,
    }
    _logger.info(json.dumps(record, ensure_ascii=False))
