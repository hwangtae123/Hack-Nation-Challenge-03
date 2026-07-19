"""At-rest encryption for a renter's uploaded documents.

Bundled demo documents under the organizer's ``synthetic_documents/`` pack are
public fixtures already checked into the repo in plain form; encrypting a copy
of already-public repo content protects nothing, so they are left alone. A
renter's own upload is the actual "persisted data" the challenge's PRIVACY AND
SECURITY requirement targets: it is encrypted before every write to disk and
only ever decrypted into a short-lived temp file for the duration of one
request (see ``decrypted_copy``), never left on disk in plaintext.

This intentionally does not touch the ``src/`` extraction pipeline: it still
just opens an ordinary filesystem path, which happens to be a temp file that
outlives a single ``with`` block instead of a permanent upload path.
"""
from __future__ import annotations

import shutil
import tempfile
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Iterator

from cryptography.fernet import Fernet, InvalidToken

from rag import config

KEY_ENV_VAR = "UPLOAD_ENCRYPTION_KEY"


class DecryptionError(Exception):
    """Raised when an encrypted upload can't be read with the current key.

    This is expected, not a bug, when no UPLOAD_ENCRYPTION_KEY is configured
    and the server process has restarted since the file was written (a fresh
    in-memory key is generated per process in that case -- see ``_fernet``).
    """


@lru_cache(maxsize=1)
def _fernet() -> Fernet:
    """Load the upload-encryption key from the environment, or generate one.

    Set UPLOAD_ENCRYPTION_KEY (a urlsafe-base64 Fernet key, e.g. the output of
    ``Fernet.generate_key()``) in production so encrypted uploads stay
    readable across restarts. Without it, a key is generated in memory for
    this process only -- acceptable because uploads are already ephemeral,
    session-scoped, and renter-deletable; losing the key on restart just means
    any not-yet-processed upload becomes unreadable, the same practical effect
    as the renter deleting the session, not a new data-loss mode.
    """
    key = config.get_env(KEY_ENV_VAR)
    if not key:
        key = Fernet.generate_key().decode()
    return Fernet(key.encode() if isinstance(key, str) else key)


def encrypt_bytes(data: bytes) -> bytes:
    """Encrypt raw bytes for storage on disk."""
    return _fernet().encrypt(data)


def decrypt_bytes(data: bytes) -> bytes:
    """Decrypt bytes previously produced by ``encrypt_bytes``."""
    try:
        return _fernet().decrypt(data)
    except InvalidToken as exc:
        raise DecryptionError(
            "This upload can't be decrypted with the current key (likely the "
            "server restarted without a persistent UPLOAD_ENCRYPTION_KEY). "
            "Please re-upload the document."
        ) from exc


@contextmanager
def decrypted_copy(encrypted_path: Path, original_name: str) -> Iterator[Path]:
    """Decrypt ``encrypted_path`` into a plaintext temp file named
    ``original_name`` (so filename-based document-type inference still works)
    for the duration of the ``with`` block only. The temp file and its
    directory are removed on exit, success or failure alike.
    """
    plaintext = decrypt_bytes(encrypted_path.read_bytes())
    tmp_dir = Path(tempfile.mkdtemp(prefix="realdoor-upload-"))
    tmp_path = tmp_dir / original_name
    try:
        tmp_path.write_bytes(plaintext)
        yield tmp_path
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
