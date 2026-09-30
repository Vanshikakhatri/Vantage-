"""File-level utilities — SHA-256 hashing used for document dedup."""

from __future__ import annotations

import hashlib
from pathlib import Path


def calculate_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Stream the file in 1MB chunks so large PDFs don't blow up memory."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()
