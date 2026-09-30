"""
Audit Logger — append-only, hash-chained JSONL trail.

Every query goes through here exactly once, at the very end of the pipeline
(request in, every guardrail verdict, retrieval sources, validation score,
reflection iterations, final answer hash). Each line embeds the SHA-256 of
the previous line, so any later tampering/deletion breaks the chain and is
detectable — the property regulators actually check for (DPDP Act Sec. 8
"reasonable security safeguards", NIST SP 800-53 AU family).

We store a HASH of the raw answer/query text, not the plaintext, in the
audit record's `answer_hash` / `query_hash` fields, and only the (already
PII-masked) sanitized text — so the audit log itself never becomes a new
place where personal data leaks.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

from src.core.config import ApplicationConfig

_lock = threading.Lock()


class AuditLogger:
    def __init__(self) -> None:
        self.path = Path(ApplicationConfig().storage["audit_log"]["path"])
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _last_hash(self) -> str:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return "0" * 64
        with open(self.path, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            block = min(4096, size)
            fh.seek(-block, 2)
            last_line = fh.read().splitlines()[-1]
        return json.loads(last_line)["record_hash"]

    def log(self, event: dict[str, Any]) -> str:
        """Append a hash-chained audit record. Returns this record's hash."""
        with _lock:
            prev_hash = self._last_hash()
            record = {
                "timestamp": time.time(),
                "iso_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "prev_hash": prev_hash,
                **event,
            }
            record_str = json.dumps(record, sort_keys=True, default=str)
            record_hash = hashlib.sha256((prev_hash + record_str).encode()).hexdigest()
            record["record_hash"] = record_hash

            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, default=str) + "\n")

            return record_hash

    def verify_chain(self) -> tuple[bool, int]:
        """Walk the whole log and verify hash-chain integrity.
        Returns (is_valid, number_of_records_checked)."""
        if not self.path.exists():
            return True, 0

        prev_hash = "0" * 64
        count = 0
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                record = json.loads(line)
                claimed_hash = record.pop("record_hash")
                record_str = json.dumps(
                    {**record, "prev_hash": prev_hash}, sort_keys=True, default=str
                )
                expected = hashlib.sha256((prev_hash + record_str).encode()).hexdigest()
                if expected != claimed_hash:
                    return False, count
                prev_hash = claimed_hash
                count += 1
        return True, count

    @staticmethod
    def text_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
