"""
SQLite-backed document registry.

Single source of truth for "what has been ingested" — the SHA-256 dedup
check (src/ingestion/sha_registry.py) reads/writes through this class.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

from src.core.config import ApplicationConfig
from src.models import Document

_lock = threading.Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    document_id   TEXT PRIMARY KEY,
    filename      TEXT NOT NULL,
    sha256        TEXT UNIQUE NOT NULL,
    total_pages   INTEGER DEFAULT 0,
    total_chunks  INTEGER DEFAULT 0,
    total_images  INTEGER DEFAULT 0,
    status        TEXT DEFAULT 'ready',
    ingested_at   REAL
);
CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(sha256);
"""


class DocumentRepository:
    def __init__(self) -> None:
        self.path = Path(ApplicationConfig().storage["sqlite"]["path"])
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def get_by_hash(self, sha256: str) -> Document | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE sha256 = ?", (sha256,)
            ).fetchone()
        return self._row_to_doc(row) if row else None

    def get(self, document_id: str) -> Document | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE document_id = ?", (document_id,)
            ).fetchone()
        return self._row_to_doc(row) if row else None

    def upsert(self, doc: Document) -> None:
        with _lock, self._conn() as conn:
            conn.execute(
                """
                INSERT INTO documents
                    (document_id, filename, sha256, total_pages, total_chunks,
                     total_images, status, ingested_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                    filename=excluded.filename, total_pages=excluded.total_pages,
                    total_chunks=excluded.total_chunks, total_images=excluded.total_images,
                    status=excluded.status
                """,
                (
                    doc.document_id, doc.filename, doc.sha256, doc.total_pages,
                    doc.total_chunks, doc.total_images, doc.status,
                    doc.ingested_at or time.time(),
                ),
            )
            conn.commit()

    def list_documents(self) -> list[Document]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM documents ORDER BY ingested_at DESC"
            ).fetchall()
        return [self._row_to_doc(r) for r in rows]

    def delete(self, document_id: str) -> None:
        with _lock, self._conn() as conn:
            conn.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
            conn.commit()

    @staticmethod
    def _row_to_doc(row: sqlite3.Row) -> Document:
        return Document(
            document_id=row["document_id"], filename=row["filename"], sha256=row["sha256"],
            total_pages=row["total_pages"], total_chunks=row["total_chunks"],
            total_images=row["total_images"], status=row["status"], ingested_at=row["ingested_at"],
        )
