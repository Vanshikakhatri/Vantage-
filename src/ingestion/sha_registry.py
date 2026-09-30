"""
SHA-256 Document Registry — dedup gate for ingestion.

WHERE IT SITS: first thing IngestionOrchestrator does with a new upload,
before any parsing/chunking/embedding cost is paid.

Every uploaded file is hashed (whole-file SHA-256). If that hash already
exists in the DocumentRepository, ingestion is skipped entirely and the
existing Document record is returned — this is what makes re-uploading the
same NIST SP 800-53 PDF five times a no-op instead of five duplicate index
entries (and is itself an audit-loggable event: "duplicate upload detected").
"""

from __future__ import annotations

from pathlib import Path

from src.core.logging import LoggerManager
from src.models import Document
from src.storage.repository import DocumentRepository
from src.utils import calculate_sha256

logger = LoggerManager.get_logger()


class ShaRegistry:
    def __init__(self, repository: DocumentRepository | None = None) -> None:
        self.repository = repository or DocumentRepository()

    def check(self, file_path: Path) -> tuple[str, Document | None]:
        """
        Hash the file and check the registry.
        Returns (sha256, existing_document_or_None).
        """
        digest = calculate_sha256(file_path)
        existing = self.repository.get_by_hash(digest)
        if existing:
            logger.info(f"SHA match — '{file_path.name}' already indexed as "
                        f"document_id={existing.document_id} (sha256={digest[:12]}...)")
        return digest, existing
