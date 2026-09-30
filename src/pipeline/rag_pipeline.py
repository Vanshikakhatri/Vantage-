"""
RAGPipeline — unified singleton facade. This is the single object backend.py
talks to. Owns:
  - the offline ingestion path (IngestionOrchestrator via IngestionFactory)
  - the online query path      (OnlinePipeline via PipelineFactory)
  - the agentic orchestration  (AgenticPipeline, built lazily)

After every successful ingestion, FAISS + BM25 + graph are hot-reloaded so
newly added documents are searchable immediately, no restart needed.
"""

from __future__ import annotations

import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path

from src.core.config import ApplicationConfig
from src.core.logging import LoggerManager
from src.ingestion.ingestion_factory import IngestionFactory
from src.models import Document
from src.pipeline.pipeline_factory import PipelineFactory
from src.pipeline.pipeline_result import PipelineResult
from src.security.audit_logger import AuditLogger
from src.utils import calculate_sha256

logger = LoggerManager.get_logger()
_lock = threading.Lock()


@dataclass
class IngestResult:
    document_id: str
    filename: str
    status: str
    total_pages: int = 0
    total_chunks: int = 0
    total_images: int = 0
    duplicate: bool = False


@dataclass
class QueryResult:
    answer: str
    sources: list[str] = field(default_factory=list)
    latency_ms: int = 0


class RAGPipeline:
    _instance: "RAGPipeline | None" = None

    @classmethod
    def get(cls) -> "RAGPipeline":
        if cls._instance is None:
            with _lock:
                if cls._instance is None:
                    cls._instance = cls._create()
        return cls._instance

    @classmethod
    def _create(cls) -> "RAGPipeline":
        logger.info("Initialising Vantage RAGPipeline...")
        instance = cls()
        instance._config = ApplicationConfig()
        instance._ingestion = IngestionFactory.create()
        instance._online = PipelineFactory.create()
        instance._agentic = None
        instance._audit = AuditLogger()
        logger.info("Vantage RAGPipeline ready.")
        return instance

    # ---------------------------------------------------------------- ingestion

    def ingest(self, file_bytes: bytes, filename: str) -> IngestResult:
        logger.info(f"Ingesting document: {filename}")
        tmp_dir = Path(tempfile.mkdtemp())
        tmp_path = tmp_dir / filename

        try:
            tmp_path.write_bytes(file_bytes)
            digest = calculate_sha256(tmp_path)
            self._ingestion.ingest_document(tmp_path)

        except FileExistsError:
            existing = self._ingestion.repository.get_by_hash(digest)
            self._audit.log({"event": "duplicate_upload_blocked", "filename": filename,
                              "sha256": digest, "existing_document_id":
                              existing.document_id if existing else None})
            self._cleanup(tmp_path, tmp_dir)
            if existing:
                return IngestResult(existing.document_id, existing.filename, "ready",
                                     existing.total_pages, existing.total_chunks,
                                     existing.total_images, duplicate=True)
            return IngestResult("unknown", filename, "ready", duplicate=True)

        except Exception as exc:
            logger.error(f"Ingestion failed for {filename}: {exc}")
            self._cleanup(tmp_path, tmp_dir)
            return IngestResult("", filename, "failed")

        self._cleanup(tmp_path, tmp_dir)
        self._reload_indices()

        doc = self._ingestion.repository.get_by_hash(digest)
        self._audit.log({"event": "document_ingested", "filename": filename, "sha256": digest,
                          "document_id": doc.document_id if doc else None,
                          "total_chunks": doc.total_chunks if doc else 0})

        if doc:
            return IngestResult(doc.document_id, doc.filename, "ready",
                                 doc.total_pages, doc.total_chunks, doc.total_images)
        return IngestResult("", filename, "ready")

    @staticmethod
    def _cleanup(tmp_path: Path, tmp_dir: Path) -> None:
        tmp_path.unlink(missing_ok=True)
        try:
            tmp_dir.rmdir()
        except Exception:
            pass

    # ------------------------------------------------------------------ query

    def query(self, question: str) -> QueryResult:
        import time
        start = time.perf_counter()
        answer, chunks, _ = self._online.ask_with_details(question)
        latency_ms = int((time.perf_counter() - start) * 1000)
        sources = sorted({c.chunk.source for c in chunks})
        return QueryResult(answer=answer, sources=sources, latency_ms=latency_ms)

    # --------------------------------------------------------- document mgmt

    def list_documents(self) -> list[Document]:
        return self._ingestion.repository.list_documents()

    def delete_document(self, document_id: str) -> None:
        logger.info(f"Deleting document: {document_id}")
        self._ingestion.repository.delete(document_id)
        try:
            self._online.retriever.dense.faiss_manager.delete_by_document_id(document_id)
            self._online.retriever.sparse.bm25_manager.delete_by_document_id(document_id)
            self._ingestion.graph_builder.remove_document(document_id)
            self._ingestion.graph_builder.save()
        except Exception as exc:
            logger.error(f"Failed to delete vectors/graph for {document_id}: {exc}")
        self._audit.log({"event": "document_deleted", "document_id": document_id})

    # --------------------------------------------------------------- agentic

    def _get_agentic(self):
        if self._agentic is None:
            from src.pipeline.agentic_pipeline import AgenticPipeline
            self._agentic = AgenticPipeline.create(self._online)
        return self._agentic

    def agentic_query(self, question: str, direct_rag: bool = False,
                      rag_only: bool = False) -> PipelineResult:
        """direct_rag = Fast mode (skip validation/regeneration).
        rag_only    = docs-only Agentic RAG (never answers from outside the documents)."""
        return self._get_agentic().run(question, direct_rag=direct_rag, rag_only=rag_only)

    def agentic_query_stream(self, question: str, direct_rag: bool = False,
                             rag_only: bool = False):
        return self._get_agentic().run_stream(question, direct_rag=direct_rag, rag_only=rag_only)

    def audit_chain_status(self) -> tuple[bool, int]:
        return self._audit.verify_chain()

    # ------------------------------------------------------------ private

    def _reload_indices(self) -> None:
        logger.info("Hot-reloading retrieval indices...")
        try:
            self._online.retriever.dense.faiss_manager.load()
            self._online.retriever.sparse.bm25_manager.load()
            self._online.retriever.graph.reload()
            logger.info("Retrieval indices reloaded successfully.")
        except Exception as exc:
            logger.warning(f"Index hot-reload failed (query will use stale index): {exc}")
