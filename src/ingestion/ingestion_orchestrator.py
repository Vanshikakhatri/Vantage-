"""
IngestionOrchestrator — the full offline path for a single document.

Steps (in order):
  1. SHA-256 dedup check (ShaRegistry) — abort early if already indexed.
  2. Parse the PDF page-by-page with PyMuPDF: pull extractable text, and
     separately pull embedded raster images + a full-page render for pages
     with little/no extractable text (i.e. scans).
  3. VLM captioning (ImageCaptioner) for every image/scanned page — output
     becomes its own text, tagged modality="image_caption".
  4. Chunking (ChunkingService) — both text and image captions go through
     the same recursive splitter.
  5. GraphRAG entity/relation extraction (GraphBuilder) per chunk.
  6. Embedding (EmbeddingManager, bge-m3) per chunk.
  7. Write to FAISS + BM25 + graph + SQLite registry.
"""

from __future__ import annotations

import io
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

from src.core.config import ApplicationConfig
from src.core.logging import LoggerManager
from src.embeddings import EmbeddingManager
from src.graph.graph_builder import GraphBuilder
from src.ingestion.sha_registry import ShaRegistry
from src.models import Document
from src.preprocessing.chunking import ChunkingService
from src.retrieval.bm25_manager import BM25Manager
from src.retrieval.faiss_manager import FAISSManager
from src.storage.repository import DocumentRepository
from src.vlm import ImageCaptioner

logger = LoggerManager.get_logger()

_MIN_TEXT_CHARS_PER_PAGE = 40   # below this, treat the page as a scan and VLM-caption it


class IngestionOrchestrator:
    def __init__(self, repository: DocumentRepository, faiss_manager: FAISSManager,
                 bm25_manager: BM25Manager, embedding_manager: EmbeddingManager,
                 graph_builder: GraphBuilder, image_captioner: ImageCaptioner,
                 chunker: ChunkingService | None = None) -> None:
        self.repository = repository
        self.faiss_manager = faiss_manager
        self.bm25_manager = bm25_manager
        self.embedding_manager = embedding_manager
        self.graph_builder = graph_builder
        self.image_captioner = image_captioner
        self.chunker = chunker or ChunkingService()
        self.sha_registry = ShaRegistry(repository)

    def ingest_document(self, file_path: Path) -> Document:
        digest, existing = self.sha_registry.check(file_path)
        if existing:
            raise FileExistsError(f"Document already indexed: {existing.document_id}")

        document_id = digest[:16]
        pdf = fitz.open(file_path)
        all_chunks = []
        total_images = 0

        for page_index in range(len(pdf)):
            page = pdf[page_index]
            page_num = page_index + 1
            text = page.get_text("text")

            if len(text.strip()) >= _MIN_TEXT_CHARS_PER_PAGE:
                all_chunks += self.chunker.make_chunks(
                    document_id, file_path.name, page_num, text, modality="text"
                )
            else:
                caption = self._caption_page(page)
                if caption:
                    total_images += 1
                    all_chunks += self.chunker.make_chunks(
                        document_id, file_path.name, page_num, caption, modality="image_caption"
                    )

            for image_info in page.get_images(full=True):
                caption = self._caption_embedded_image(pdf, image_info[0])
                if caption:
                    total_images += 1
                    all_chunks += self.chunker.make_chunks(
                        document_id, file_path.name, page_num, caption, modality="image_caption"
                    )

        pdf.close()

        if not all_chunks:
            logger.warning(f"No extractable content in '{file_path.name}'.")

        for chunk in all_chunks:
            chunk.entities = self.graph_builder.add_chunk(chunk)
        self.graph_builder.save()

        if all_chunks:
            texts = [c.text for c in all_chunks]
            embeddings = self.embedding_manager.embed_batch(texts)
            self.faiss_manager.add(embeddings, all_chunks)
            self.bm25_manager.add(all_chunks)

        doc = Document(
            document_id=document_id, filename=file_path.name, sha256=digest,
            total_pages=len(pdf) if pdf else 0, total_chunks=len(all_chunks),
            total_images=total_images, status="ready",
        )
        self.repository.upsert(doc)
        logger.info(f"Ingested '{file_path.name}': {doc.total_chunks} chunks, "
                    f"{doc.total_images} captioned images/pages.")
        return doc

    def _caption_page(self, page) -> str:
        try:
            pix = page.get_pixmap(dpi=150)
            image = Image.open(io.BytesIO(pix.tobytes("png")))
            return self.image_captioner.caption(image)
        except Exception as exc:
            logger.warning(f"Page render/caption failed: {exc}")
            return ""

    def _caption_embedded_image(self, pdf, xref: int) -> str:
        try:
            base_image = pdf.extract_image(xref)
            image = Image.open(io.BytesIO(base_image["image"]))
            if image.width < 64 or image.height < 64:
                return ""  # skip tiny icons/decorations
            return self.image_captioner.caption(image)
        except Exception as exc:
            logger.debug(f"Embedded image caption skipped: {exc}")
            return ""
