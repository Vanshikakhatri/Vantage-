"""IngestionFactory — builds a fully wired IngestionOrchestrator from config."""

from __future__ import annotations

from src.core.config import ApplicationConfig
from src.embeddings import EmbeddingManager
from src.graph.graph_builder import GraphBuilder
from src.ingestion.ingestion_orchestrator import IngestionOrchestrator
from src.llm.ollama_manager import OllamaManager
from src.preprocessing.chunking import ChunkingService
from src.retrieval.bm25_manager import BM25Manager
from src.retrieval.faiss_manager import FAISSManager
from src.storage.repository import DocumentRepository
from src.vlm import ImageCaptioner


class IngestionFactory:
    @staticmethod
    def create() -> IngestionOrchestrator:
        config = ApplicationConfig()

        repository = DocumentRepository()

        faiss_manager = FAISSManager(
            dimension=config.models["embeddings"]["dimension"],
            index_path=config.storage["faiss"]["index_path"],
            metadata_path=config.storage["faiss"]["metadata_path"],
        )
        faiss_manager.load()

        bm25_manager = BM25Manager(index_path=config.storage["bm25"]["index_path"])
        bm25_manager.load()

        embedding_manager = EmbeddingManager.get(
            model_name=config.models["embeddings"]["model"],
            device=config.models["embeddings"].get("device"),
        )

        ollama_manager = OllamaManager(host=config.models["ollama"]["host"])
        graph_builder = GraphBuilder(ollama_manager)
        image_captioner = ImageCaptioner(ollama_manager)
        chunker = ChunkingService()

        return IngestionOrchestrator(
            repository=repository, faiss_manager=faiss_manager, bm25_manager=bm25_manager,
            embedding_manager=embedding_manager, graph_builder=graph_builder,
            image_captioner=image_captioner, chunker=chunker,
        )
