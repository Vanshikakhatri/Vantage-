"""DenseRetriever — embeds the query with bge-m3 and searches FAISS."""

from __future__ import annotations

from src.embeddings import EmbeddingManager
from src.models import EmbeddedChunk
from src.retrieval.faiss_manager import FAISSManager


class DenseRetriever:
    def __init__(self, embedding_manager: EmbeddingManager, faiss_manager: FAISSManager) -> None:
        self.embedding_manager = embedding_manager
        self.faiss_manager = faiss_manager

    def retrieve(self, query: str, top_k: int = 15) -> list[EmbeddedChunk]:
        query_embedding = self.embedding_manager.embed_text(query)
        return self.faiss_manager.search(query_embedding, top_k=top_k)
