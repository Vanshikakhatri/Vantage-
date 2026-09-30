"""SparseRetriever — thin wrapper over BM25Manager, kept as its own class
so HybridRetriever composes two equal-shaped retriever interfaces."""

from __future__ import annotations

from src.models import EmbeddedChunk
from src.retrieval.bm25_manager import BM25Manager


class SparseRetriever:
    def __init__(self, bm25_manager: BM25Manager) -> None:
        self.bm25_manager = bm25_manager

    def retrieve(self, query: str, top_k: int = 15) -> list[EmbeddedChunk]:
        return self.bm25_manager.search(query, top_k=top_k)
