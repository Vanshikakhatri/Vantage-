"""PipelineFactory — builds a fully wired OnlinePipeline (dense + sparse +
graph retrievers, reranker, context/prompt builders, Ollama client) from config."""

from __future__ import annotations

from src.context import ContextBuilder
from src.core.config import ApplicationConfig
from src.embeddings import EmbeddingManager
from src.generation import PromptBuilder
from src.graph.graph_retriever import GraphRetriever
from src.llm.ollama_manager import OllamaManager
from src.pipeline.online_pipeline import OnlinePipeline
from src.reranking import CrossEncoderReranker
from src.retrieval import BM25Manager, DenseRetriever, FAISSManager, HybridRetriever, SparseRetriever


class PipelineFactory:
    @staticmethod
    def create() -> OnlinePipeline:
        config = ApplicationConfig()

        embedding_manager = EmbeddingManager.get(
            model_name=config.models["embeddings"]["model"],
            device=config.models["embeddings"].get("device"),
        )

        faiss_manager = FAISSManager(
            dimension=config.models["embeddings"]["dimension"],
            index_path=config.storage["faiss"]["index_path"],
            metadata_path=config.storage["faiss"]["metadata_path"],
        )
        faiss_manager.load()

        bm25_manager = BM25Manager(index_path=config.storage["bm25"]["index_path"])
        bm25_manager.load()

        dense = DenseRetriever(embedding_manager, faiss_manager)
        sparse = SparseRetriever(bm25_manager)
        graph = GraphRetriever()
        hybrid = HybridRetriever(dense, sparse, graph)

        reranker = CrossEncoderReranker.get(model_name=config.models["reranker"]["model"])
        context_builder = ContextBuilder()
        prompt_builder = PromptBuilder()
        ollama_manager = OllamaManager(host=config.models["ollama"]["host"])

        return OnlinePipeline(
            retriever=hybrid, reranker=reranker, context_builder=context_builder,
            prompt_builder=prompt_builder, ollama_manager=ollama_manager, config=config,
        )
