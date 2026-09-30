"""OnlinePipeline — retrieve (hybrid: dense + sparse + graph) -> rerank ->
build context -> build prompt -> generate. Shared by the Agentic RAG tool
and the eval runner, so the retrieval/generation logic is written exactly once."""

from __future__ import annotations

from src.context import ContextBuilder
from src.core.config import ApplicationConfig
from src.core.logging import LoggerManager
from src.generation import PromptBuilder
from src.llm.ollama_manager import OllamaManager
from src.models import EmbeddedChunk
from src.reranking import CrossEncoderReranker
from src.retrieval import HybridRetriever

logger = LoggerManager.get_logger()


class OnlinePipeline:
    def __init__(self, retriever: HybridRetriever, reranker: CrossEncoderReranker,
                 context_builder: ContextBuilder, prompt_builder: PromptBuilder,
                 ollama_manager: OllamaManager, config: ApplicationConfig) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.context_builder = context_builder
        self.prompt_builder = prompt_builder
        self.ollama_manager = ollama_manager
        self.chat_model = config.models["chat"]["model"]
        self.rerank_top_k = config.pipeline["retrieval"]["rerank_top_k"]

    def retrieve_candidates(self, query: str) -> list[EmbeddedChunk]:
        return self.retriever.retrieve(query, top_k=15)

    def rerank_candidates(self, query: str, chunks: list[EmbeddedChunk]) -> list[EmbeddedChunk]:
        return self.reranker.rerank(query=query, chunks=chunks, top_k=self.rerank_top_k)

    def _retrieve(self, query: str) -> list[EmbeddedChunk]:
        return self.rerank_candidates(query, self.retrieve_candidates(query))

    # -- building blocks used by the Agentic RAG tool (retrieval and generation
    #    are separated so the agent can inspect/improve evidence before answering)

    def retrieve(self, query: str) -> list[EmbeddedChunk]:
        """Hybrid retrieve -> rerank. Returns chunks with `rerank_score` set."""
        return self._retrieve(query)

    def generate_from_chunks(self, query: str, chunks: list[EmbeddedChunk],
                             feedback: str = "") -> tuple[str, str]:
        context = self.context_builder.build(chunks)
        prompt = self.prompt_builder.build(query, context, feedback)
        return self.ollama_manager.generate(model=self.chat_model, prompt=prompt), context

    def stream_from_chunks(self, query: str, chunks: list[EmbeddedChunk], feedback: str = ""):
        context = self.context_builder.build(chunks)
        prompt = self.prompt_builder.build(query, context, feedback)
        stream = self.ollama_manager.generate_stream(model=self.chat_model, prompt=prompt)
        return stream, context

    def ask(self, query: str) -> str:
        answer, _, _ = self.ask_with_details(query)
        return answer

    def ask_with_details(self, query: str) -> tuple[str, list[EmbeddedChunk], str]:
        chunks = self._retrieve(query)
        context = self.context_builder.build(chunks)
        prompt = self.prompt_builder.build(query, context)
        answer = self.ollama_manager.generate(model=self.chat_model, prompt=prompt)
        return answer, chunks, context

    def ask_stream(self, query: str):
        chunks = self._retrieve(query)
        context = self.context_builder.build(chunks)
        prompt = self.prompt_builder.build(query, context)
        stream = self.ollama_manager.generate_stream(model=self.chat_model, prompt=prompt)
        return stream, chunks, context
