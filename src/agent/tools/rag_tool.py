"""
RAGTool — Agentic RAG. GraphRAG + hybrid retrieval + reranking live behind it,
but instead of "retrieve once, answer with whatever came back" the tool runs a
small decision loop:

    retrieve (hybrid + graph -> rerank)
        -> GRADE the evidence            (RetrievalGrader: reranker score, then LLM
                                          only when the evidence is ambiguous/weak)
        -> weak? REWRITE the search and retrieve again   (bounded: max_search_rounds)
        -> still nothing relevant?  DECIDE what to do:
              strict=True  (RAG mode)  -> say "not found in your documents" — never
                                          answer from outside knowledge
              strict=False (Auto mode) -> fall back to the chat model, clearly
                                          labelled RAG_FALLBACK (previous behaviour)
        -> otherwise generate a cited answer from the best evidence.

Retrieval and generation always use the *user's* question for the final answer;
refined queries are only used to widen what is retrieved, and every merged pool
is re-ranked against the original question so a rewrite can't drag in passages
that answer a different question.

On a confidence-triggered regeneration the reflector passes `feedback`; the tool
then re-examines its evidence (one extra search round at most) and feeds the
feedback to the generator, rather than blindly re-running the same prompt.
"""

from __future__ import annotations

from src.agent.base_tool import BaseTool, ToolResult, with_feedback
from src.agent.retrieval_grader import RetrievalGrader, normalise_score, top_score
from src.llm.ollama_manager import OllamaManager
from src.models import EmbeddedChunk
from src.pipeline.online_pipeline import OnlinePipeline
from src.pipeline.pipeline_result import AnswerSource

_NOT_FOUND_MESSAGE = (
    "I couldn't find anything relevant to this question in your uploaded documents, "
    "so I won't guess. Try rephrasing the question, or upload the document that covers it."
)

_FALLBACK_PROMPT = (
    "The document corpus has no relevant information for this question. Answer "
    "briefly from general knowledge, and say clearly that this is not from the "
    "uploaded documents.\n\n{query}"
)


def _serialize_chunks(chunks: list[EmbeddedChunk]) -> list[dict]:
    """Flattens retrieved chunks + their per-signal scores for the citation-card
    UI (retrieval display: dense/BM25/rerank score, source doc, page, text)."""
    return [
        {
            "chunk_id": c.chunk.chunk_id,
            "source": c.chunk.source,
            "page": c.chunk.page,
            "text": c.chunk.text,
            "modality": c.chunk.modality,
            "dense_score": round(c.dense_score, 4),
            "bm25_score": round(c.bm25_score, 4),
            "rerank_score": round(c.rerank_score, 4),
        }
        for c in chunks
    ]


class RAGTool(BaseTool):
    name = "rag"

    def __init__(self, online_pipeline: OnlinePipeline, ollama_manager: OllamaManager,
                 fallback_model: str, grader: RetrievalGrader, max_search_rounds: int = 2,
                 relevance_floor: float = 0.15, strict: bool = False) -> None:
        self.pipeline = online_pipeline
        self.ollama = ollama_manager
        self.fallback_model = fallback_model
        self.grader = grader
        self.max_search_rounds = max_search_rounds
        self.relevance_floor = relevance_floor
        self.strict = strict

    # ------------------------------------------------------------------ public

    def execute(self, query: str, feedback: str = "") -> ToolResult:
        result = None
        for event in self._run(query, feedback):
            if event["type"] == "result":
                result = event["data"]
        return result

    def execute_stream(self, query: str, feedback: str = ""):
        yield from self._run(query, feedback)

    # ----------------------------------------------------------------- the loop

    def _run(self, query: str, feedback: str):
        steps: list[str] = []
        tried = [query]

        yield {"type": "status", "message": "Searching your documents..."}
        chunks = self.pipeline.retrieve(query)
        steps.append(f"Searched documents — best passage match {top_score(chunks):.0%} "
                     f"({len(chunks)} passage{'s' if len(chunks) != 1 else ''}).")

        # A regeneration gets one extra search round at most: it is a second look
        # at the evidence, not a fresh open-ended research session.
        max_rounds = min(1, self.max_search_rounds) if feedback else self.max_search_rounds

        for round_no in range(max_rounds + 1):
            yield {"type": "status", "message": "Checking the retrieved passages..."}
            grade = self.grader.grade(query, chunks, tried, feedback)
            if grade.used_llm:
                steps.append(f"Evidence check: {'enough' if grade.sufficient else 'not enough'}"
                             f"{' — ' + grade.reason if grade.reason else ''}")
            if grade.sufficient or round_no == max_rounds or not grade.better_query:
                break

            tried.append(grade.better_query)
            yield {"type": "status",
                   "message": f"Evidence looks weak — searching again ({round_no + 1}/{max_rounds})..."}
            steps.append(f"Re-searched with a refined query: “{grade.better_query}”")
            chunks = self._merge(query, chunks, self.pipeline.retrieve(grade.better_query))
            steps.append(f"After re-search — best passage match {top_score(chunks):.0%}.")

        if not chunks or top_score(chunks) < self.relevance_floor:
            yield from self._no_evidence(query, feedback, steps)
            return

        stream, context = self.pipeline.stream_from_chunks(query, chunks, feedback)
        full = ""
        for piece in stream:
            full += piece
            yield {"type": "chunk", "text": piece}
        yield {"type": "result", "data": ToolResult(
            answer=full, sources=sorted({c.chunk.source for c in chunks}), context=context,
            answer_source=AnswerSource.RAG, retrieved_chunks=_serialize_chunks(chunks),
            agent_steps=steps,
        )}

    def _no_evidence(self, query: str, feedback: str, steps: list[str]):
        if self.strict:
            steps.append("No relevant passages found — declined to answer from outside the documents.")
            yield {"type": "chunk", "text": _NOT_FOUND_MESSAGE}
            yield {"type": "result", "data": ToolResult(
                answer=_NOT_FOUND_MESSAGE, answer_source=AnswerSource.RAG_NOT_FOUND,
                agent_steps=steps)}
            return

        steps.append("No relevant passages found — answered from general knowledge instead.")
        prompt = with_feedback(_FALLBACK_PROMPT.format(query=query), feedback)
        full = ""
        for piece in self.ollama.generate_stream(model=self.fallback_model, prompt=prompt):
            full += piece
            yield {"type": "chunk", "text": piece}
        yield {"type": "result", "data": ToolResult(
            answer=full, answer_source=AnswerSource.RAG_FALLBACK, agent_steps=steps)}

    def _merge(self, query: str, current: list[EmbeddedChunk],
               extra: list[EmbeddedChunk]) -> list[EmbeddedChunk]:
        """Union of two retrievals, de-duplicated by chunk id, re-ranked against the
        ORIGINAL question so a reworded search can't promote off-topic passages."""
        pool: dict[str, EmbeddedChunk] = {}
        for c in [*current, *extra]:
            cid = c.chunk.chunk_id
            if cid not in pool or normalise_score(c.rerank_score) > normalise_score(pool[cid].rerank_score):
                pool[cid] = c
        merged = list(pool.values())
        try:
            return self.pipeline.rerank_candidates(query, merged)
        except Exception:
            return sorted(merged, key=lambda c: normalise_score(c.rerank_score),
                          reverse=True)[:self.pipeline.rerank_top_k]
