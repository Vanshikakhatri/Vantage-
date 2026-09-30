"""
RetrievalGrader — the "decide" step of Agentic RAG.

Plain RAG retrieves once and answers with whatever came back. Agentic RAG puts
a decision between retrieval and generation: *is this evidence actually enough
to answer the question?* If not, the agent rewrites its search and tries again
before it ever writes an answer.

Two-tier decision, cheapest first:

  1. Reranker score (free, already computed). The cross-encoder scores every
     passage against the question on 0-1. If the best passage is >=
     `strong_rerank_score`, the evidence is clearly relevant -> sufficient, and
     no LLM call is made. This is the common case, so the agent adds ~0 latency
     to easy questions.
  2. LLM judgement (only in the ambiguous / weak zone, or on a confidence-driven
     regeneration). The controller model reads short snippets of the top
     passages and returns JSON: is it sufficient, why, and — if not — a
     differently-worded search query likely to surface the missing information.

If the LLM call fails (Ollama down, bad JSON) we degrade to the reranker-only
rule, so grading can never take the pipeline down.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

from src.core.logging import LoggerManager
from src.llm.ollama_manager import OllamaManager
from src.models import EmbeddedChunk

logger = LoggerManager.get_logger()

_SNIPPET_CHARS = 350
_MAX_PASSAGES_SHOWN = 4
_MAX_QUERY_CHARS = 200

_PROMPT = """You are the retrieval controller of a document question-answering agent.
Decide whether the PASSAGES contain enough information to answer the QUESTION.
Return ONLY JSON:
{{"sufficient": true|false, "reason": "<one short sentence>", "better_query": "<string>"}}

Rules:
- sufficient=true only if the passages directly address what the question asks.
- If sufficient=false, better_query must be a DIFFERENT search query (synonyms,
  the formal/legal term, or the specific missing sub-topic) likely to find the
  missing information. Do not repeat any previous query. If sufficient=true,
  better_query must be "".
- Previous queries already tried: {tried}
{feedback_line}
QUESTION: {query}

PASSAGES:
{passages}
"""


@dataclass
class RetrievalGrade:
    sufficient: bool
    top_score: float
    reason: str
    better_query: str = ""
    used_llm: bool = False


def normalise_score(score: float) -> float:
    """Reranker output -> 0-1. The cross-encoder normally returns a sigmoid
    probability; if a raw logit leaks through, squash it the same way."""
    if 0.0 <= score <= 1.0:
        return score
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, score))))


def top_score(chunks: list[EmbeddedChunk]) -> float:
    return max((normalise_score(c.rerank_score) for c in chunks), default=0.0)


class RetrievalGrader:
    def __init__(self, ollama_manager: OllamaManager, model: str,
                 strong_score: float = 0.60, relevance_floor: float = 0.15,
                 temperature: float = 0.0) -> None:
        self.ollama = ollama_manager
        self.model = model
        self.strong_score = strong_score
        self.relevance_floor = relevance_floor
        self.temperature = temperature

    def grade(self, query: str, chunks: list[EmbeddedChunk], tried_queries: list[str],
              feedback: str = "") -> RetrievalGrade:
        top = top_score(chunks)

        # Tier 1 — clearly strong evidence: no LLM call. Skipped when we are
        # regenerating because confidence was low (the evidence we already have
        # is then the prime suspect, so it deserves a second look).
        if chunks and top >= self.strong_score and not feedback:
            return RetrievalGrade(True, top, f"Top passage is a strong match ({top:.0%}).")

        # Tier 2 — LLM judgement.
        try:
            raw = self.ollama.generate(
                model=self.model, prompt=self._prompt(query, chunks, tried_queries, feedback),
                temperature=self.temperature, format="json",
            )
            data = json.loads(raw)
            sufficient = bool(data.get("sufficient", False))
            better = "" if sufficient else self._clean_query(data.get("better_query"), tried_queries)
            return RetrievalGrade(sufficient, top, str(data.get("reason", "")).strip()[:200],
                                  better, used_llm=True)
        except Exception as exc:
            logger.warning(f"RetrievalGrader LLM call failed ({exc}); using reranker-only rule.")
            ok = bool(chunks) and top >= self.relevance_floor
            return RetrievalGrade(ok, top, "Grader unavailable — judged on reranker score alone.")

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _clean_query(candidate, tried: list[str]) -> str:
        if not isinstance(candidate, str):
            return ""
        candidate = candidate.strip().strip('"').strip()[:_MAX_QUERY_CHARS]
        seen = {t.strip().lower() for t in tried}
        return "" if not candidate or candidate.lower() in seen else candidate

    @staticmethod
    def _prompt(query: str, chunks: list[EmbeddedChunk], tried: list[str], feedback: str) -> str:
        if chunks:
            passages = "\n".join(
                f"[P{i}] ({c.chunk.source}, p.{c.chunk.page}) "
                f"{' '.join(c.chunk.text.split())[:_SNIPPET_CHARS]}"
                for i, c in enumerate(chunks[:_MAX_PASSAGES_SHOWN], start=1)
            )
        else:
            passages = "(nothing was retrieved)"
        feedback_line = (f"- A previous answer built on this evidence was judged weak: {feedback}\n"
                         if feedback else "")
        return _PROMPT.format(query=query, passages=passages, feedback_line=feedback_line,
                              tried=json.dumps(tried))
