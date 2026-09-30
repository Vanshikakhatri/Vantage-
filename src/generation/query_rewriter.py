"""
QueryRewriter — expands vague/incomplete/follow-up queries into
self-contained, retrieval-optimised queries.

WHERE IT SITS: after the input guardrail stack passes a query, before the
tool-selection step. Runs on a tiny model (qwen2.5:1.5b) since it's a
cheap rewriting task, not reasoning — keeps latency low on a small GPU.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.core.config import ApplicationConfig
from src.llm.ollama_manager import OllamaManager

_PROMPT = """Rewrite the user query below into a single, self-contained, precise
search query suitable for a document retrieval system. Expand abbreviations,
resolve vague pronouns using nothing but the query itself, and keep it under
40 words. If the query is already clear and self-contained, return it
UNCHANGED. Return ONLY the rewritten query text, nothing else.

QUERY: {query}
REWRITTEN:"""


@dataclass
class RewriteResult:
    original_query: str
    rewritten_query: str
    was_rewritten: bool


class QueryRewriter:
    def __init__(self, ollama_manager: OllamaManager, model: str | None = None) -> None:
        self.ollama = ollama_manager
        self.model = model or ApplicationConfig().models["query_rewriter"]["model"]

    def rewrite(self, query: str) -> RewriteResult:
        try:
            raw = self.ollama.generate(model=self.model, prompt=_PROMPT.format(query=query),
                                        temperature=0.0).strip()
        except Exception:
            return RewriteResult(query, query, False)

        rewritten = raw.strip().strip('"')
        was_rewritten = bool(rewritten) and rewritten.lower() != query.strip().lower()
        return RewriteResult(query, rewritten if was_rewritten else query, was_rewritten)
