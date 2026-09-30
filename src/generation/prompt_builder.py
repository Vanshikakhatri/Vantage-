"""PromptBuilder — assembles the final chat-model prompt from query + context.
Instructs the model to answer only from context, cite [S#] tags, and say so
explicitly when the context doesn't cover the question (this instruction is
what keeps the hallucination rate measurable/low in the eval report)."""

from __future__ import annotations

_TEMPLATE = """You are Vantage, a compliance-grade document assistant. Answer the
user's question using ONLY the numbered source blocks below. Cite every claim
with its [S#] tag. If the sources do not contain the answer, say so plainly —
never guess or use outside knowledge.

SOURCES:
{context}
{feedback_block}
QUESTION:
{query}

ANSWER (cite [S#] inline):"""

_FEEDBACK_BLOCK = """
REVIEWER FEEDBACK ON YOUR PREVIOUS ATTEMPT (fix these, still using ONLY the sources):
{feedback}
"""


class PromptBuilder:
    def build(self, query: str, context: str, feedback: str = "") -> str:
        block = _FEEDBACK_BLOCK.format(feedback=feedback) if feedback else ""
        return _TEMPLATE.format(context=context, query=query, feedback_block=block)
