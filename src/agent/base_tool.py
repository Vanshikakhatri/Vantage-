"""BaseTool — common interface + result shape every agent tool implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from src.pipeline.pipeline_result import AnswerSource


@dataclass
class ToolResult:
    answer: str
    sources: list[str] = field(default_factory=list)
    context: str = ""                      # what was fed to the LLM (for validation/reflection)
    answer_source: str = AnswerSource.LLM
    retrieved_chunks: list[dict] = field(default_factory=list)   # for the citation-card UI
    agent_steps: list[str] = field(default_factory=list)         # e.g. "Re-searched with: ..."


def with_feedback(query: str, feedback: str) -> str:
    """Appends reviewer feedback to a generation prompt. Used by tools that have no
    retrieval step (chat/code). Feedback is passed separately from `query` so it never
    pollutes anything derived from the query (e.g. a retrieval search string)."""
    if not feedback:
        return query
    return (f"{query}\n\n[Reviewer feedback on your previous attempt: {feedback} "
            f"Write an improved answer that fixes these problems.]")


class BaseTool(ABC):
    name: str

    @abstractmethod
    def execute(self, query: str, feedback: str = "") -> ToolResult: ...

    @abstractmethod
    def execute_stream(self, query: str, feedback: str = ""):
        """Yields {'type': 'chunk'|'status', ...} events, ending with
        {'type': 'result', 'data': ToolResult}.

        `feedback` is non-empty only on a confidence-triggered regeneration and
        describes what the validator found lacking in the previous attempt."""
        ...
