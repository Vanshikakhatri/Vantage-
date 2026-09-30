"""CodeTool — routes coding questions to the dedicated code model."""

from __future__ import annotations

from src.agent.base_tool import BaseTool, ToolResult, with_feedback
from src.llm.ollama_manager import OllamaManager
from src.pipeline.pipeline_result import AnswerSource

_SYSTEM = "You are a precise coding assistant. Return working, well-commented code."


class CodeTool(BaseTool):
    name = "code"

    def __init__(self, ollama_manager: OllamaManager, model: str) -> None:
        self.ollama = ollama_manager
        self.model = model

    def _prompt(self, query: str, feedback: str) -> str:
        return f"{_SYSTEM}\n\n{with_feedback(query, feedback)}"

    def execute(self, query: str, feedback: str = "") -> ToolResult:
        answer = self.ollama.generate(model=self.model, prompt=self._prompt(query, feedback))
        return ToolResult(answer=answer, answer_source=AnswerSource.CODE)

    def execute_stream(self, query: str, feedback: str = ""):
        full = ""
        for piece in self.ollama.generate_stream(model=self.model,
                                                 prompt=self._prompt(query, feedback)):
            full += piece
            yield {"type": "chunk", "text": piece}
        yield {"type": "result", "data": ToolResult(answer=full, answer_source=AnswerSource.CODE)}
