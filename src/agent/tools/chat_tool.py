"""ChatTool — answers from the model's own knowledge, no retrieval. Router
sends here for greetings/general knowledge that isn't document-specific."""

from __future__ import annotations

from src.agent.base_tool import BaseTool, ToolResult, with_feedback
from src.llm.ollama_manager import OllamaManager
from src.pipeline.pipeline_result import AnswerSource


class ChatTool(BaseTool):
    name = "chat"

    def __init__(self, ollama_manager: OllamaManager, model: str) -> None:
        self.ollama = ollama_manager
        self.model = model

    def execute(self, query: str, feedback: str = "") -> ToolResult:
        answer = self.ollama.generate(model=self.model, prompt=with_feedback(query, feedback))
        return ToolResult(answer=answer, answer_source=AnswerSource.LLM)

    def execute_stream(self, query: str, feedback: str = ""):
        full = ""
        for piece in self.ollama.generate_stream(model=self.model,
                                                 prompt=with_feedback(query, feedback)):
            full += piece
            yield {"type": "chunk", "text": piece}
        yield {"type": "result", "data": ToolResult(answer=full, answer_source=AnswerSource.LLM)}
