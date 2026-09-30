"""
AgentRouter — LLM classifier that decides which tool handles a query:
"chat" (general knowledge / conversation), "code" (programming help),
or "rag" (needs the ingested document corpus).

Runs on the smallest model in the stack (qwen2.5:1.5b, JSON mode) — this
is a 3-way classification, not reasoning, and it's on the critical path
of every non-direct-RAG query, so keeping it fast matters more than
using the big chat model here.
"""

from __future__ import annotations

import json

from src.core.config import ApplicationConfig
from src.core.logging import LoggerManager
from src.llm.ollama_manager import OllamaManager

logger = LoggerManager.get_logger()

_PROMPT = """Classify the user query into exactly one tool. Return ONLY JSON:
{{"tool": "chat"|"code"|"rag"}}

- "rag": question about the uploaded documents, clauses, policies, facts that
  would need a source citation, or anything document/compliance-specific.
- "code": asks for code, a script, a regex, SQL, debugging help.
- "chat": general knowledge, greetings, small talk, anything not needing the
  document corpus.

QUERY: {query}
"""


class AgentRouter:
    VALID_TOOLS = {"chat", "code", "rag"}

    def __init__(self, ollama_manager: OllamaManager, model: str | None = None) -> None:
        self.ollama = ollama_manager
        self.model = model or ApplicationConfig().models["router"]["model"]

    def route(self, query: str) -> str:
        try:
            raw = self.ollama.generate(
                model=self.model, prompt=_PROMPT.format(query=query),
                temperature=0.0, format="json",
            )
            tool = json.loads(raw).get("tool", "rag")
            return tool if tool in self.VALID_TOOLS else "rag"
        except Exception as exc:
            logger.warning(f"AgentRouter failed ({exc}); defaulting to 'rag'.")
            return "rag"
