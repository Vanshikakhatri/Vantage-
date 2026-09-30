"""
OllamaManager — thin wrapper around the local Ollama client.

Single place every model call goes through (chat generation, streaming,
vision/VLM captioning, JSON-mode calls for the router/validator/rewriter),
so swapping the HTTP client or adding retry/timeout logic happens once.
"""

from __future__ import annotations

from typing import Iterator

import ollama

from src.core.logging import LoggerManager

logger = LoggerManager.get_logger()


class OllamaManager:
    def __init__(self, host: str = "http://localhost:11434") -> None:
        self.client = ollama.Client(host=host)

    def list_models(self) -> list[str]:
        try:
            resp = self.client.list()
            return [m["model"] for m in resp.get("models", [])]
        except Exception as exc:
            logger.warning(f"Could not reach Ollama at list_models(): {exc}")
            return []

    def generate(self, model: str, prompt: str, temperature: float = 0.2,
                 format: str | None = None) -> str:
        resp = self.client.generate(
            model=model, prompt=prompt, format=format,
            options={"temperature": temperature},
        )
        return resp["response"]

    def generate_stream(self, model: str, prompt: str, temperature: float = 0.2) -> Iterator[str]:
        stream = self.client.generate(
            model=model, prompt=prompt, stream=True, options={"temperature": temperature},
        )
        for chunk in stream:
            yield chunk.get("response", "")

    def generate_vision(self, model: str, prompt: str, images: list[str]) -> str:
        """images: list of base64-encoded PNG/JPEG strings."""
        resp = self.client.generate(model=model, prompt=prompt, images=images)
        return resp["response"]

    def embed(self, model: str, text: str) -> list[float]:
        resp = self.client.embeddings(model=model, prompt=text)
        return resp["embedding"]
