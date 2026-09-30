"""ContextBuilder — turns reranked chunks into a single, cite-able LLM
context string. Each block is tagged [S1], [S2]... so the chat model can
cite sources inline, and so ValidationLayer / groundedness-cosine can map
an answer's citations back to real chunks."""

from __future__ import annotations

from src.models import EmbeddedChunk


class ContextBuilder:
    def build(self, chunks: list[EmbeddedChunk]) -> str:
        if not chunks:
            return "(no relevant context retrieved)"
        blocks = []
        for i, ec in enumerate(chunks, start=1):
            c = ec.chunk
            tag = f"[S{i}] {c.source} — page {c.page}" + (
                " (image caption)" if c.modality == "image_caption" else ""
            )
            blocks.append(f"{tag}\n{c.text}")
        return "\n\n".join(blocks)
