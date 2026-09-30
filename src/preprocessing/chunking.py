"""
ChunkingService — recursive character splitter with token-aware sizing.

WHERE IT SITS: after text is extracted per-page (and after VLM captions are
generated for image-only pages), before embedding. Same chunker is reused
for both text chunks and image-caption chunks so both flow through one
consistent pipeline downstream.
"""

from __future__ import annotations

import re
import uuid

from src.models import Chunk

_SPLIT_SEPARATORS = ["\n\n", "\n", ". ", " "]


class ChunkingService:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> list[str]:
        text = re.sub(r"\s+\n", "\n", text).strip()
        if len(text) <= self.chunk_size:
            return [text] if text else []
        return self._recursive_split(text, self._SEPARATORS())

    def _SEPARATORS(self):
        return _SPLIT_SEPARATORS

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        if not separators:
            return self._fixed_split(text)

        sep, rest = separators[0], separators[1:]
        parts = text.split(sep)
        chunks: list[str] = []
        current = ""

        for part in parts:
            candidate = (current + sep + part) if current else part
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if len(part) > self.chunk_size:
                    chunks.extend(self._recursive_split(part, rest))
                    current = ""
                else:
                    current = part

        if current:
            chunks.append(current)

        return self._apply_overlap(chunks)

    def _fixed_split(self, text: str) -> list[str]:
        step = max(1, self.chunk_size - self.chunk_overlap)
        return [text[i:i + self.chunk_size] for i in range(0, len(text), step)]

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        if self.chunk_overlap <= 0 or len(chunks) <= 1:
            return chunks
        overlapped = [chunks[0]]
        for prev, cur in zip(chunks, chunks[1:]):
            tail = prev[-self.chunk_overlap:]
            overlapped.append((tail + " " + cur).strip())
        return overlapped

    def make_chunks(self, document_id: str, source: str, page: int, text: str,
                     modality: str = "text") -> list[Chunk]:
        return [
            Chunk(
                chunk_id=str(uuid.uuid4()), document_id=document_id, source=source,
                page=page, text=piece, modality=modality,
            )
            for piece in self.split(text)
        ]
