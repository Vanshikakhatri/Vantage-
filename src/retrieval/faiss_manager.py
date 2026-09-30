"""FAISSManager — flat inner-product index (vectors are pre-normalized by
EmbeddingManager, so inner product == cosine similarity). Flat index chosen
over IVF/HNSW deliberately: at the document scale this project targets
(single-tenant legal/compliance corpora, thousands not millions of chunks),
flat search is exact, has zero training step, and is fast enough — no
recall loss for an accuracy-first, audit-sensitive use case."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import faiss
import numpy as np

from src.models import Chunk, EmbeddedChunk

_lock = threading.Lock()


class FAISSManager:
    def __init__(self, dimension: int, index_path: str, metadata_path: str) -> None:
        self.dimension = dimension
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata: list[dict] = []   # parallel array: row i <-> chunk i

    def load(self) -> None:
        with _lock:
            if self.index_path.exists():
                self.index = faiss.read_index(str(self.index_path))
            if self.metadata_path.exists():
                self.metadata = json.loads(self.metadata_path.read_text())

    def save(self) -> None:
        with _lock:
            faiss.write_index(self.index, str(self.index_path))
            self.metadata_path.write_text(json.dumps(self.metadata))

    def add(self, embeddings: np.ndarray, chunks: list[Chunk]) -> None:
        with _lock:
            self.index.add(embeddings.astype("float32"))
            self.metadata.extend([
                {"chunk_id": c.chunk_id, "document_id": c.document_id, "source": c.source,
                 "page": c.page, "text": c.text, "modality": c.modality}
                for c in chunks
            ])
        self.save()

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> list[EmbeddedChunk]:
        if self.index.ntotal == 0:
            return []
        scores, indices = self.index.search(
            query_embedding.astype("float32").reshape(1, -1), min(top_k, self.index.ntotal)
        )
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or idx >= len(self.metadata):
                continue
            meta = self.metadata[idx]
            chunk = Chunk(chunk_id=meta["chunk_id"], document_id=meta["document_id"],
                          source=meta["source"], page=meta["page"], text=meta["text"],
                          modality=meta.get("modality", "text"))
            results.append(EmbeddedChunk(chunk=chunk, score=float(score), dense_score=float(score)))
        return results

    def delete_by_document_id(self, document_id: str) -> None:
        """Flat index has no in-place delete — rebuild from surviving metadata."""
        with _lock:
            keep = [(i, m) for i, m in enumerate(self.metadata) if m["document_id"] != document_id]
            if len(keep) == len(self.metadata):
                return
            new_index = faiss.IndexFlatIP(self.dimension)
            if keep:
                vectors = np.vstack([
                    self.index.reconstruct(i) for i, _ in keep
                ]).astype("float32")
                new_index.add(vectors)
            self.index = new_index
            self.metadata = [m for _, m in keep]
        self.save()
