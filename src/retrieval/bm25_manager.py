"""BM25Manager — sparse/lexical index. Complements dense (FAISS) search:
catches exact-term hits (clause numbers, defined-term capitalisation,
statute citations like "Section 8(5)") that embeddings can under-weight."""

from __future__ import annotations

import pickle
import re
import threading
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.models import Chunk, EmbeddedChunk

_lock = threading.Lock()
_TOKEN = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Manager:
    def __init__(self, index_path: str) -> None:
        self.index_path = Path(index_path)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.bm25: BM25Okapi | None = None
        self.metadata: list[dict] = []

    def load(self) -> None:
        with _lock:
            if self.index_path.exists():
                with open(self.index_path, "rb") as fh:
                    self.bm25, self.metadata = pickle.load(fh)

    def save(self) -> None:
        with _lock:
            with open(self.index_path, "wb") as fh:
                pickle.dump((self.bm25, self.metadata), fh)

    def add(self, chunks: list[Chunk]) -> None:
        with _lock:
            self.metadata.extend([
                {"chunk_id": c.chunk_id, "document_id": c.document_id, "source": c.source,
                 "page": c.page, "text": c.text, "modality": c.modality}
                for c in chunks
            ])
            corpus = [_tokenize(m["text"]) for m in self.metadata]
            self.bm25 = BM25Okapi(corpus) if corpus else None
        self.save()

    def search(self, query: str, top_k: int = 10) -> list[EmbeddedChunk]:
        if self.bm25 is None or not self.metadata:
            return []
        scores = self.bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for idx in ranked:
            if scores[idx] <= 0:
                continue
            meta = self.metadata[idx]
            chunk = Chunk(chunk_id=meta["chunk_id"], document_id=meta["document_id"],
                          source=meta["source"], page=meta["page"], text=meta["text"],
                          modality=meta.get("modality", "text"))
            results.append(EmbeddedChunk(chunk=chunk, score=float(scores[idx]),
                                          bm25_score=float(scores[idx])))
        return results

    def delete_by_document_id(self, document_id: str) -> None:
        with _lock:
            self.metadata = [m for m in self.metadata if m["document_id"] != document_id]
            corpus = [_tokenize(m["text"]) for m in self.metadata]
            self.bm25 = BM25Okapi(corpus) if corpus else None
        self.save()
