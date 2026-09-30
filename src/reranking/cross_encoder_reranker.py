"""
CrossEncoderReranker — BAAI/bge-reranker-v2-m3.

Kept from the previous pipeline — still the strongest small multilingual
cross-encoder reranker (lightweight, ~560M params, runs fine on a small
GPU or CPU). Reranking matters here specifically because RRF fusion
(hybrid_retriever.py) is rank-based, not relevance-based — it's good at
"don't lose a candidate either list liked" but bad at fine-grained ordering.
The cross-encoder reads query+chunk together (unlike bi-encoder embedding
search) so it catches relevance nuances dense/sparse retrieval miss, and
its output score also feeds the Validation Layer's groundedness signal.

Loads local-first (src.core.model_paths) — no Hugging Face download at
inference time once the weights are on disk — and is process-wide cached
like EmbeddingManager, since nothing here changes between requests.
"""

from __future__ import annotations

import threading

from sentence_transformers import CrossEncoder

from src.core.logging import LoggerManager
from src.core.model_paths import resolve
from src.models import EmbeddedChunk

logger = LoggerManager.get_logger()

_cache: dict[str, "CrossEncoderReranker"] = {}
_cache_lock = threading.Lock()


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3") -> None:
        local_path = resolve(model_name)
        logger.info(f"Loading reranker '{model_name}' (source={local_path})...")
        try:
            self.model = CrossEncoder(local_path)
        except Exception as exc:
            logger.warning(f"Reranker GPU load failed ({exc}); falling back to CPU.")
            self.model = CrossEncoder(local_path, device="cpu")

    @classmethod
    def get(cls, model_name: str = "BAAI/bge-reranker-v2-m3") -> "CrossEncoderReranker":
        with _cache_lock:
            if model_name not in _cache:
                _cache[model_name] = cls(model_name=model_name)
            return _cache[model_name]

    def rerank(self, query: str, chunks: list[EmbeddedChunk], top_k: int = 6) -> list[EmbeddedChunk]:
        if not chunks:
            return []
        pairs = [[query, c.chunk.text] for c in chunks]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)
        return [
            EmbeddedChunk(
                chunk=c.chunk,
                score=float(s),
                dense_score=c.dense_score,
                bm25_score=c.bm25_score,
                rerank_score=float(s),
            )
            for c, s in ranked[:top_k]
        ]
