"""
EmbeddingManager — wraps BAAI/bge-m3.

Why bge-m3 (kept from the previous pipeline, still the right pick):
  - Multilingual (100+ languages) — matters for legal/compliance PDFs that
    mix English with regional-language annexures.
  - 8192-token context — chunk-level docs embed without truncation.
  - Natively multi-functional: dense + sparse (lexical) + ColBERT-style
    multi-vector output from ONE forward pass. We only use the dense vector
    here (BM25 already covers sparse/lexical), but it's there if you want to
    swap BM25 for bge-m3's own sparse output later.
  - 1024-dim, ~2.3GB in fp16 — comfortably fits a small (6-8GB) GPU
    alongside a 7B quantized chat model, or runs fine on CPU for ingestion.

Loading is local-first (see src.core.model_paths) so inference never depends
on a Hugging Face download, and process-wide cached: IngestionFactory,
PipelineFactory and AgenticPipeline.create() each ask for an EmbeddingManager
independently, and without caching that silently loaded the ~2.3GB model
three times into memory on every cold start. get() returns the same instance
for a given (model_name, device) pair.
"""

from __future__ import annotations

import threading

import numpy as np
from sentence_transformers import SentenceTransformer

from src.core.logging import LoggerManager
from src.core.model_paths import resolve

logger = LoggerManager.get_logger()

_cache: dict[tuple[str, str | None], "EmbeddingManager"] = {}
_cache_lock = threading.Lock()


def _auto_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"  # Apple Silicon
    except Exception:
        pass
    return "cpu"


class EmbeddingManager:
    def __init__(self, model_name: str = "BAAI/bge-m3", device: str | None = None) -> None:
        resolved_device = device or _auto_device()
        local_path = resolve(model_name)
        logger.info(f"Loading embedding model '{model_name}' "
                    f"(source={local_path}, device={resolved_device})...")
        try:
            self.model = SentenceTransformer(local_path, device=resolved_device)
        except Exception as exc:
            logger.warning(f"'{resolved_device}' load failed ({exc}); falling back to CPU.")
            self.model = SentenceTransformer(local_path, device="cpu")

    @classmethod
    def get(cls, model_name: str = "BAAI/bge-m3", device: str | None = None) -> "EmbeddingManager":
        """Process-wide cached accessor — prefer this over the constructor
        so the same model weights aren't loaded into memory more than once."""
        key = (model_name, device)
        with _cache_lock:
            if key not in _cache:
                _cache[key] = cls(model_name=model_name, device=device)
            return _cache[key]

    def embed_text(self, text: str) -> np.ndarray:
        return self.model.encode([text], normalize_embeddings=True)[0]

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return self.model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
        )

    @staticmethod
    def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Vectors are already L2-normalized by embed_text/embed_batch, so this
        is just the dot product — used by the Validation Layer for groundedness scoring."""
        return float(np.dot(vec_a, vec_b))
