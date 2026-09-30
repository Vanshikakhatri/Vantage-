"""
model_paths — resolves a HuggingFace repo id (e.g. "BAAI/bge-m3") to a local
on-disk model directory when one is available, so embeddings/reranker/
semantic-guardrail models never try to hit the network at inference time.

Resolution order:
  1. `VANTAGE_MODELS_DIR` env var (defaults to "models" next to the repo
     root, or "/app/models" inside the Docker image) + the repo id's last
     path segment, e.g. models/bge-m3, models/bge-reranker-v2-m3.
  2. The bare HuggingFace repo id — sentence-transformers/CrossEncoder will
     use the local HF cache if the model was already downloaded once, or
     fetch it if a network is available. This keeps the app usable before
     the user has copied model weights into VANTAGE_MODELS_DIR.

Set VANTAGE_OFFLINE=1 (or drop the weights in VANTAGE_MODELS_DIR) to
guarantee zero network calls: transformers/sentence-transformers respect
the HF_HUB_OFFLINE env var, which this module sets automatically whenever
a local path is actually found.
"""

from __future__ import annotations

import os
from pathlib import Path

_DEFAULT_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


def models_dir() -> Path:
    configured = os.environ.get("VANTAGE_MODELS_DIR")
    return Path(configured) if configured else _DEFAULT_MODELS_DIR


def resolve(repo_id: str) -> str:
    """Return a local directory for `repo_id` if one exists on disk,
    otherwise return `repo_id` unchanged (HF cache / network fallback)."""
    local_name = repo_id.split("/")[-1]
    candidate = models_dir() / local_name
    if candidate.exists() and any(candidate.iterdir()):
        # A local copy is present — make sure nothing downstream tries to
        # reach the network even opportunistically (e.g. to check for
        # updated config files).
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        return str(candidate)
    return repo_id
