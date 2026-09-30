"""
ApplicationConfig — single YAML-backed config singleton.

Loads configs/application.yaml, models.yaml, storage.yaml, guardrails.yaml
and exposes them as plain dicts. Editing YAML is enough to change models,
paths, thresholds, or guardrail rules — no source changes needed.

A handful of values that legitimately differ per machine/OS/container are
also overridable with environment variables, so the same YAML works
unmodified on macOS, Linux and inside Docker:

  VANTAGE_DATA_DIR      - root dir for sqlite/faiss/bm25/graph/uploads/audit
                           (defaults to the "data/" paths in storage.yaml)
  VANTAGE_MODELS_DIR     - local-first model weights dir (see model_paths.py)
  VANTAGE_OLLAMA_HOST    - overrides models.yaml > ollama.host
  VANTAGE_CHAT_MODEL     - overrides models.yaml > chat.model
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import yaml

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
_lock = threading.Lock()


class ApplicationConfig:
    """Singleton. Call ApplicationConfig() anywhere — same instance every time."""

    _instance: "ApplicationConfig | None" = None

    def __new__(cls):
        if cls._instance is None:
            with _lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._load()
        return cls._instance

    def _load(self) -> None:
        self.app = self._read("application.yaml")["app"]
        self.compliance = self._read("application.yaml")["compliance"]
        self.pipeline = self._read("application.yaml")["pipeline"]
        self.models = self._read("models.yaml")
        self.storage = self._read("storage.yaml")["storage"]
        self.guardrails = self._read("guardrails.yaml")

        self._apply_env_overrides()

        # Make sure runtime data directories exist.
        for key in ("faiss", "bm25", "graph", "uploads", "audit_log"):
            entry = self.storage[key]
            path_str = entry.get("index_path") or entry.get("path") or entry.get("dir")
            p = Path(path_str)
            target_dir = p if entry.get("dir") else p.parent
            target_dir.mkdir(parents=True, exist_ok=True)
        Path(self.storage["sqlite"]["path"]).parent.mkdir(parents=True, exist_ok=True)

    def _apply_env_overrides(self) -> None:
        data_dir = os.environ.get("VANTAGE_DATA_DIR")
        if data_dir:
            base = Path(data_dir)
            self.storage["sqlite"]["path"] = str(base / "vantage.db")
            self.storage["faiss"]["index_path"] = str(base / "faiss" / "index.faiss")
            self.storage["faiss"]["metadata_path"] = str(base / "faiss" / "metadata.json")
            self.storage["bm25"]["index_path"] = str(base / "bm25" / "index.pkl")
            self.storage["graph"]["path"] = str(base / "graph" / "knowledge_graph.gpickle")
            self.storage["uploads"]["dir"] = str(base / "uploads")
            self.storage["audit_log"]["path"] = str(base / "audit" / "audit_log.jsonl")

        ollama_host = os.environ.get("VANTAGE_OLLAMA_HOST")
        if ollama_host:
            self.models["ollama"]["host"] = ollama_host

        chat_model = os.environ.get("VANTAGE_CHAT_MODEL")
        if chat_model:
            self.models["chat"]["model"] = chat_model

    @staticmethod
    def _read(filename: str) -> dict:
        with open(_CONFIG_DIR / filename, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)

    def reload(self) -> None:
        """Hot-reload config from disk (e.g. after editing guardrails.yaml in the UI)."""
        self._load()
