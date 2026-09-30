"""
Layer 2 — Semantic Guardrail (embedding similarity).

Regex/keyword rules only catch phrasing you thought to write down. This
layer embeds the incoming query with a small, dedicated sentence embedder
and compares it (cosine similarity) against a bank of known-bad "reference
intents" (jailbreaks, prompt extraction, weapons, malware, PII exfiltration
attempts, etc). Catches paraphrases and translations the rule layer misses.

Runs AFTER the input rule guardrail (so obviously-blocked text never even
pays the embedding-model cost) and BEFORE the query reaches the router/RAG.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from src.core.config import ApplicationConfig
from src.core.logging import LoggerManager
from src.core.model_paths import resolve
from src.security.schemas import GuardrailFinding, GuardrailVerdict

logger = LoggerManager.get_logger()

_model_cache: dict[str, SentenceTransformer] = {}


def _get_model(name: str) -> SentenceTransformer:
    if name not in _model_cache:
        local_path = resolve(name)
        logger.info(f"Loading semantic guardrail model '{name}' (source={local_path})...")
        _model_cache[name] = SentenceTransformer(local_path)
    return _model_cache[name]


class SemanticGuardrail:
    def __init__(self) -> None:
        cfg = ApplicationConfig().models["semantic_guardrail"]
        self.threshold = cfg["similarity_threshold"]
        self.model = _get_model(cfg["model"])
        self.reference_intents: list[str] = ApplicationConfig().guardrails[
            "semantic_guardrail"
        ]["reference_intents"]
        self._ref_embeddings = self.model.encode(
            self.reference_intents, normalize_embeddings=True
        )

    def check(self, text: str) -> GuardrailVerdict:
        query_emb = self.model.encode([text], normalize_embeddings=True)[0]
        sims = self._ref_embeddings @ query_emb  # cosine sim (vectors are normalized)
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])

        if best_score >= self.threshold:
            finding = GuardrailFinding(
                layer="semantic",
                rule_or_intent=self.reference_intents[best_idx],
                span=f"cosine={best_score:.3f}",
            )
            logger.warning(
                f"SemanticGuardrail blocked query (sim={best_score:.3f} "
                f"vs intent='{self.reference_intents[best_idx]}')"
            )
            return GuardrailVerdict(
                allowed=False,
                findings=[finding],
                reason=f"Query semantically matches a restricted intent "
                       f"(similarity {best_score:.2f} \u2265 {self.threshold}).",
            )

        return GuardrailVerdict(allowed=True, normalized_query=text.strip())
