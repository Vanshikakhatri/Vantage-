"""
ValidationLayer — LLM-as-judge + deterministic checks -> one confidence score.

Every answer (Chat, Code, RAG — unless the user opted into Fast mode) is scored
0-10. That single number drives the Reflection loop: below `threshold` (5.5) the
answer is regenerated, up to 2 more times, and the best attempt is kept.

HOW THE SCORE IS BUILT
----------------------
Four signals, each on a 0-10 scale, combined as a weighted average:

  correctness   (LLM judge)     does the answer correctly reflect the context /
                                 the facts?                              weight 0.40
  completeness  (LLM judge)     does it cover everything the question asked?  0.20
  citations     (deterministic)  fraction of the [S#] tags in the answer that point
                                 at a real retrieved source. No LLM involved, so a
                                 judge can't "imagine" a citation that isn't there. 0.15
  groundedness  (deterministic)  cosine similarity between the answer's embedding and
                                 the retrieved context's embedding, scaled to 0-10.
                                 Independent of the judge, so a badly-calibrated
                                 judge can't singlehandedly wave through drift.     0.25

Two rules keep that average honest:

  1. Only signals that APPLY are counted. Chat/Code answers (and RAG fallbacks) have
     no retrieved context, so citations and groundedness are meaningless for them;
     including them as 0 / neutral would unfairly drag every such answer toward the
     threshold. For those, the score is the correctness+completeness average only
     (weights renormalised).
  2. Correctness is a veto, not just a weight. If the judge scores correctness below
     `correctness_floor` (default 4.0) the confidence is capped just under the
     threshold, so strong citations/groundedness can never mask a wrong answer.

Weights and floor are configurable in application.yaml (pipeline.validation).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from src.core.logging import LoggerManager
from src.embeddings import EmbeddingManager
from src.llm.ollama_manager import OllamaManager

logger = LoggerManager.get_logger()

_NO_CONTEXT_MARKER = "(no relevant context retrieved)"
_NO_CONTEXT_NOTE = ("(none — this is a general-knowledge answer; judge factual "
                    "correctness and completeness only)")

_JUDGE_PROMPT = """You are grading an AI answer for a compliance document QA system.
Score strictly. Return ONLY JSON:
{{"correctness": <0-10>, "completeness": <0-10>, "reasoning": "<one sentence>"}}

QUESTION: {query}
CONTEXT: {context}
ANSWER: {answer}
"""

_CITE_RE = re.compile(r"\[S(\d+)\]")
_SOURCE_TAG_RE = re.compile(r"^\[S(\d+)\]", re.MULTILINE)


@dataclass
class ValidationResult:
    confidence_score: float
    correctness: float
    completeness: float
    has_citations: bool
    groundedness_cosine: float = 0.0
    reasoning: str = ""
    has_context: bool = True          # False for chat/code/fallback answers (no retrieved sources)


class ValidationLayer:
    def __init__(self, ollama_manager: OllamaManager, model: str, threshold: float,
                 weights: dict, embedding_manager: EmbeddingManager | None = None,
                 correctness_floor: float = 4.0) -> None:
        self.ollama = ollama_manager
        self.model = model
        self.threshold = threshold
        self.weights = weights
        self.embedder = embedding_manager
        self.correctness_floor = correctness_floor

    # ------------------------------------------------------------------ scoring

    def validate(self, query: str, answer: str, context: str) -> ValidationResult:
        has_context = self._has_context(context)
        if not answer or not answer.strip():
            return ValidationResult(0.0, 0.0, 0.0, False, 0.0, "The answer was empty.",
                                    has_context=has_context)

        judge = self._judge(query, answer, context if has_context else "")

        w = self.weights
        terms: list[tuple[float, float]] = [          # (weight, score 0-10)
            (w.get("correctness", 0.4), judge["correctness"]),
            (w.get("completeness", 0.2), judge["completeness"]),
        ]
        citation_score, groundedness_10 = 0.0, 0.0
        if has_context:
            citation_score = self._citation_score(answer, context)
            groundedness_10 = self._groundedness(answer, context)
            terms.append((w.get("citations", 0.15), citation_score))
            terms.append((w.get("groundedness_cosine", 0.25), groundedness_10))

        total_weight = sum(wt for wt, _ in terms) or 1.0
        confidence = sum(wt * score for wt, score in terms) / total_weight

        reasoning = judge.get("reasoning", "")
        if judge["correctness"] < self.correctness_floor:
            capped = max(0.0, self.threshold - 0.1)
            if confidence > capped:
                confidence = capped
                reasoning = (reasoning + " " if reasoning else "") + \
                    "[Confidence capped: judge rated correctness too low to trust this answer.]"

        return ValidationResult(
            confidence_score=round(min(10.0, max(0.0, confidence)), 2),
            correctness=judge["correctness"], completeness=judge["completeness"],
            has_citations=citation_score > 0, groundedness_cosine=round(groundedness_10, 2),
            reasoning=reasoning, has_context=has_context,
        )

    def passed(self, result: ValidationResult) -> bool:
        """Whether this validator's threshold is met — use this instead of a
        hardcoded cutoff on ValidationResult, since the threshold is per-instance
        (configurable in application.yaml)."""
        return result.confidence_score >= self.threshold

    # ------------------------------------------------------------------ signals

    @staticmethod
    def _has_context(context: str) -> bool:
        stripped = (context or "").strip()
        return bool(stripped) and stripped != _NO_CONTEXT_MARKER

    def _judge(self, query: str, answer: str, context: str) -> dict:
        try:
            raw = self.ollama.generate(
                model=self.model,
                prompt=_JUDGE_PROMPT.format(query=query, context=(context[:4000] or _NO_CONTEXT_NOTE),
                                            answer=answer),
                temperature=0.0, format="json",
            )
            data = json.loads(raw)
            return {
                "correctness": self._clamp10(data.get("correctness", 5.0)),
                "completeness": self._clamp10(data.get("completeness", 5.0)),
                "reasoning": str(data.get("reasoning", "")),
            }
        except Exception as exc:
            logger.warning(f"ValidationLayer judge call failed ({exc}); using neutral fallback.")
            return {"correctness": 5.0, "completeness": 5.0,
                    "reasoning": "Judge LLM unavailable — neutral score."}

    @staticmethod
    def _clamp10(value) -> float:
        return max(0.0, min(10.0, float(value)))

    @staticmethod
    def _citation_score(answer: str, context: str) -> float:
        """0-10: share of the answer's [S#] tags that reference a source that was
        actually in the context. No tags at all -> 0."""
        cited = {int(n) for n in _CITE_RE.findall(answer)}
        if not cited:
            return 0.0
        available = {int(n) for n in _SOURCE_TAG_RE.findall(context)}
        return 10.0 * len(cited & available) / len(cited)

    def _groundedness(self, answer: str, context: str) -> float:
        if not self.embedder or not context.strip() or not answer.strip():
            return 5.0  # neutral default when there's nothing to compare
        try:
            answer_vec = self.embedder.embed_text(answer)
            context_vec = self.embedder.embed_text(context[:4000])
            cosine = self.embedder.cosine_similarity(answer_vec, context_vec)  # -1..1
            return max(0.0, min(10.0, (cosine + 1.0) / 2.0 * 10.0))
        except Exception as exc:
            logger.warning(f"Groundedness cosine check failed: {exc}")
            return 5.0
