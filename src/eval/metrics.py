"""
Evaluation metrics for the RAG pipeline.

Why these, not a plain classification confusion matrix:
RAG output is open-ended text, not a class label, so a textbook confusion
matrix doesn't directly apply to "the answer". The right adaptation (used
by RAGAS/TruLens-style RAG evaluation, which this module follows) is:

  1. Score each answer on continuous 0-1 metrics:
       - Faithfulness       : fraction of answer claims supported by the
                               retrieved context (this IS the hallucination
                               metric — 1 - faithfulness = hallucination rate)
       - Answer Relevancy    : does the answer actually address the question
       - Context Precision   : fraction of retrieved chunks that are relevant
       - Context Recall      : fraction of needed information that was retrieved
       - Groundedness Cosine : embedding similarity, answer vs. context
                               (cheap, deterministic cross-check on faithfulness)

  2. THEN, for a binary "is this answer acceptable" decision (used to build
     an actual confusion matrix against your own reference/expected labels
     on a labeled eval set), we threshold: predicted_grounded = faithfulness
     >= threshold. Compare that to your `expected_grounded` (1 = should be
     answerable & correct from the docs, 0 = should be refused/out-of-scope)
     to get real TP/FP/TN/FN, precision, recall, F1 — see confusion_matrix().

This is the standard evaluation practice for RAG systems (RAGAS is the
reference implementation of metrics 1-4; groundedness-cosine and the
threshold-to-confusion-matrix step are the same pattern Vantage's own
ValidationLayer uses at inference time, reused here for offline eval).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from src.embeddings import EmbeddingManager
from src.llm.ollama_manager import OllamaManager

_FAITHFULNESS_PROMPT = """Break the ANSWER into individual factual claims. For each
claim, decide if it is directly supported by the CONTEXT. Return ONLY JSON:
{{"total_claims": <int>, "supported_claims": <int>, "unsupported_examples": ["..."]}}

CONTEXT: {context}
ANSWER: {answer}
"""

_RELEVANCY_PROMPT = """Rate 0-10 how directly the ANSWER addresses the QUESTION
(ignore correctness, only judge relevance/on-topic-ness). Return ONLY JSON:
{{"relevancy_score": <0-10>}}

QUESTION: {question}
ANSWER: {answer}
"""

_CONTEXT_PRECISION_PROMPT = """For each numbered source block, say whether it is
relevant to answering the QUESTION. Return ONLY JSON:
{{"relevant_indices": [1, 3]}}

QUESTION: {question}
SOURCES:
{context}
"""


@dataclass
class EvalRecord:
    query_id: str
    question: str
    answer: str
    context: str
    num_sources_retrieved: int
    expected_grounded: int | None = None     # 1/0 label from a curated eval set, optional
    faithfulness: float = 0.0
    hallucination_rate: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    groundedness_cosine: float = 0.0
    confidence_score: float = 0.0
    predicted_grounded: int = 0
    latency_ms: int = 0


class RagEvaluator:
    def __init__(self, ollama: OllamaManager, judge_model: str, embedder: EmbeddingManager,
                 faithfulness_threshold: float = 0.7) -> None:
        self.ollama = ollama
        self.judge_model = judge_model
        self.embedder = embedder
        self.threshold = faithfulness_threshold

    def evaluate(self, query_id: str, question: str, answer: str, context: str,
                 num_sources_retrieved: int, confidence_score: float, latency_ms: int,
                 expected_grounded: int | None = None) -> EvalRecord:
        faithfulness = self._faithfulness(context, answer)
        relevancy = self._relevancy(question, answer) / 10.0
        precision = self._context_precision(question, context)
        cosine = self._groundedness_cosine(answer, context)

        record = EvalRecord(
            query_id=query_id, question=question, answer=answer, context=context,
            num_sources_retrieved=num_sources_retrieved, expected_grounded=expected_grounded,
            faithfulness=round(faithfulness, 3), hallucination_rate=round(1 - faithfulness, 3),
            answer_relevancy=round(relevancy, 3), context_precision=round(precision, 3),
            groundedness_cosine=round(cosine, 3), confidence_score=confidence_score,
            predicted_grounded=1 if faithfulness >= self.threshold else 0,
            latency_ms=latency_ms,
        )
        return record

    def _faithfulness(self, context: str, answer: str) -> float:
        try:
            raw = self.ollama.generate(model=self.judge_model,
                                        prompt=_FAITHFULNESS_PROMPT.format(context=context[:4000], answer=answer),
                                        temperature=0.0, format="json")
            data = json.loads(raw)
            total, supported = data.get("total_claims", 1), data.get("supported_claims", 0)
            return supported / total if total else 0.0
        except Exception:
            return 0.5

    def _relevancy(self, question: str, answer: str) -> float:
        try:
            raw = self.ollama.generate(model=self.judge_model,
                                        prompt=_RELEVANCY_PROMPT.format(question=question, answer=answer),
                                        temperature=0.0, format="json")
            return float(json.loads(raw).get("relevancy_score", 5.0))
        except Exception:
            return 5.0

    def _context_precision(self, question: str, context: str) -> float:
        blocks = context.count("[S")
        if blocks == 0:
            return 0.0
        try:
            raw = self.ollama.generate(model=self.judge_model,
                                        prompt=_CONTEXT_PRECISION_PROMPT.format(question=question, context=context[:4000]),
                                        temperature=0.0, format="json")
            relevant = len(json.loads(raw).get("relevant_indices", []))
            return min(1.0, relevant / blocks)
        except Exception:
            return 0.5

    def _groundedness_cosine(self, answer: str, context: str) -> float:
        if not answer.strip() or not context.strip():
            return 0.0
        a = self.embedder.embed_text(answer)
        c = self.embedder.embed_text(context[:4000])
        return max(0.0, min(1.0, (self.embedder.cosine_similarity(a, c) + 1.0) / 2.0))


def confusion_matrix(records: list[EvalRecord]) -> dict:
    """Binary confusion matrix: predicted_grounded vs expected_grounded, for
    records that carry a curated expected_grounded label (skips unlabeled rows)."""
    labeled = [r for r in records if r.expected_grounded is not None]
    tp = sum(1 for r in labeled if r.predicted_grounded == 1 and r.expected_grounded == 1)
    fp = sum(1 for r in labeled if r.predicted_grounded == 1 and r.expected_grounded == 0)
    tn = sum(1 for r in labeled if r.predicted_grounded == 0 and r.expected_grounded == 0)
    fn = sum(1 for r in labeled if r.predicted_grounded == 0 and r.expected_grounded == 1)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(labeled) if labeled else 0.0

    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": round(precision, 3),
            "recall": round(recall, 3), "f1": round(f1, 3), "accuracy": round(accuracy, 3),
            "n_labeled": len(labeled)}
