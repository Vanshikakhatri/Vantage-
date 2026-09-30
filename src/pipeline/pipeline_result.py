"""Unified result dataclass returned by the agentic pipeline (Fast mode and full
validation paths alike), plus the guardrail-block/audit fields the UI needs to show
"why" a query was refused or an answer was redacted."""

from __future__ import annotations

from dataclasses import dataclass, field


class AnswerSource:
    RAG = "rag"
    LLM = "llm"
    CODE = "code"
    RAG_FALLBACK = "rag_fallback"
    RAG_NOT_FOUND = "rag_not_found"     # docs-only (strict) RAG found nothing relevant
    BLOCKED = "blocked"


def confidence_label(score: float) -> str:
    if score >= 7.5:
        return "High"
    if score >= 5.0:
        return "Medium"
    return "Low"


def confidence_badge(score: float) -> str:
    label = confidence_label(score)
    icon = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}[label]
    return f"{icon} {label} ({score:.1f}/10)"


def answer_source_label(source: str) -> str:
    return {
        AnswerSource.RAG: "📄 Grounded in your documents",
        AnswerSource.LLM: "🤖 LLM answer (own knowledge)",
        AnswerSource.CODE: "💻 Code answer (own knowledge)",
        AnswerSource.RAG_FALLBACK: "⚠️ RAG fallback — LLM answered from general knowledge",
        AnswerSource.RAG_NOT_FOUND: "🔎 Not found in your documents",
        AnswerSource.BLOCKED: "⛔ Blocked by guardrail",
    }.get(source, "❓ Unknown source")


@dataclass
class PipelineResult:
    answer: str
    sources: list[str] = field(default_factory=list)
    confidence_score: float = 0.0
    answer_source: str = AnswerSource.RAG
    validation_reasoning: str = ""
    iterations: int = 1
    latency_ms: int = 0
    guardrail_findings: list[str] = field(default_factory=list)
    audit_record_hash: str = ""
    retrieved_chunks: list[dict] = field(default_factory=list)   # per-chunk retrieval scores
    agent_steps: list[str] = field(default_factory=list)          # what the agent decided/did

    @property
    def confidence_badge(self) -> str:
        return confidence_badge(self.confidence_score)

    @property
    def answer_source_label(self) -> str:
        return answer_source_label(self.answer_source)
