"""
ui/engine.py — front-end adapter around backend.py.

It wraps what the backend exposes so the UI can:

  * pick a mode — `Auto` (the agent decides Chat / Code / RAG by itself) or
    `RAG` (docs-only Agentic RAG: answers only from your documents, and says so
    when they don't cover the question);
  * learn which tool the agent chose (the backend emits a `route` event, used
    for the routing badge);
  * translate PipelineResult -> a small JSON-safe `meta` dict for rendering
    and for persistence in chat_history.db;
  * turn the validator's 0-10 score into the friendly confidence card
    (raw retrieval/embedding scores are never exposed).
"""

from __future__ import annotations

import math
import re
from typing import Callable, Iterator

import backend

MODES = ("Auto", "RAG")


def normalize_mode(saved: str | None) -> str:
    """Older installs persisted Chat/Code as the mode; the agent now decides
    those itself, so anything unrecognised falls back to Auto."""
    return saved if saved in MODES else "Auto"


# answer_source (from pipeline_result.AnswerSource) -> tool that produced it
_SOURCE_TO_TOOL = {"rag": "rag", "rag_fallback": "rag", "rag_not_found": "rag",
                   "llm": "chat", "code": "code", "blocked": "blocked"}


def stream_answer(query: str, mode: str = "Auto", fast: bool = False,
                  on_route: Callable[[str], None] | None = None) -> Iterator[dict]:
    """Yield the backend's stream events ({'type': 'status'|'chunk'|'reset'|
    'result', ...}). 'route' events are consumed here and reported via `on_route`.

    mode : Auto | RAG
    fast : opt-in Fast mode — skips the validation/regeneration pass for lower
           latency. Routing still happens, so Auto stays Auto.
    """
    on_route = on_route or (lambda _t: None)
    for event in backend.ask_stream(query, direct_rag=fast, rag_only=(mode == "RAG")):
        if event["type"] == "route":
            on_route(event["tool"])
            continue
        yield event


# ---------------------------------------------------------------- confidence

def _match_pct(score: float) -> int:
    """Reranker output -> 0-100 'match'. The cross-encoder normally returns
    0-1 (sigmoid); if a raw logit sneaks through, squash it."""
    if score < 0 or score > 1:
        score = 1 / (1 + math.exp(-max(-30.0, min(30.0, score))))
    return int(round(score * 100))


def confidence_level(pct: int) -> tuple[str, str]:
    if pct >= 90:
        return "High Confidence", "high"
    if pct >= 70:
        return "Medium Confidence", "medium"
    return "Low Confidence", "low"


def build_confidence(result, chunks: list[dict], fast: bool) -> dict:
    pct = max(0, min(100, int(round(result.confidence_score * 10))))
    label, level = confidence_level(pct)
    matches = sorted((c["match"] for c in chunks), reverse=True)
    docs = {c["source"] for c in chunks}
    n = len(chunks)

    if n == 0:
        coverage = "No supporting sections were retrieved."
    else:
        strong = sum(1 for m in matches if m >= 50)
        coverage = (f"{strong} of {n} retrieved sections closely match your question."
                    if strong else "Retrieved sections only loosely match your question.")

    top = matches[:3]
    avg = sum(top) / len(top) if top else 0
    similarity = ("Very close match to the question" if avg >= 75 else
                  "Good match to the question" if avg >= 50 else "Loose match to the question")

    if len(docs) >= 2:
        agreement = (f"Cross-verified across {len(docs)} documents" if pct >= 70
                     else f"Drawn from {len(docs)} documents that may not fully agree")
    else:
        agreement = "Based on a single document"

    score = result.confidence_score
    grounding = ("Grounded — statements trace back to the retrieved text" if score >= 7.5 else
                 "Mostly grounded — a few statements are only loosely supported" if score >= 5 else
                 "Weakly grounded — treat this answer with care")
    if result.iterations > 1:
        grounding += f" (refined after {result.iterations - 1} self-check pass"
        grounding += "es)" if result.iterations > 2 else ")"

    return {
        "percent": pct, "label": label, "level": level, "verified": not fast,
        "sections": n, "documents": len(docs),
        "reasons": {"Retrieved": f"{n} supporting section{'s' if n != 1 else ''}",
                    "Cross-document agreement": agreement,
                    "Grounding validation": grounding,
                    "Coverage": coverage, "Similarity": similarity},
        "note": (result.validation_reasoning or "").strip()[:400],
    }


def _clean_chunks(raw: list[dict]) -> list[dict]:
    out = []
    for c in raw or []:
        text = re.sub(r"\s+", " ", c.get("text", "") or "").strip()
        out.append({"source": c.get("source", "?"), "page": c.get("page", 0),
                    "match": _match_pct(float(c.get("rerank_score", 0.0))),
                    "text": text[:1200]})
    return out


def build_meta(result, mode: str, fast: bool, routed: str | None) -> dict:
    """PipelineResult -> JSON-safe dict stored on the assistant message."""
    tool = _SOURCE_TO_TOOL.get(result.answer_source, routed or "chat")
    chunks = _clean_chunks(result.retrieved_chunks) if tool == "rag" else []
    meta = {
        "tool": tool, "mode": mode, "fast": fast,
        "answer_source": result.answer_source,
        "latency_ms": result.latency_ms, "iterations": result.iterations,
        "sources": list(result.sources or []),
        "chunks": chunks,
        "guardrail_findings": list(result.guardrail_findings or []),
        "audit_hash": (result.audit_record_hash or "")[:12],
        "agent_steps": list(getattr(result, "agent_steps", []) or []),
        "confidence_score": result.confidence_score,
        "confidence": None,
    }
    if tool == "rag" and result.answer_source == "rag" and chunks:
        meta["confidence"] = build_confidence(result, chunks, fast)
    return meta


# ------------------------------------------------------------ prompt helpers

QUICK_ACTIONS: dict[str, str] = {
    "Summarize PDF": "Summarize the key points of my uploaded documents.",
    "Explain Simply": "Explain this in simple terms, as if to a beginner: ",
    "Generate Notes": "Generate concise study notes from my uploaded documents.",
    "Compare PDFs": "Compare the uploaded documents and highlight similarities and differences.",
    "Extract Tables": "Extract any tables from my uploaded documents and present them in Markdown.",
    "Translate": "Translate the following into Hindi: ",
    "Find Citations": "Find the passages in my documents that support this claim: ",
    "Generate Flashcards": "Create flashcards (question and answer) from my uploaded documents.",
    "Create MCQs": "Create 5 multiple-choice questions with answers from my uploaded documents.",
}

SUGGESTIONS: list[tuple[str, str, str]] = [
    ("Summarize a PDF", "Get the key points of a document you've ingested",
     "Summarize the key points of my uploaded documents."),
    ("Ask a cybersecurity question", "General security concepts, explained clearly",
     "Explain the difference between symmetric and asymmetric encryption."),
    ("Compare two documents", "Similarities and differences across your PDFs",
     "Compare the uploaded documents and highlight similarities and differences."),
]

SLASH_COMMANDS: dict[str, str] = {
    "/summarize": "Summarize the following from my uploaded documents:",
    "/explain": "Explain in simple terms:",
    "/compare": "Compare the uploaded documents regarding:",
    "/find": "Find the passages in my documents about:",
    "/notes": "Generate concise study notes on:",
}


def expand_slash(text: str) -> str:
    """'/summarize chapter 2' -> 'Summarize the following from my uploaded
    documents: chapter 2'. Unknown commands are left untouched."""
    stripped = text.strip()
    if not stripped.startswith("/"):
        return text
    head, _, rest = stripped.partition(" ")
    template = SLASH_COMMANDS.get(head.lower())
    if not template:
        return text
    return f"{template} {rest}".strip()
