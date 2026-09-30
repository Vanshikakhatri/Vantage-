"""
Reflector — confidence-gated regeneration.

WHERE IT SITS: right after ValidationLayer. If the answer's confidence is below
the threshold (5.5 by default) the SAME tool is asked to answer again — up to
`max_retries` (2) more times, i.e. at most 3 answers per question — and the
best-scoring attempt is kept (a retry is never allowed to make things worse).

Each retry is *informed*, not a blind re-roll: the validator's verdict is turned
into concrete feedback ("correctness low: ...; citations missing") and handed to
the tool via `feedback=`. The feedback travels separately from the question, so
it can't leak into anything derived from the question (e.g. the RAG search
string). The RAG tool also uses it to take a second look at its evidence.
"""

from __future__ import annotations

from src.agent.base_tool import BaseTool, ToolResult
from src.core.config import ApplicationConfig
from src.core.logging import LoggerManager
from src.validation.validation_layer import ValidationLayer, ValidationResult

logger = LoggerManager.get_logger()

_WEAK = 6.0    # an individual signal below this is called out in the feedback


def build_feedback(validation: ValidationResult, threshold: float) -> str:
    """Turns a low-confidence verdict into concrete instructions for the retry."""
    problems = []
    if validation.correctness < _WEAK:
        problems.append("factual correctness — only state what is supported")
    if validation.completeness < _WEAK:
        problems.append("completeness — address every part of the question")
    if validation.has_context and not validation.has_citations:
        problems.append("citations — cite claims with valid [S#] tags")
    if validation.has_context and validation.groundedness_cosine < _WEAK:
        problems.append("grounding — stay close to the source text")
    parts = [f"Your previous answer scored {validation.confidence_score:.1f}/10 "
             f"(required: {threshold:.1f})."]
    if problems:
        parts.append("Weak areas: " + "; ".join(problems) + ".")
    if validation.reasoning:
        parts.append(f"Reviewer note: {validation.reasoning}")
    return " ".join(parts)


class Reflector:
    def __init__(self, validator: ValidationLayer, max_retries: int | None = None) -> None:
        self.validator = validator
        self.max_retries = (max_retries if max_retries is not None
                            else ApplicationConfig().pipeline["reflection"]["max_retries"])

    def reflect(self, tool: BaseTool, query: str, initial_result: ToolResult,
                initial_validation: ValidationResult) -> tuple[ToolResult, ValidationResult, int]:
        best_result, best_validation, iterations = initial_result, initial_validation, 1

        for attempt in range(self.max_retries):
            if self.validator.passed(best_validation):
                break
            feedback = build_feedback(best_validation, self.validator.threshold)
            candidate = tool.execute(query, feedback=feedback)
            candidate_validation = self.validator.validate(query, candidate.answer, candidate.context)
            iterations += 1
            logger.info(f"Reflection attempt {attempt + 1}/{self.max_retries}: score "
                        f"{candidate_validation.confidence_score:.1f} (best so far: "
                        f"{best_validation.confidence_score:.1f})")
            if candidate_validation.confidence_score > best_validation.confidence_score:
                best_result, best_validation = candidate, candidate_validation

        return best_result, best_validation, iterations

    def reflect_stream(self, tool: BaseTool, query: str, initial_result: ToolResult,
                       initial_validation: ValidationResult):
        """Streaming variant. Yields UI events and RETURNS (result, validation,
        iterations) — call it with `yield from`.

        Event protocol: a 'status' event announces each retry, then a 'reset' event
        tells the UI to discard the text streamed so far, then the retry's 'chunk'
        events stream in."""
        best_result, best_validation, iterations = initial_result, initial_validation, 1

        for attempt in range(self.max_retries):
            if self.validator.passed(best_validation):
                break
            yield {"type": "status", "message":
                   f"Confidence {best_validation.confidence_score:.1f}/10 is below "
                   f"{self.validator.threshold:.1f} — regenerating "
                   f"(attempt {attempt + 1} of {self.max_retries})..."}
            feedback = build_feedback(best_validation, self.validator.threshold)

            candidate = None
            first_chunk = True
            for event in tool.execute_stream(query, feedback=feedback):
                if event["type"] == "result":
                    candidate = event["data"]
                    continue
                if event["type"] == "chunk" and first_chunk:
                    yield {"type": "reset"}
                    first_chunk = False
                yield event
            if candidate is None:
                continue

            yield {"type": "status", "message": "Re-checking confidence..."}
            candidate_validation = self.validator.validate(query, candidate.answer, candidate.context)
            iterations += 1
            logger.info(f"Reflection attempt {attempt + 1}/{self.max_retries}: score "
                        f"{candidate_validation.confidence_score:.1f} (best so far: "
                        f"{best_validation.confidence_score:.1f})")
            if candidate_validation.confidence_score > best_validation.confidence_score:
                best_result, best_validation = candidate, candidate_validation

        return best_result, best_validation, iterations
