"""
AgenticPipeline — full end-to-end orchestrator.

Flow
----
Query
  -> Input Guardrails  (rule -> PII -> semantic)     [src/security]
      BLOCKED -> audit-logged, return immediately
  -> Query Rewriter                                   [src/generation]
  -> Tool selection
        rag_only=True  -> Agentic RAG, STRICT (answers only from the documents;
                          says "not found" instead of guessing)
        otherwise      -> Agent Router decides {chat, code, rag}
                          (rag = Agentic RAG, lenient: may fall back to general
                          knowledge, clearly labelled)
  -> Tool runs
        rag: retrieve -> grade evidence -> re-search if weak -> answer   [src/agent]
  -> Validation Layer  (LLM judge + citation check + cosine groundedness) [src/validation]
        -> confidence score 0-10 for EVERY answer
  -> Reflection        confidence < threshold (5.5)? regenerate, up to 2 more
                       times, keep the best attempt                      [src/validation]
  -> Output Guardrails (secret leakage -> PII)         [src/security]
  -> Audit Logger      (hash-chained record)           [src/security]
  -> PipelineResult

`direct_rag=True` is the opt-in Fast mode: the validation + regeneration pass is
skipped for lower latency (routing still happens, so Auto stays Auto).
"""

from __future__ import annotations

import time

from src.agent.base_tool import BaseTool, ToolResult
from src.agent.retrieval_grader import RetrievalGrader
from src.agent.router import AgentRouter
from src.agent.tools.chat_tool import ChatTool
from src.agent.tools.code_tool import CodeTool
from src.agent.tools.rag_tool import RAGTool
from src.core.config import ApplicationConfig
from src.core.logging import LoggerManager
from src.embeddings import EmbeddingManager
from src.generation.query_rewriter import QueryRewriter
from src.llm.ollama_manager import OllamaManager
from src.pipeline.online_pipeline import OnlinePipeline
from src.pipeline.pipeline_result import AnswerSource, PipelineResult
from src.security.audit_logger import AuditLogger
from src.security.guardrail_orchestrator import GuardrailOrchestrator
from src.validation.reflection import Reflector
from src.validation.validation_layer import ValidationLayer, ValidationResult

logger = LoggerManager.get_logger()


def _fast_path_validation() -> ValidationResult:
    """Pre-built passing result for Fast mode — skips the LLM-judge round trip
    entirely to cut latency roughly in half."""
    return ValidationResult(
        confidence_score=8.0, correctness=8.0, completeness=8.0,
        has_citations=True, groundedness_cosine=8.0,
        reasoning="Validation skipped in Fast mode.",
    )


def _not_found_validation() -> ValidationResult:
    """Docs-only RAG found nothing relevant. There is no answer to score and
    nothing a regeneration could improve, so validation is bypassed."""
    return ValidationResult(
        confidence_score=0.0, correctness=0.0, completeness=0.0, has_citations=False,
        groundedness_cosine=0.0, has_context=False,
        reasoning="No relevant passages were found in the documents; validation skipped.",
    )


class AgenticPipeline:
    def __init__(self, guardrails: GuardrailOrchestrator, router: AgentRouter,
                 chat_tool: ChatTool, code_tool: CodeTool, rag_tool: RAGTool,
                 validator: ValidationLayer, reflector: Reflector,
                 query_rewriter: QueryRewriter, audit_logger: AuditLogger,
                 rag_only_tool: RAGTool | None = None) -> None:
        self.guardrails = guardrails
        self.router = router
        self.chat_tool = chat_tool
        self.code_tool = code_tool
        self.rag_tool = rag_tool
        self.rag_only_tool = rag_only_tool or rag_tool
        self.validator = validator
        self.reflector = reflector
        self.query_rewriter = query_rewriter
        self.audit = audit_logger
        self._tool_map: dict[str, BaseTool] = {"chat": chat_tool, "code": code_tool, "rag": rag_tool}

    # ---------------------------------------------------------------- non-stream

    def run(self, query: str, direct_rag: bool = False, rag_only: bool = False) -> PipelineResult:
        start = time.perf_counter()
        logger.info(f"AgenticPipeline.run | fast={direct_rag} | rag_only={rag_only} | query={query!r}")

        input_check = self.guardrails.check_input(query)
        if not input_check.allowed:
            return self._blocked_result(query, input_check, start)

        rewrite = self.query_rewriter.rewrite(input_check.normalized_query)
        normalized_query = rewrite.rewritten_query

        tool, tool_name = self._select_tool(normalized_query, rag_only)
        logger.info(f"AgenticPipeline: tool='{tool_name}' (rag_only={rag_only})")

        try:
            tool_result = tool.execute(normalized_query)
        except Exception as exc:
            logger.error(f"Tool execution failed: {exc}")
            return self._error_result(exc, start)

        validation = self._validate(normalized_query, tool_result, fast=direct_rag)
        iterations = 1
        if self._needs_reflection(tool_result, validation, fast=direct_rag):
            tool_result, validation, iterations = self.reflector.reflect(
                tool=tool, query=normalized_query, initial_result=tool_result,
                initial_validation=validation,
            )

        return self._finalize(
            query=query, input_check=input_check, tool_name=tool_name, tool_result=tool_result,
            validation=validation, iterations=iterations, start=start, fast=direct_rag,
            rag_only=rag_only, event="query_answered", rewritten=rewrite.was_rewritten,
        )

    # -------------------------------------------------------------------- stream

    def run_stream(self, query: str, direct_rag: bool = False, rag_only: bool = False):
        """Streaming variant of run() — yields events and ends with
        {'type': 'result', 'data': PipelineResult}.

        Event types: 'status' (progress text), 'route' (which tool the agent chose),
        'chunk' (answer text), 'reset' (discard streamed text — a regeneration is
        about to stream a replacement), 'result' (final PipelineResult)."""
        start = time.perf_counter()

        input_check = self.guardrails.check_input(query)
        if not input_check.allowed:
            yield {"type": "result", "data": self._blocked_result(query, input_check, start)}
            return

        yield {"type": "status", "message": "Rewriting query..."}
        rewrite = self.query_rewriter.rewrite(input_check.normalized_query)
        normalized_query = rewrite.rewritten_query

        if not rag_only:
            yield {"type": "status", "message": "Agent is choosing the right tool..."}
        tool, tool_name = self._select_tool(normalized_query, rag_only)
        yield {"type": "route", "tool": tool_name}

        try:
            tool_result = None
            for event in tool.execute_stream(normalized_query):
                if event["type"] == "result":
                    tool_result = event["data"]
                else:
                    yield event
        except Exception as exc:
            yield {"type": "result", "data": self._error_result(exc, start)}
            return

        if tool_result is None:
            return

        if not direct_rag and tool_result.answer_source != AnswerSource.RAG_NOT_FOUND:
            yield {"type": "status", "message": "Checking answer confidence..."}
        validation = self._validate(normalized_query, tool_result, fast=direct_rag)
        iterations = 1
        if self._needs_reflection(tool_result, validation, fast=direct_rag):
            tool_result, validation, iterations = yield from self.reflector.reflect_stream(
                tool=tool, query=normalized_query, initial_result=tool_result,
                initial_validation=validation,
            )

        yield {"type": "result", "data": self._finalize(
            query=query, input_check=input_check, tool_name=tool_name, tool_result=tool_result,
            validation=validation, iterations=iterations, start=start, fast=direct_rag,
            rag_only=rag_only, event="query_answered_stream", rewritten=rewrite.was_rewritten,
        )}

    # ------------------------------------------------------------------- helpers

    def _select_tool(self, query: str, rag_only: bool) -> tuple[BaseTool, str]:
        if rag_only:
            return self.rag_only_tool, "rag"
        tool_name = self.router.route(query)
        return self._tool_map.get(tool_name, self.rag_tool), tool_name

    def _validate(self, query: str, tool_result: ToolResult, fast: bool) -> ValidationResult:
        if fast:
            return _fast_path_validation()
        if tool_result.answer_source == AnswerSource.RAG_NOT_FOUND:
            return _not_found_validation()
        return self.validator.validate(query, tool_result.answer, tool_result.context)

    def _needs_reflection(self, tool_result: ToolResult, validation: ValidationResult,
                          fast: bool) -> bool:
        return (not fast
                and tool_result.answer_source != AnswerSource.RAG_NOT_FOUND
                and not self.validator.passed(validation))

    def _finalize(self, *, query: str, input_check, tool_name: str, tool_result: ToolResult,
                  validation: ValidationResult, iterations: int, start: float, fast: bool,
                  rag_only: bool, event: str, rewritten: bool) -> PipelineResult:
        output_check = self.guardrails.check_output(tool_result.answer)
        final_answer = output_check.sanitized_answer if output_check.findings else tool_result.answer
        if not output_check.allowed:
            final_answer = "[Answer withheld: contained content blocked by the output guardrail.]"

        result = PipelineResult(
            answer=final_answer, sources=tool_result.sources,
            confidence_score=validation.confidence_score,
            answer_source=tool_result.answer_source, validation_reasoning=validation.reasoning,
            iterations=iterations, latency_ms=self._elapsed(start),
            guardrail_findings=[f"{f.layer}:{f.rule_or_intent}" for f in
                                 input_check.findings + output_check.findings],
            retrieved_chunks=tool_result.retrieved_chunks, agent_steps=list(tool_result.agent_steps),
        )

        # The audit log stores hashes, never plaintext — so agent steps (which quote
        # refined search queries) are counted here, not written out.
        result.audit_record_hash = self.audit.log({
            "event": event, "tool": tool_name, "direct_rag": fast, "rag_only": rag_only,
            "query_hash": self.audit.text_hash(query), "rewritten": rewritten,
            "answer_hash": self.audit.text_hash(final_answer),
            "sources": result.sources, "confidence_score": result.confidence_score,
            "answer_source": result.answer_source, "iterations": iterations,
            "agent_step_count": len(result.agent_steps),
            "guardrail_findings": result.guardrail_findings, "latency_ms": result.latency_ms,
        })

        logger.info(f"AgenticPipeline completed: score={result.confidence_score:.1f}, "
                    f"source={result.answer_source}, iterations={iterations}, "
                    f"latency={result.latency_ms}ms")
        return result

    def _error_result(self, exc: Exception, start: float) -> PipelineResult:
        return PipelineResult(
            answer=f"Sorry, an error occurred: {exc}", answer_source=AnswerSource.LLM,
            confidence_score=0.0, validation_reasoning=str(exc), latency_ms=self._elapsed(start),
        )

    def _blocked_result(self, query: str, input_check, start: float) -> PipelineResult:
        record_hash = self.audit.log({
            "event": "query_blocked", "query_hash": self.audit.text_hash(query),
            "reason": input_check.reason,
            "guardrail_findings": [f"{f.layer}:{f.rule_or_intent}" for f in input_check.findings],
        })
        return PipelineResult(
            answer=f"⛔ {input_check.reason}", answer_source=AnswerSource.BLOCKED,
            confidence_score=0.0, validation_reasoning=input_check.reason,
            latency_ms=self._elapsed(start),
            guardrail_findings=[f"{f.layer}:{f.rule_or_intent}" for f in input_check.findings],
            audit_record_hash=record_hash,
        )

    @staticmethod
    def _elapsed(start: float) -> int:
        return int((time.perf_counter() - start) * 1000)

    # ------------------------------------------------------------------- factory

    @staticmethod
    def create(online_pipeline: OnlinePipeline) -> "AgenticPipeline":
        config = ApplicationConfig()
        models_cfg = config.models

        ollama = OllamaManager(host=models_cfg["ollama"]["host"])
        # Reuse the embedder already loaded for retrieval (via EmbeddingManager's
        # process-wide cache) instead of loading the ~2.3GB bge-m3 weights a
        # second time just for groundedness scoring.
        embedder = EmbeddingManager.get(model_name=models_cfg["embeddings"]["model"],
                                         device=models_cfg["embeddings"].get("device"))

        guardrails = GuardrailOrchestrator()
        router = AgentRouter(ollama_manager=ollama, model=models_cfg["router"]["model"])
        chat_tool = ChatTool(ollama_manager=ollama, model=models_cfg["chat"]["model"])
        code_tool = CodeTool(ollama_manager=ollama, model=models_cfg["code_tool"]["model"])

        # Agentic RAG: one grader shared by two RAG tools that differ only in what
        # they do when nothing relevant is found (strict = decline, lenient = fall back).
        rag_cfg = config.pipeline.get("agentic_rag", {})
        agent_cfg = models_cfg.get("rag_agent", models_cfg["validation"])
        relevance_floor = float(rag_cfg.get("relevance_floor", 0.15))
        grader = RetrievalGrader(
            ollama_manager=ollama, model=agent_cfg["model"],
            strong_score=float(rag_cfg.get("strong_rerank_score", 0.60)),
            relevance_floor=relevance_floor, temperature=float(agent_cfg.get("temperature", 0.0)),
        )
        rag_kwargs = dict(online_pipeline=online_pipeline, ollama_manager=ollama,
                          fallback_model=models_cfg["chat"]["model"], grader=grader,
                          max_search_rounds=int(rag_cfg.get("max_search_rounds", 2)),
                          relevance_floor=relevance_floor)
        rag_tool = RAGTool(strict=False, **rag_kwargs)
        rag_only_tool = RAGTool(strict=True, **rag_kwargs)

        val_cfg = models_cfg["validation"]
        pipe_val_cfg = config.pipeline["validation"]
        validator = ValidationLayer(
            ollama_manager=ollama, model=val_cfg["model"],
            threshold=float(val_cfg.get("confidence_threshold", 5.5)),
            weights=pipe_val_cfg["weights"], embedding_manager=embedder,
            correctness_floor=float(pipe_val_cfg.get("correctness_floor", 4.0)),
        )
        reflector = Reflector(validator=validator, max_retries=config.pipeline["reflection"]["max_retries"])
        query_rewriter = QueryRewriter(ollama_manager=ollama, model=models_cfg["query_rewriter"]["model"])
        audit_logger = AuditLogger()

        return AgenticPipeline(
            guardrails=guardrails, router=router, chat_tool=chat_tool, code_tool=code_tool,
            rag_tool=rag_tool, validator=validator, reflector=reflector,
            query_rewriter=query_rewriter, audit_logger=audit_logger, rag_only_tool=rag_only_tool,
        )
