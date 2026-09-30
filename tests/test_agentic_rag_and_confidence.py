"""
Logic tests for: confidence scoring, the <5.5 regeneration gate (max 2 retries),
the retrieval grader, Agentic RAG (retrieve -> grade -> re-search -> answer/decline)
and the pipeline wiring (agent-decided routing, RAG-only, Fast mode).

No Ollama / model weights needed — LLM calls are scripted fakes. Heavy third-party
libs (sentence-transformers, faiss, ...) are stubbed only if they aren't installed.

Run:  python -m unittest discover -s tests -v
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("VANTAGE_DATA_DIR", tempfile.mkdtemp(prefix="vantage_test_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _stub_missing_heavy_deps() -> None:
    for name in ("sentence_transformers", "faiss", "rank_bm25", "fitz"):
        try:
            importlib.import_module(name)
        except ImportError:
            sys.modules[name] = types.ModuleType(name)
    st = sys.modules["sentence_transformers"]
    for attr in ("SentenceTransformer", "CrossEncoder"):
        if not hasattr(st, attr):
            setattr(st, attr, type(attr, (), {}))
    if not hasattr(sys.modules["rank_bm25"], "BM25Okapi"):
        sys.modules["rank_bm25"].BM25Okapi = type("BM25Okapi", (), {})


_stub_missing_heavy_deps()

from src.agent.base_tool import BaseTool, ToolResult                      # noqa: E402
from src.agent.retrieval_grader import RetrievalGrader                     # noqa: E402
from src.agent.tools.rag_tool import RAGTool                               # noqa: E402
from src.models import Chunk, EmbeddedChunk                                # noqa: E402
from src.pipeline.agentic_pipeline import AgenticPipeline                  # noqa: E402
from src.pipeline.pipeline_result import AnswerSource                      # noqa: E402
from src.validation.reflection import Reflector, build_feedback           # noqa: E402
from src.validation.validation_layer import ValidationLayer, ValidationResult  # noqa: E402

WEIGHTS = {"correctness": 0.4, "completeness": 0.2, "citations": 0.15, "groundedness_cosine": 0.25}
THRESHOLD = 5.5
CONTEXT = "[S1] policy.pdf — page 2\nConsent records are kept 5 years.\n\n[S2] policy.pdf — page 3\nMore."


# ------------------------------------------------------------------------ fakes

class FakeOllama:
    """generate(): pops scripted responses (str or Exception) in order."""

    def __init__(self, responses=None, stream_text="fallback answer"):
        self.responses = list(responses or [])
        self.calls: list[dict] = []
        self.stream_text = stream_text

    def generate(self, model, prompt, temperature=0.2, format=None):
        self.calls.append({"model": model, "prompt": prompt, "format": format})
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    def generate_stream(self, model, prompt, temperature=0.2):
        self.calls.append({"model": model, "prompt": prompt, "stream": True})
        yield self.stream_text


def judge(correctness, completeness, reasoning="ok"):
    return json.dumps({"correctness": correctness, "completeness": completeness,
                       "reasoning": reasoning})


class FakeEmbedder:
    def __init__(self, cosine):
        self.cosine = cosine

    def embed_text(self, text):
        return [1.0]

    def cosine_similarity(self, a, b):
        return self.cosine


def make_validator(judge_responses, cosine=0.5):
    return ValidationLayer(FakeOllama(judge_responses), "judge", THRESHOLD, WEIGHTS,
                           embedding_manager=FakeEmbedder(cosine), correctness_floor=4.0)


def ec(cid, score, source="policy.pdf", text="some passage text"):
    return EmbeddedChunk(chunk=Chunk(chunk_id=cid, document_id="d1", source=source, page=1,
                                     text=text), score=score, rerank_score=score)


# ------------------------------------------------------------ 1. confidence score

class TestConfidenceScore(unittest.TestCase):
    def test_grounded_weighted_average(self):
        # cos 0.5 -> groundedness 7.5 ; valid [S1] -> citations 10
        v = make_validator([judge(8, 8)], cosine=0.5)
        r = v.validate("q", "Kept 5 years [S1].", CONTEXT)
        expected = 0.4 * 8 + 0.2 * 8 + 0.15 * 10 + 0.25 * 7.5
        self.assertAlmostEqual(r.confidence_score, expected, delta=0.01)
        self.assertTrue(r.has_citations and r.has_context)
        self.assertTrue(v.passed(r))

    def test_no_context_answers_are_not_penalised_for_missing_citations(self):
        # chat/code: judge says 7/7. Old formula gave 0.4*7+0.2*7+0+0.25*5 = 5.45 -> would
        # have been regenerated for no reason. Now only correctness+completeness count.
        v = make_validator([judge(7, 7)])
        r = v.validate("hi there", "Hello! How can I help?", "")
        self.assertAlmostEqual(r.confidence_score, 7.0, delta=0.01)
        self.assertFalse(r.has_context)
        self.assertTrue(v.passed(r))

    def test_correctness_veto_caps_confidence_below_threshold(self):
        # correctness 2 but perfect citations/grounding: plain average = 5.9 (a PASS).
        v = make_validator([judge(2, 8)], cosine=0.6)          # groundedness 8.0
        r = v.validate("q", "Wrong claim [S1].", CONTEXT)
        plain = 0.4 * 2 + 0.2 * 8 + 0.15 * 10 + 0.25 * 8
        self.assertGreater(plain, THRESHOLD)                    # proves the hole existed
        self.assertLess(r.confidence_score, THRESHOLD)
        self.assertFalse(v.passed(r))
        self.assertIn("capped", r.reasoning)

    def test_citations_must_point_at_real_sources(self):
        v = make_validator([judge(8, 8)])
        r = v.validate("q", "Something [S9].", CONTEXT)          # S9 does not exist
        self.assertFalse(r.has_citations)
        half = make_validator([judge(8, 8)]).validate("q", "A [S1] B [S9].", CONTEXT)
        self.assertTrue(half.has_citations)                      # half-valid -> partial credit

    def test_empty_answer_scores_zero(self):
        r = make_validator([]).validate("q", "   ", CONTEXT)
        self.assertEqual(r.confidence_score, 0.0)

    def test_judge_failure_degrades_to_neutral_not_crash(self):
        v = make_validator([RuntimeError("ollama down")])
        r = v.validate("q", "A [S1].", CONTEXT)
        self.assertGreater(r.confidence_score, 0.0)
        self.assertIn("unavailable", r.reasoning)


# -------------------------------------------------- 2. regeneration gate (Reflector)

class ScriptedTool(BaseTool):
    name = "scripted"

    def __init__(self, answers):
        self.answers = list(answers)
        self.feedbacks: list[str] = []

    def execute(self, query, feedback=""):
        self.feedbacks.append(feedback)
        return ToolResult(answer=self.answers.pop(0), answer_source=AnswerSource.LLM)

    def execute_stream(self, query, feedback=""):
        self.feedbacks.append(feedback)
        text = self.answers.pop(0)
        yield {"type": "status", "message": "working"}
        yield {"type": "chunk", "text": text}
        yield {"type": "result", "data": ToolResult(answer=text, answer_source=AnswerSource.LLM)}


class FakeValidator:
    """validate() returns scripted confidence scores; passed() uses the real 5.5 rule."""
    threshold = THRESHOLD

    def __init__(self, scores):
        self.scores = list(scores)

    def validate(self, query, answer, context):
        return ValidationResult(self.scores.pop(0), 5, 5, False, reasoning="too vague",
                                has_context=False)

    def passed(self, result):
        return result.confidence_score >= self.threshold


def vr(score):
    return ValidationResult(score, 5, 5, False, reasoning="too vague", has_context=False)


class TestRegenerationGate(unittest.TestCase):
    def test_regenerates_twice_then_keeps_best(self):
        tool = ScriptedTool(["retry1", "retry2"])
        refl = Reflector(FakeValidator([4.5, 5.0]), max_retries=2)
        res, val, iters = refl.reflect(tool, "q", ToolResult("first"), vr(4.0))
        self.assertEqual(len(tool.feedbacks), 2)                # exactly 2 regenerations
        self.assertEqual(iters, 3)                              # initial + 2 retries
        self.assertEqual(res.answer, "retry2")                  # best of the three (5.0)
        self.assertEqual(val.confidence_score, 5.0)

    def test_stops_as_soon_as_an_attempt_passes(self):
        tool = ScriptedTool(["retry1", "never used"])
        res, val, iters = Reflector(FakeValidator([6.2]), 2).reflect(tool, "q", ToolResult("first"), vr(4.0))
        self.assertEqual((len(tool.feedbacks), iters, res.answer), (1, 2, "retry1"))

    def test_a_worse_retry_never_replaces_a_better_answer(self):
        tool = ScriptedTool(["worse", "slightly better"])
        res, val, _ = Reflector(FakeValidator([2.0, 4.9]), 2).reflect(tool, "q", ToolResult("first"), vr(4.5))
        self.assertEqual(res.answer, "slightly better")         # 4.9 beats 4.5; 2.0 was ignored

    def test_no_regeneration_when_already_confident(self):
        tool = ScriptedTool([])
        _, _, iters = Reflector(FakeValidator([]), 2).reflect(tool, "q", ToolResult("first"), vr(5.5))
        self.assertEqual((iters, tool.feedbacks), (1, []))      # 5.5 is a pass (>= threshold)

    def test_feedback_is_targeted_and_kept_out_of_the_query(self):
        tool = ScriptedTool(["r1"])
        Reflector(FakeValidator([6.0]), 2).reflect(tool, "the question", ToolResult("first"), vr(4.0))
        self.assertIn("4.0/10", tool.feedbacks[0])
        self.assertIn("completeness", tool.feedbacks[0])
        self.assertIn("too vague", tool.feedbacks[0])

    def test_feedback_mentions_citations_only_for_grounded_answers(self):
        grounded = ValidationResult(4.0, 9, 9, False, 9.0, has_context=True)
        self.assertIn("citations", build_feedback(grounded, THRESHOLD))
        chat = ValidationResult(4.0, 9, 9, False, 0.0, has_context=False)
        self.assertNotIn("citations", build_feedback(chat, THRESHOLD))

    def test_stream_variant_announces_attempts_and_resets_ui_text(self):
        tool = ScriptedTool(["retry1", "retry2"])
        gen = Reflector(FakeValidator([4.5, 6.0]), 2).reflect_stream(tool, "q", ToolResult("first"), vr(4.0))
        events, ret = [], None
        try:
            while True:
                events.append(next(gen))
        except StopIteration as stop:
            ret = stop.value
        statuses = [e["message"] for e in events if e["type"] == "status"]
        self.assertTrue(any("attempt 1 of 2" in m for m in statuses))
        self.assertTrue(any("attempt 2 of 2" in m for m in statuses))
        self.assertEqual(sum(e["type"] == "reset" for e in events), 2)
        # every 'reset' immediately precedes streamed chunk text
        for i, e in enumerate(events):
            if e["type"] == "reset":
                self.assertEqual(events[i + 1]["type"], "chunk")
        self.assertEqual((ret[0].answer, ret[2]), ("retry2", 3))


# ------------------------------------------------------------- 3. retrieval grader

class TestRetrievalGrader(unittest.TestCase):
    def grader(self, responses):
        return RetrievalGrader(FakeOllama(responses), "m", strong_score=0.6, relevance_floor=0.15)

    def test_strong_evidence_skips_llm_entirely(self):
        g = self.grader([])
        grade = g.grade("q", [ec("a", 0.9)], ["q"])
        self.assertTrue(grade.sufficient)
        self.assertFalse(grade.used_llm)
        self.assertEqual(g.ollama.calls, [])

    def test_weak_evidence_asks_llm_and_returns_new_query(self):
        g = self.grader([json.dumps({"sufficient": False, "reason": "off-topic",
                                     "better_query": "data retention period consent"})])
        grade = g.grade("how long do we keep stuff", [ec("a", 0.3)], ["how long do we keep stuff"])
        self.assertFalse(grade.sufficient)
        self.assertEqual(grade.better_query, "data retention period consent")

    def test_repeating_a_tried_query_is_discarded(self):
        g = self.grader([json.dumps({"sufficient": False, "reason": "x", "better_query": "Original Q"})])
        grade = g.grade("original q", [ec("a", 0.3)], ["original q"])
        self.assertEqual(grade.better_query, "")                 # can't loop on the same query

    def test_regeneration_feedback_forces_a_second_look_even_when_strong(self):
        g = self.grader([json.dumps({"sufficient": True, "reason": "fine", "better_query": ""})])
        grade = g.grade("q", [ec("a", 0.95)], ["q"], feedback="answer was incomplete")
        self.assertTrue(grade.used_llm)
        self.assertIn("answer was incomplete", g.ollama.calls[0]["prompt"])

    def test_llm_failure_falls_back_to_reranker_rule(self):
        g = self.grader([RuntimeError("down"), RuntimeError("down")])
        self.assertTrue(g.grade("q", [ec("a", 0.3)], ["q"]).sufficient)      # >= 0.15 floor
        self.assertFalse(g.grade("q", [ec("a", 0.05)], ["q"]).sufficient)    # below floor

    def test_nothing_retrieved_is_insufficient(self):
        g = self.grader([json.dumps({"sufficient": False, "reason": "empty", "better_query": "alt"})])
        self.assertFalse(g.grade("q", [], ["q"]).sufficient)


# --------------------------------------------------------------- 4. Agentic RAG tool

class FakePipeline:
    """retrieve(query) serves scripted chunk lists; records every query it was asked."""
    rerank_top_k = 6

    def __init__(self, results_by_call):
        self.results = list(results_by_call)
        self.queries: list[str] = []
        self.generated_with: list[tuple] = []

    def retrieve(self, query):
        self.queries.append(query)
        return self.results.pop(0)

    def rerank_candidates(self, query, chunks):
        return sorted(chunks, key=lambda c: c.rerank_score, reverse=True)[:self.rerank_top_k]

    def stream_from_chunks(self, query, chunks, feedback=""):
        self.generated_with.append((query, [c.chunk.chunk_id for c in chunks], feedback))
        return iter(["Answer [S1]."]), "[S1] policy.pdf — page 1\ntext"


def make_rag(pipeline, grader_responses, strict=False, max_rounds=2):
    ollama = FakeOllama(grader_responses, stream_text="general knowledge answer")
    grader = RetrievalGrader(ollama, "m", strong_score=0.6, relevance_floor=0.15)
    return RAGTool(pipeline, ollama, "fallback-model", grader, max_search_rounds=max_rounds,
                   relevance_floor=0.15, strict=strict), ollama


def bad_query(n):
    return json.dumps({"sufficient": False, "reason": "weak", "better_query": f"reworded query {n}"})


class TestAgenticRAG(unittest.TestCase):
    def test_strong_first_retrieval_answers_immediately(self):
        pipe = FakePipeline([[ec("a", 0.9)]])
        tool, ollama = make_rag(pipe, [])
        res = tool.execute("what is retention?")
        self.assertEqual(res.answer_source, AnswerSource.RAG)
        self.assertEqual(len(pipe.queries), 1)
        self.assertEqual(ollama.calls, [])                       # no LLM grading cost
        self.assertEqual(res.sources, ["policy.pdf"])
        self.assertIn("Answer", res.answer)

    def test_weak_evidence_triggers_re_search_and_uses_original_question_to_answer(self):
        pipe = FakePipeline([[ec("weak", 0.3)], [ec("good", 0.85)]])
        tool, _ = make_rag(pipe, [bad_query(1)])
        res = tool.execute("how long do we keep stuff")
        self.assertEqual(pipe.queries, ["how long do we keep stuff", "reworded query 1"])
        self.assertEqual(res.answer_source, AnswerSource.RAG)
        query_used, chunk_ids, _ = pipe.generated_with[0]
        self.assertEqual(query_used, "how long do we keep stuff")   # answer targets the USER's question
        self.assertEqual(chunk_ids[0], "good")                     # merged pool, best first
        self.assertIn("weak", chunk_ids)                           # union kept, not replaced
        self.assertTrue(any("Re-searched" in s for s in res.agent_steps))

    def test_search_rounds_are_bounded(self):
        pipe = FakePipeline([[ec("a", 0.3)]] * 3)
        tool, _ = make_rag(pipe, [bad_query(1), bad_query(2), bad_query(3)], max_rounds=2)
        tool.execute("q")
        self.assertEqual(len(pipe.queries), 3)                     # 1 initial + 2 re-searches, no more

    def test_strict_mode_declines_instead_of_guessing(self):
        pipe = FakePipeline([[ec("junk", 0.05)], [ec("junk", 0.04)], [ec("junk", 0.03)]])
        tool, ollama = make_rag(pipe, [bad_query(1), bad_query(2), bad_query(3)], strict=True)
        res = tool.execute("what is the capital of France?")
        self.assertEqual(res.answer_source, AnswerSource.RAG_NOT_FOUND)
        self.assertIn("couldn't find", res.answer)
        self.assertEqual(pipe.generated_with, [])                  # never generated from junk
        self.assertFalse(any(c.get("stream") for c in ollama.calls))  # never used general knowledge
        self.assertEqual(res.sources, [])

    def test_lenient_mode_falls_back_to_general_knowledge_labelled(self):
        pipe = FakePipeline([[ec("junk", 0.05)]] * 3)
        tool, ollama = make_rag(pipe, [bad_query(1), bad_query(2), bad_query(3)], strict=False)
        res = tool.execute("what is the capital of France?")
        self.assertEqual(res.answer_source, AnswerSource.RAG_FALLBACK)
        self.assertEqual(res.answer, "general knowledge answer")
        self.assertEqual(ollama.calls[-1]["model"], "fallback-model")

    def test_empty_index_is_handled(self):
        pipe = FakePipeline([[]])
        tool, _ = make_rag(pipe, [json.dumps({"sufficient": False, "reason": "none", "better_query": ""})],
                           strict=True)
        self.assertEqual(tool.execute("q").answer_source, AnswerSource.RAG_NOT_FOUND)

    def test_regeneration_passes_feedback_to_generator_and_limits_to_one_extra_search(self):
        pipe = FakePipeline([[ec("a", 0.9)], [ec("b", 0.9)], [ec("c", 0.9)]])
        tool, _ = make_rag(pipe, [bad_query(1), bad_query(2)], max_rounds=2)
        tool.execute("q", feedback="Your previous answer scored 4.0/10")
        self.assertLessEqual(len(pipe.queries), 2)                 # 1 initial + at most 1 extra
        self.assertIn("scored 4.0/10", pipe.generated_with[0][2])  # feedback reached the prompt

    def test_stream_emits_status_then_chunks_then_result(self):
        pipe = FakePipeline([[ec("a", 0.9)]])
        tool, _ = make_rag(pipe, [])
        kinds = [e["type"] for e in tool.execute_stream("q")]
        self.assertEqual(kinds[0], "status")
        self.assertIn("chunk", kinds)
        self.assertEqual(kinds[-1], "result")


# ----------------------------------------------------------- 5. pipeline wiring

class FakeGuardrails:
    def check_input(self, q):
        return SimpleNamespace(allowed=True, normalized_query=q, findings=[], reason="")

    def check_output(self, a):
        return SimpleNamespace(allowed=True, findings=[], sanitized_answer=a)


class FakeAudit:
    def __init__(self):
        self.records = []

    def log(self, rec):
        self.records.append(rec)
        return "hash" * 4

    def text_hash(self, t):
        return "h"


class FakeRouter:
    def __init__(self, tool):
        self.tool, self.calls = tool, 0

    def route(self, q):
        self.calls += 1
        return self.tool


class FakeRewriter:
    def rewrite(self, q):
        return SimpleNamespace(rewritten_query=q, was_rewritten=False)


class ChatLike(ScriptedTool):
    name = "chat"


def build_pipeline(router_choice="chat", chat_answers=("a1", "a2", "a3"), judge_responses=(),
                   rag_tool=None, rag_only_tool=None):
    validator = make_validator(list(judge_responses))
    chat = ChatLike(chat_answers)
    router = FakeRouter(router_choice)
    audit = FakeAudit()
    rag_tool = rag_tool or ScriptedTool([])
    pipe = AgenticPipeline(
        guardrails=FakeGuardrails(), router=router, chat_tool=chat, code_tool=ScriptedTool([]),
        rag_tool=rag_tool, validator=validator, reflector=Reflector(validator, max_retries=2),
        query_rewriter=FakeRewriter(), audit_logger=audit, rag_only_tool=rag_only_tool,
    )
    return pipe, chat, router, validator, audit


class TestPipelineWiring(unittest.TestCase):
    def test_low_confidence_answer_is_regenerated_then_accepted(self):
        pipe, chat, _, _, audit = build_pipeline(judge_responses=[judge(3, 3), judge(8, 8)])
        res = pipe.run("explain X")
        self.assertEqual(res.iterations, 2)
        self.assertEqual(res.answer, "a2")
        self.assertGreaterEqual(res.confidence_score, THRESHOLD)
        self.assertEqual(chat.feedbacks[0], "")                    # first attempt: no feedback
        self.assertNotEqual(chat.feedbacks[1], "")                 # retry: targeted feedback
        self.assertEqual(audit.records[-1]["iterations"], 2)

    def test_never_more_than_two_regenerations(self):
        pipe, chat, _, _, _ = build_pipeline(judge_responses=[judge(3, 3)] * 5)
        res = pipe.run("explain X")
        self.assertEqual(len(chat.feedbacks), 3)                   # initial + exactly 2 retries
        self.assertEqual(res.iterations, 3)
        self.assertLess(res.confidence_score, THRESHOLD)           # still low -> honestly reported

    def test_confident_answer_is_not_regenerated(self):
        pipe, chat, _, _, _ = build_pipeline(judge_responses=[judge(9, 9)])
        res = pipe.run("hi")
        self.assertEqual((res.iterations, len(chat.feedbacks)), (1, 1))

    def test_auto_mode_lets_the_router_decide(self):
        pipe, chat, router, _, _ = build_pipeline(router_choice="chat", judge_responses=[judge(9, 9)])
        pipe.run("hello")
        self.assertEqual(router.calls, 1)
        self.assertEqual(len(chat.feedbacks), 1)

    def test_rag_only_bypasses_router_and_uses_strict_tool(self):
        strict = ScriptedTool(["strict answer"])
        pipe, chat, router, _, audit = build_pipeline(
            router_choice="chat", judge_responses=[judge(9, 9)], rag_only_tool=strict)
        res = pipe.run("q", rag_only=True)
        self.assertEqual(res.answer, "strict answer")
        self.assertEqual((router.calls, len(chat.feedbacks)), (0, 0))
        self.assertTrue(audit.records[-1]["rag_only"])

    def test_not_found_skips_validation_and_regeneration(self):
        class NotFoundTool(ScriptedTool):
            def execute(self, query, feedback=""):
                self.feedbacks.append(feedback)
                return ToolResult("nothing found", answer_source=AnswerSource.RAG_NOT_FOUND,
                                  agent_steps=["No relevant passages found"])
        nf = NotFoundTool([])
        pipe, _, _, validator, _ = build_pipeline(rag_only_tool=nf)
        res = pipe.run("q", rag_only=True)
        self.assertEqual(res.answer_source, AnswerSource.RAG_NOT_FOUND)
        self.assertEqual((res.iterations, len(nf.feedbacks)), (1, 1))
        self.assertEqual(validator.ollama.calls, [])               # judge never invoked
        self.assertEqual(res.agent_steps, ["No relevant passages found"])

    def test_fast_mode_skips_validation_but_router_still_decides(self):
        pipe, chat, router, validator, _ = build_pipeline(router_choice="chat")
        res = pipe.run("hello", direct_rag=True)
        self.assertEqual(router.calls, 1)                          # Auto stays Auto
        self.assertEqual(validator.ollama.calls, [])
        self.assertEqual(res.iterations, 1)

    def test_stream_emits_route_event_and_reset_on_regeneration(self):
        pipe, _, _, _, _ = build_pipeline(judge_responses=[judge(3, 3), judge(8, 8)])
        events = list(pipe.run_stream("explain X"))
        kinds = [e["type"] for e in events]
        route = next(e for e in events if e["type"] == "route")
        self.assertEqual(route["tool"], "chat")
        self.assertIn("reset", kinds)
        self.assertLess(kinds.index("route"), kinds.index("chunk"))
        self.assertEqual(kinds[-1], "result")
        final = events[-1]["data"]
        self.assertEqual((final.answer, final.iterations), ("a2", 2))

    def test_stream_rag_only_reports_rag_route(self):
        strict = ScriptedTool(["ans"])
        pipe, *_ = build_pipeline(judge_responses=[judge(9, 9)], rag_only_tool=strict)
        route = next(e for e in pipe.run_stream("q", rag_only=True) if e["type"] == "route")
        self.assertEqual(route["tool"], "rag")


if __name__ == "__main__":
    unittest.main(verbosity=2)
