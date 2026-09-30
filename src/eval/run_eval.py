"""
Run this after ingesting your test documents to produce
data/reports/vantage_eval_report.xlsx.

Usage:
    python -m src.eval.run_eval --testset tests/eval_testset.json

Test-set JSON shape (see tests/eval_testset.json for a starter file):
    [
      {"id": "q1", "question": "...", "expected_grounded": 1},
      {"id": "q2", "question": "...", "expected_grounded": 0}
    ]
`expected_grounded` is optional — set 1 if the docs *should* answer it,
0 if it's intentionally out-of-scope/adversarial (used for the confusion
matrix). Leave it out to still get every other metric, ungated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.core.config import ApplicationConfig
from src.embeddings import EmbeddingManager
from src.eval.metrics import RagEvaluator
from src.eval.report_generator import ReportGenerator
from src.llm.ollama_manager import OllamaManager
from src.pipeline.rag_pipeline import RAGPipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--testset", type=str, default="tests/eval_testset.json")
    parser.add_argument("--out", type=str, default="data/reports/vantage_eval_report.xlsx")
    parser.add_argument("--direct_rag", action="store_true")
    args = parser.parse_args()

    config = ApplicationConfig()
    questions = json.loads(Path(args.testset).read_text())

    pipeline = RAGPipeline.get()
    ollama = OllamaManager(host=config.models["ollama"]["host"])
    embedder = EmbeddingManager(model_name=config.models["embeddings"]["model"])
    evaluator = RagEvaluator(ollama=ollama, judge_model=config.models["validation"]["model"],
                              embedder=embedder)

    records = []
    for item in questions:
        result = pipeline.agentic_query(item["question"], direct_rag=args.direct_rag)
        _, chunks, context = pipeline._online.ask_with_details(item["question"])
        record = evaluator.evaluate(
            query_id=item["id"], question=item["question"], answer=result.answer,
            context=context, num_sources_retrieved=len(chunks),
            confidence_score=result.confidence_score, latency_ms=result.latency_ms,
            expected_grounded=item.get("expected_grounded"),
        )
        records.append(record)
        print(f"[{item['id']}] confidence={result.confidence_score:.1f} "
              f"faithfulness={record.faithfulness:.2f} hallucination={record.hallucination_rate:.2f}")

    report_path = ReportGenerator().generate(
        records, Path(args.out), compliance_frameworks=config.compliance["active_frameworks"],
    )
    print(f"\nReport written to: {report_path}")


if __name__ == "__main__":
    main()
