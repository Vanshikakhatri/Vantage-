# Vantage — Architecture

## 1. Full pipeline, in order

```
┌─────────────────────────────── OFFLINE / INGESTION ───────────────────────────────┐
│ Upload PDF                                                                         │
│   → SHA-256 dedup check (ShaRegistry)          — skip if already indexed           │
│   → Page parse (PyMuPDF): text pages vs scanned/image pages                        │
│   → VLM captioning (Qwen2.5-VL) for scans + embedded figures/tables-as-images      │
│   → Chunking (recursive splitter, 800/120 overlap) — text AND captions             │
│   → GraphRAG entity/relation extraction per chunk → networkx graph                 │
│   → Embedding (BGE-M3) per chunk                                                   │
│   → Write to FAISS (dense) + BM25 (sparse) + graph + SQLite registry               │
└──────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────── ONLINE / QUERY ────────────────────────────────────┐
│ User query                                                                          │
│   → INPUT GUARDRAILS (src/security/guardrail_orchestrator.py)                       │
│       1. Input Rule Guardrail   — regex + keyword, no model call, runs first        │
│       2. PII Scanner (input)    — masks Aadhaar/PAN/phone/email/etc before anything │
│                                    is embedded, logged, or sent to any LLM          │
│       3. Semantic Guardrail     — embedding-cosine match vs. known-bad intents,     │
│                                    catches paraphrased jailbreaks the regex misses   │
│       BLOCKED → audit-logged, return immediately, nothing downstream runs           │
│   → Query Rewriter (tiny model) — expands vague/follow-up queries                   │
│   → Mode?  Auto → Agent Router (tiny model) decides {chat, code, rag}               │
│            RAG  → docs-only Agentic RAG (strict: never answers from outside docs)   │
│   → [rag tool] AGENTIC RAG loop:                                                    │
│         Hybrid Retrieval: Dense (FAISS/BGE-M3) + Sparse (BM25) + GraphRAG           │
│         multi-hop, fused with RRF → Cross-Encoder Reranker (bge-reranker)           │
│         → Retrieval Grader: reranker score, then LLM only if evidence is weak       │
│         → weak? rewrite search + retrieve again (max 2 rounds), re-rank vs. original │
│         → nothing relevant? RAG mode: "not found"; Auto: labelled general-knowledge │
│         → Context Builder (citation-tagged [S#] blocks) → Prompt → Chat LLM         │
│   → Validation Layer — confidence 0-10 for EVERY answer: LLM-judge (correctness /   │
│                          completeness) + citation check + cosine groundedness        │
│                          (last two only when there are retrieved sources)            │
│   → Reflection — confidence < 5.5? regenerate up to 2 more times with targeted      │
│                  feedback; best attempt kept                                         │
│   (Fast mode, opt-in: skips validation + regeneration for latency)                   │
│   → OUTPUT GUARDRAILS                                                               │
│       4. Output Guardrail (secrets/leakage regex) — API keys, private key blocks,    │
│                          bearer tokens, JWTs — masked even if a source doc had them  │
│       5. PII Scanner (output) — masks PII the model may have copied from a chunk    │
│   → Audit Logger — hash-chained JSONL record (query hash, answer hash, guardrail     │
│                     findings, sources, confidence, iterations, latency)             │
│   → PipelineResult → UI                                                             │
└──────────────────────────────────────────────────────────────────────────────────┘
```

## 2. Where each requested component lives, and why there

| Component | File | Why here |
|---|---|---|
| Input rule guardrail (regex+keyword) | `security/input_rule_guardrail.py` | Runs **first**, before any model/embedding call — zero-cost rejection of known jailbreak phrases and oversized/encoded payloads. |
| Semantic guardrail | `security/semantic_guardrail.py` | Runs **after** the rule layer (so blocked input never pays the embedding cost), catches paraphrases/translations of the same attacks via cosine similarity against reference-intent embeddings. |
| PII scanner | `security/pii_scanner.py` | Used **twice**: on the input query (before it's embedded/logged/sent to any LLM) and on the final answer (in case a retrieved chunk contained PII the model copied verbatim). Same rule-pack, different call sites — DPDP Act "personal data" categories (Aadhaar, PAN, phone, email, DOB, etc). |
| SHA-256 document dedup | `ingestion/sha_registry.py` | First step of `IngestionOrchestrator.ingest_document()` — hash the whole file, look it up in the SQLite registry, skip all downstream parsing/VLM/embedding cost if it's already indexed. |
| Agentic RAG (router + tools + grader) | `agent/router.py`, `agent/tools/*`, `agent/retrieval_grader.py` | Sits between the input guardrails and generation. A tiny model classifies the query into chat/code/rag so general questions and coding help don't pay the retrieval cost. Document questions run the Agentic RAG loop: retrieve → grade evidence (reranker score first, LLM only when ambiguous) → re-search with a reworded query if weak → answer, or decline in docs-only mode. |
| LLM Validation Layer | `validation/validation_layer.py` | Runs on every answer (except in opt-in Fast mode), right after generation. Combines an LLM-judge score with **deterministic** citation and cosine-groundedness checks so a badly-calibrated judge can't singlehandedly wave through a hallucination; a correctness floor vetoes answers the judge deems wrong. Citation/groundedness only count when there are retrieved sources. |
| Reflection | `validation/reflection.py` | Triggered only when validation confidence is below threshold (5.5) — re-runs the same tool with targeted feedback on what the validator flagged (passed separately from the query, so it never pollutes retrieval), keeps the best-scoring attempt, capped at 2 retries. |
| Output guardrail (leakage/secrets) | `security/output_guardrail.py` | Runs **after** validation+reflection, on the final answer text — a defense-in-depth check independent of how good/confident the answer is, because a well-grounded answer can still leak a secret that was sitting in a source PDF. |
| Audit logger | `security/audit_logger.py` | Last step before the UI. Every query — blocked or answered — gets one hash-chained record. Stores **hashes** of query/answer text (not plaintext), so the audit log itself never becomes a new PII surface. `verify_chain()` detects tampering. |
| Ingestion pipeline + SHA record | `ingestion/ingestion_orchestrator.py` | Ties SHA check → parse → VLM caption → chunk → GraphRAG extraction → embed → index, in that order, so every downstream artifact (chunks, graph nodes, vectors) is keyed to a verified `document_id`. |
| Chunking | `preprocessing/chunking.py` | Recursive character splitter (800 chars / 120 overlap), reused for both text chunks and VLM image captions so both flow through one consistent path. |
| Embeddings (BGE-M3) | `embeddings/embedding_manager.py` | Used for (a) chunk embedding at ingest, (b) query embedding at retrieval, (c) the Validation Layer's groundedness-cosine check — one embedder, three call sites. |
| FAISS + BM25 | `retrieval/faiss_manager.py`, `retrieval/bm25_manager.py` | Dense (semantic) and sparse (lexical/exact-term) indices, built and queried independently, fused downstream. |
| Hybrid + sparse retriever | `retrieval/hybrid_retriever.py` | Combines dense + sparse + GraphRAG candidates with **Reciprocal Rank Fusion** — chosen over a hand-weighted score sum because the three sources live on incomparable scales (cosine sim, BM25 score, hop-distance). |
| GraphRAG (multi-hop) | `graph/graph_builder.py`, `graph/graph_retriever.py` | Entities/relations extracted from every chunk at ingest time into a networkx graph. At query time, entities matched in the question are used as start points for a bounded-hop walk (`graph_hop_limit`, default 2) — this is what answers "which vendor named in Clause 12 also appears in the DPIA Annex" style questions that plain vector/BM25 similarity can't connect across chunks. |
| VLM image captioning | `vlm/image_captioner.py` | Called during ingestion for scanned pages (<40 extractable chars) and every embedded raster image ≥64×64px. Output is a normal text chunk (`modality="image_caption"`), so it's retrievable and citable exactly like text. |
| Query rewriter | `generation/query_rewriter.py` | Runs after input guardrails pass, before tool selection — expands vague/follow-up queries into self-contained retrieval queries, on the cheapest model in the stack. |
| Agent Router | `agent/router.py` | 3-way classifier (chat/code/rag) on the tiny model — keeps latency down since it runs on every Auto query. |
| Reranking | `reranking/cross_encoder_reranker.py` | Runs after hybrid fusion, before context building — RRF is rank-based (good recall, weak fine-grained ordering); the cross-encoder reads query+chunk jointly for accurate final ordering. |
| Cosine similarity | `validation/validation_layer.py::_groundedness`, `eval/metrics.py::_groundedness_cosine` | Used at **both** inference time (Validation Layer's groundedness signal) and offline eval time (same computation, reused) — one deterministic, model-judge-independent hallucination signal throughout. |

## 3. Model choices & why (small-GPU targeted)

| Role | Chosen | Size | Why over the alternative |
|---|---|---|---|
| Embeddings | **BAAI/bge-m3** | ~2.3GB fp16 | Kept from the prior stack — still best-in-class open embedder: multilingual (100+ langs, matters for regional-language annexures), 8192-token context, and natively dense+sparse+ColBERT in one model. |
| Reranker | **BAAI/bge-reranker-v2-m3** | ~1.1GB | Lightweight (560M) multilingual cross-encoder, strong accuracy/latency trade-off for a small GPU. |
| Chat / answer generation | **qwen2.5:7b-instruct (q4_K_M)** | ~4.7GB | Swapped from `gemma3:4b` — Qwen2.5-7B-Instruct scores meaningfully higher on instruction-following and long-context grounded QA benchmarks while still fitting an 8GB card at q4 quantization; `gemma3:4b` is a safe fallback if VRAM is tighter (edit `configs/models.yaml`). |
| Router / query rewriter | **qwen2.5:1.5b-instruct (q4_K_M)** | ~1GB | These are classification/rewriting tasks, not reasoning — a 1.5B model is both faster and, being purpose-narrow, just as reliable here as a 7B model, freeing VRAM headroom for the chat + embedding + reranker models running alongside it. |
| Code tool | **qwen2.5-coder:7b-instruct (q4_K_M)** | ~4.7GB | Kept — still a top open code model at this size class. |
| Validation (LLM-judge) | **qwen2.5:7b-instruct** | shared w/ chat | Reuses the already-loaded chat model instead of a dedicated judge model, saving VRAM; JSON-mode scoring keeps it cheap per call. |
| GraphRAG extraction | **qwen2.5:1.5b-instruct** | shared w/ router | Cheap enough to run on every chunk at ingest without becoming the ingestion bottleneck. |
| VLM (image captioning) | **qwen2.5vl:3b** | ~3.2GB | Qwen2.5-VL is far stronger than tiny VLMs at document OCR, tables and charts — what scanned compliance pages need — while the 3B still leaves headroom next to the 7B chat/judge models (Ollama swaps models in and out as needed). `qwen2.5vl:7b` is the automatic fallback (`configs/models.yaml > vlm.fallback_model`) for richer captions. Requires Ollama ≥ 0.7.0. |
| Agentic RAG controller | **qwen2.5:7b-instruct** | shared w/ validation | Grades retrieved evidence and proposes a better search query. Same tag as the judge so Ollama reuses the loaded instance; only invoked when the reranker score is not clearly strong. |
| Semantic guardrail embedder | **BAAI/bge-small-en-v1.5** | ~130MB | Deliberately a *separate*, tiny embedder from the retrieval embedder — so updating the guardrail reference-intent bank never touches the document index, and the safety check stays cheap even on every single query. |

All models are pulled via Ollama except the two SentenceTransformer-family models
(embeddings, reranker, semantic-guardrail embedder), which load through
`sentence-transformers` directly and auto-fall-back to CPU if CUDA init fails —
see `EmbeddingManager.__init__` / `CrossEncoderReranker.__init__`.

## 4. Evaluation methodology

RAG output is open-ended text, so a classic single confusion matrix over
"the answer" doesn't directly apply. Vantage follows the same approach used
by RAGAS/TruLens-style RAG evaluation (`src/eval/metrics.py`):

1. **Continuous metrics per answer** — Faithfulness (⇒ hallucination rate =
   1 − faithfulness), Answer Relevancy, Context Precision, Groundedness
   Cosine (deterministic cross-check independent of the judge LLM).
2. **Binary confusion matrix**, built by thresholding faithfulness
   (`predicted_grounded`) against a curated `expected_grounded` label on a
   labeled test set (`tests/eval_testset.json`) — this is where TP/FP/TN/FN,
   precision, recall, F1 and accuracy come from (`eval/metrics.py::confusion_matrix`,
   materialized as live formulas in the "Confusion Matrix" sheet of the xlsx
   report).

Run `python -m src.eval.run_eval` to generate
`data/reports/vantage_eval_report.xlsx` (Query Results / Confusion Matrix /
Summary sheets, all formula-driven so pasting in a new run updates the
aggregates automatically).
