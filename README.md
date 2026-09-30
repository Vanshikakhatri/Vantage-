# 🛡️ Vantage — Governed Agentic RAG for Regulated Documents

Offline-first, agentic RAG platform built for document sets that need to be
handled under real compliance regimes (tested against **DPDP Act 2023** and
**NIST SP 800-53 Rev.5**). Runs entirely on local Ollama models — nothing
leaves the machine.

Guardrails, PII scanning, SHA-256 dedup, hybrid + graph retrieval, VLM image
captioning, LLM-judge validation, reflection, and a hash-chained audit log
are all first-class parts of the pipeline, not bolted on. See
**[ARCHITECTURE.md](ARCHITECTURE.md)** for exactly where each piece sits and
why, plus the model-choice comparison table for small-GPU setups.

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

ollama serve &
ollama pull llama3.2:3b                     # chat model
ollama pull qwen2.5:7b-instruct-q4_K_M      # validation / LLM-judge
ollama pull qwen2.5:1.5b-instruct-q4_K_M    # router, query rewriter, graph extraction
ollama pull qwen2.5-coder:7b-instruct-q4_K_M
ollama pull qwen2.5vl:3b                    # VLM for scanned pages / figures (needs Ollama >= 0.7.0)
ollama pull qwen2.5vl:7b                    # optional: automatic VLM fallback with richer captions

streamlit run app.py
```

Open http://localhost:8501, upload a PDF, and ask questions.

### Modes

| Mode | What happens |
|---|---|
| **Auto** (default) | The agent decides for itself whether the question needs plain **Chat**, **Code** help, or a search of your documents (**Agentic RAG**). |
| **RAG** | Documents only. Agentic RAG answers *only* from your uploaded documents and says so when they don't cover the question — it never falls back to general knowledge. |

**Agentic RAG** = retrieve → grade the evidence → if it is weak, rewrite the
search and retrieve again → answer with citations (or decline, in RAG mode).

### Confidence & automatic regeneration

Every answer gets a 0–10 confidence score (LLM judge + citation check + cosine
groundedness; see `src/validation/validation_layer.py`). If it is **below 5.5**,
the answer is regenerated — up to **2 more times** — using feedback about what was
weak, and the best attempt is kept. Fast mode (sidebar) skips this for lower
latency. Thresholds live in `configs/models.yaml` (`validation.confidence_threshold`)
and `configs/application.yaml` (`pipeline.reflection`, `pipeline.agentic_rag`).

## Evaluation report

```bash
python -m src.eval.run_eval --testset tests/eval_testset.json
```

Produces `data/reports/vantage_eval_report.xlsx` — per-query faithfulness /
hallucination rate / relevancy / context precision / groundedness, plus a
confusion matrix (precision/recall/F1/accuracy) against your labeled test
set, and a summary sheet. All aggregate cells are live formulas.

## Project layout

```
vantage/
├── app.py                    Streamlit UI
├── backend.py                UI <-> RAGPipeline adapter
├── configs/                  application.yaml, models.yaml, storage.yaml, guardrails.yaml
├── src/
│   ├── security/              input rule guardrail, semantic guardrail, PII scanner,
│   │                          output guardrail, audit logger, orchestrator
│   ├── ingestion/              SHA registry, ingestion orchestrator, factory
│   ├── preprocessing/          chunking
│   ├── embeddings/             BGE-M3 embedding manager
│   ├── vlm/                    image/scan captioning (Qwen2.5-VL via Ollama)
│   ├── graph/                  GraphRAG builder + multi-hop retriever
│   ├── retrieval/               FAISS, BM25, dense, sparse, hybrid (RRF fusion)
│   ├── reranking/               BGE cross-encoder reranker
│   ├── context/                 context builder (citation-tagged blocks)
│   ├── generation/              prompt builder, query rewriter
│   ├── agent/                   router + chat/code/agentic-rag tools, retrieval grader
│   ├── validation/              LLM-judge + cosine groundedness, reflection
│   ├── pipeline/                OnlinePipeline, AgenticPipeline, RAGPipeline facade
│   ├── eval/                    RAGAS-style metrics, xlsx report generator
│   └── storage/, models/, llm/, core/, utils/
├── tests/eval_testset.json
└── data/                      runtime indices, DB, audit log (git-ignored)
```

## Knowledge Graph workspace

The **🕸️ Knowledge Graph** tab (next to Chat) visualizes the entities and
relationships GraphBuilder extracts from every chunk at ingest time —
zoom/pan the graph, click a node to see its direct relations and every
source chunk (document + page) it's mentioned in. Entities matched in your
current question are highlighted automatically. It reads the same
`data/graph/knowledge_graph.gpickle` the retrieval pipeline uses, so there's
nothing extra to run — it appears as soon as you've ingested a document.

## Running fully offline

Vantage's LLM calls already go through local Ollama (no cloud API, ever).
To also remove the one-time Hugging Face download for the embedding
model, reranker and semantic-guardrail model, drop their weights into a
`models/` folder next to the repo (or wherever `VANTAGE_MODELS_DIR` points),
one subfolder per model:

```
models/
├── bge-m3/                    # BAAI/bge-m3, local copy
├── bge-reranker-v2-m3/        # BAAI/bge-reranker-v2-m3
└── bge-small-en-v1.5/         # BAAI/bge-small-en-v1.5
```

You can populate these once (with internet access) via `huggingface-cli
download BAAI/bge-m3 --local-dir models/bge-m3` (repeat per model). Once
present, Vantage never touches the network for inference — only Ollama
(also local) and your filesystem are used. Without them, the app still
works by falling back to the Hugging Face cache / network.

Relevant env vars (all optional, useful for Docker / multi-machine setups):

| Variable               | Default                  | Purpose                          |
|-------------------------|--------------------------|-----------------------------------|
| `VANTAGE_MODELS_DIR`    | `./models`               | local embedding/reranker weights |
| `VANTAGE_DATA_DIR`      | `./data`                 | sqlite/FAISS/BM25/graph/uploads  |
| `VANTAGE_OLLAMA_HOST`   | `http://localhost:11434` | Ollama endpoint                  |
| `VANTAGE_CHAT_MODEL`    | `llama3.2:3b`            | overrides `configs/models.yaml`  |

## Docker

```bash
docker compose up --build
docker compose exec ollama ollama pull llama3.2:3b
docker compose exec ollama ollama pull qwen2.5:1.5b-instruct-q4_K_M
docker compose exec ollama ollama pull qwen2.5-coder:7b-instruct-q4_K_M
docker compose exec ollama ollama pull qwen2.5vl:3b
```

Open http://localhost:8501. The compose file runs the app + a local Ollama
service as two containers and works unmodified on macOS (Apple Silicon),
Linux and Windows Docker Desktop — no OS-specific steps. Mount local model
weights into `./models` (see above) for zero-download startup.
