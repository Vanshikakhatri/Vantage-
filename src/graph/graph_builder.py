"""
GraphRAG — GraphBuilder.

WHERE IT SITS: runs once per chunk, right after chunking/captioning and
alongside (not instead of) embedding — every chunk gets BOTH a vector
(for FAISS) and a small set of (entity, relation, entity) triples pulled
out of it and merged into a persistent networkx graph.

WHY: hybrid dense+sparse retrieval answers "what does the document say
about X" well, but struggles with MULTI-HOP questions like "which vendor
that processes data under Clause 12 is also named in the DPIA Annex" —
answering that means walking Clause12 -> vendor -> DPIA Annex across
chunks that may never co-occur in the same passage, so no single embedding
similarity search will surface all of them together. The graph closes that
gap: GraphRetriever (see graph_retriever.py) starts from entities matched
in the query, walks up to N hops, and pulls in every chunk touching any
node on that path.

Extraction uses the small router-class model (qwen2.5:1.5b) in JSON mode —
cheap enough to run on every chunk at ingest time without becoming the
ingestion bottleneck. Falls back to a lightweight regex/capitalized-phrase
heuristic if the LLM call fails or returns unparsable JSON, so ingestion
never hard-fails because of the graph layer.
"""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

import networkx as nx

from src.core.config import ApplicationConfig
from src.core.logging import LoggerManager
from src.llm.ollama_manager import OllamaManager
from src.models import Chunk

logger = LoggerManager.get_logger()

_EXTRACTION_PROMPT = """Extract entities and relations from the text below for a
knowledge graph. Return ONLY valid JSON, no prose, in this exact shape:
{{"entities": ["Entity A", "Entity B"], "triples": [["Entity A", "relation", "Entity B"]]}}
Keep entity names short (proper nouns, clause/section numbers, defined terms,
organisation names). Extract at most 8 entities and 6 triples.

TEXT:
{text}
"""

_CAPITALIZED_PHRASE = re.compile(r"\b([A-Z][a-zA-Z0-9&,.]*(?:\s+[A-Z][a-zA-Z0-9&,.]*){0,3})\b")


class GraphBuilder:
    def __init__(self, ollama_manager: OllamaManager | None = None) -> None:
        cfg = ApplicationConfig()
        self.model = cfg.models["graph_rag"]["extraction_model"]
        self.ollama = ollama_manager or OllamaManager(host=cfg.models["ollama"]["host"])
        self.graph_path = Path(cfg.storage["graph"]["path"])
        self.graph_path.parent.mkdir(parents=True, exist_ok=True)
        self.graph = self._load()

    def _load(self) -> nx.MultiDiGraph:
        if self.graph_path.exists():
            with open(self.graph_path, "rb") as fh:
                return pickle.load(fh)
        return nx.MultiDiGraph()

    def save(self) -> None:
        with open(self.graph_path, "wb") as fh:
            pickle.dump(self.graph, fh)

    def extract(self, text: str) -> tuple[list[str], list[tuple[str, str, str]]]:
        try:
            raw = self.ollama.generate(
                model=self.model,
                prompt=_EXTRACTION_PROMPT.format(text=text[:2000]),
                temperature=0.0,
                format="json",
            )
            data = json.loads(raw)
            entities = [e.strip() for e in data.get("entities", []) if e.strip()]
            triples = [tuple(t) for t in data.get("triples", []) if len(t) == 3]
            return entities, triples
        except Exception as exc:
            logger.debug(f"GraphBuilder LLM extraction failed, using heuristic fallback: {exc}")
            entities = list(dict.fromkeys(_CAPITALIZED_PHRASE.findall(text)))[:8]
            return entities, []

    def add_chunk(self, chunk: Chunk) -> list[str]:
        """Extract entities/relations from a chunk and merge into the graph.
        Returns the entity list so the caller can store it on Chunk.entities."""
        entities, triples = self.extract(chunk.text)

        for entity in entities:
            self.graph.add_node(entity, type="entity")
            self.graph.add_edge(
                entity, chunk.chunk_id, relation="mentioned_in",
                document_id=chunk.document_id, source=chunk.source, page=chunk.page,
            )

        for head, relation, tail in triples:
            self.graph.add_edge(head, tail, relation=relation, document_id=chunk.document_id)

        return entities

    def remove_document(self, document_id: str) -> None:
        edges_to_remove = [
            (u, v, k) for u, v, k, data in self.graph.edges(keys=True, data=True)
            if data.get("document_id") == document_id
        ]
        self.graph.remove_edges_from([(u, v, k) for u, v, k in edges_to_remove])
        isolated = list(nx.isolates(self.graph))
        self.graph.remove_nodes_from(isolated)
