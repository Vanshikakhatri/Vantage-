"""
GraphRAG — GraphRetriever.

Used by RAGTool for multi-hop questions. Matches query terms against graph
node names (simple containment match — cheap, no extra model call), walks
up to `hop_limit` edges out from each matched node, and collects every
chunk_id reachable on that path. Those chunk_ids are resolved back to full
Chunk objects and merged into the hybrid retriever's candidate list before
reranking, so a multi-hop path and a plain vector hit compete on equal
footing at the reranking stage.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import networkx as nx

from src.core.config import ApplicationConfig


class GraphRetriever:
    def __init__(self) -> None:
        cfg = ApplicationConfig()
        self.hop_limit = cfg.pipeline["retrieval"]["graph_hop_limit"]
        self.graph_path = Path(cfg.storage["graph"]["path"])
        self.graph = self._load()

    def _load(self) -> nx.MultiDiGraph:
        if self.graph_path.exists():
            with open(self.graph_path, "rb") as fh:
                return pickle.load(fh)
        return nx.MultiDiGraph()

    def reload(self) -> None:
        self.graph = self._load()

    def match_entities(self, query: str) -> list[str]:
        lowered = query.lower()
        return [n for n, data in self.graph.nodes(data=True)
                if data.get("type") == "entity" and n.lower() in lowered]

    def multi_hop_chunk_ids(self, query: str) -> set[str]:
        matched = self.match_entities(query)
        chunk_ids: set[str] = set()

        for entity in matched:
            if entity not in self.graph:
                continue
            try:
                reachable = nx.single_source_shortest_path_length(
                    self.graph, entity, cutoff=self.hop_limit
                )
            except Exception:
                continue
            for node in reachable:
                if self.graph.nodes[node].get("type") != "entity" and "-" in str(node):
                    chunk_ids.add(node)   # chunk_ids are uuids (contain "-"), entities don't

        return chunk_ids
