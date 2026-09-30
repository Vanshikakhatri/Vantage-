"""
HybridRetriever — fuses dense + sparse + graph candidates with Reciprocal
Rank Fusion (RRF), the standard way to combine ranked lists that live on
incomparable score scales (cosine similarity vs. BM25 score vs. graph
hop-distance) without hand-tuning a weighted sum.

    RRF(chunk) = sum over each list L containing chunk of  1 / (k + rank_in_L)

GraphRetriever contributes candidates for multi-hop questions: any chunk
reachable within `graph_hop_limit` hops of an entity mentioned in the query
is added to the fusion pool at a fixed synthetic rank, so a multi-hop path
can surface a chunk that neither dense nor sparse search would rank highly
on its own, while an already-strong dense/sparse hit still wins ties.
"""

from __future__ import annotations

from src.core.config import ApplicationConfig
from src.graph.graph_retriever import GraphRetriever
from src.models import EmbeddedChunk
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.sparse_retriever import SparseRetriever


class HybridRetriever:
    def __init__(self, dense: DenseRetriever, sparse: SparseRetriever,
                 graph: GraphRetriever | None = None) -> None:
        self.dense = dense
        self.sparse = sparse
        self.graph = graph or GraphRetriever()
        cfg = ApplicationConfig().pipeline["retrieval"]
        self.dense_top_k = cfg["dense_top_k"]
        self.sparse_top_k = cfg["sparse_top_k"]
        self.rrf_k = cfg["rrf_k"]

    def retrieve(self, query: str, top_k: int = 15) -> list[EmbeddedChunk]:
        dense_hits = self.dense.retrieve(query, top_k=self.dense_top_k)
        sparse_hits = self.sparse.retrieve(query, top_k=self.sparse_top_k)
        graph_chunk_ids = self.graph.multi_hop_chunk_ids(query)

        fused_scores: dict[str, float] = {}
        chunk_lookup: dict[str, EmbeddedChunk] = {}

        for rank, hit in enumerate(dense_hits, start=1):
            cid = hit.chunk.chunk_id
            fused_scores[cid] = fused_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)
            chunk_lookup[cid] = hit

        for rank, hit in enumerate(sparse_hits, start=1):
            cid = hit.chunk.chunk_id
            fused_scores[cid] = fused_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)
            if cid in chunk_lookup:
                # Chunk already has a dense hit — carry its bm25 score over too,
                # so downstream UI can show both signals instead of only whichever
                # retriever happened to see it first.
                chunk_lookup[cid].bm25_score = hit.bm25_score
            else:
                chunk_lookup[cid] = hit

        # Graph hits get a flat synthetic rank boost — they represent "reachable
        # via reasoning", not a similarity score, so they don't dominate strong
        # dense/sparse hits but do surface chunks those two missed entirely.
        for cid in graph_chunk_ids:
            if cid in chunk_lookup:
                fused_scores[cid] = fused_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + 5)

        ranked_ids = sorted(fused_scores, key=lambda c: fused_scores[c], reverse=True)[:top_k]
        results = []
        for cid in ranked_ids:
            hit = chunk_lookup[cid]
            results.append(EmbeddedChunk(chunk=hit.chunk, score=fused_scores[cid],
                                          dense_score=hit.dense_score, bm25_score=hit.bm25_score))
        return results
