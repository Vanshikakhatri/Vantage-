"""backend.py — thin adapter between the Streamlit UI (app.py) and the
RAGPipeline singleton. Keeps all pipeline/session wiring in one place so
app.py stays UI-only."""

from __future__ import annotations

from src.pipeline.rag_pipeline import RAGPipeline


def get_pipeline() -> RAGPipeline:
    return RAGPipeline.get()


def ingest_file(file_bytes: bytes, filename: str):
    return get_pipeline().ingest(file_bytes, filename)


def list_documents():
    return get_pipeline().list_documents()


def delete_document(document_id: str) -> None:
    get_pipeline().delete_document(document_id)


def ask(question: str, direct_rag: bool = False, rag_only: bool = False):
    return get_pipeline().agentic_query(question, direct_rag=direct_rag, rag_only=rag_only)


def ask_stream(question: str, direct_rag: bool = False, rag_only: bool = False):
    return get_pipeline().agentic_query_stream(question, direct_rag=direct_rag, rag_only=rag_only)


def audit_chain_status():
    return get_pipeline().audit_chain_status()
