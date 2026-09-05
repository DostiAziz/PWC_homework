"""Retrieval and evidence processing subsystem."""

from rag.ingest import ChunkingConfig, CorpusChunk, CorpusDocument
from rag.store import ChromaKnowledgeBase, LexicalIndex, chroma_client
from rag.subgraph import (
    KnowledgeBase,
    RagAnswerer,
    build_rag_graph,
    prepare_query,
    to_rag_result,
)

__all__ = [
    "ChunkingConfig",
    "CorpusChunk",
    "CorpusDocument",
    "ChromaKnowledgeBase",
    "KnowledgeBase",
    "LexicalIndex",
    "chroma_client",
    "RagAnswerer",
    "build_rag_graph",
    "prepare_query",
    "to_rag_result",
]
