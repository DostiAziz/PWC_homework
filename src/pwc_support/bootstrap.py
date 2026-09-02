from __future__ import annotations

from typing import Any

import chromadb
import ollama

from pwc_support.config import Settings
from pwc_support.llm.ollama import OllamaEmbedder, OllamaGenerator
from pwc_support.rag.answer import RagAnswerer
from pwc_support.rag.lexical import LexicalIndex
from pwc_support.rag.store import ChromaKnowledgeBase
from pwc_support.workflow.graph import build_graph


def build_runtime(settings: Settings | None = None) -> Any:
    """Construct the local Ollama, Chroma, and LangGraph runtime explicitly."""
    resolved = settings or Settings.from_env()
    ollama_client = ollama.Client(host=resolved.ollama_base_url)
    if resolved.chroma_mode == "http":
        chroma_client = chromadb.HttpClient(
            host=resolved.chroma_host, port=resolved.chroma_port
        )
    else:
        chroma_client = chromadb.PersistentClient(path=str(resolved.chroma_path))
    knowledge_base = ChromaKnowledgeBase(
        chroma_client,
        resolved.collection_name,
        OllamaEmbedder(ollama_client, resolved.embedding_model),
        LexicalIndex(resolved.lexical_db),
    )
    if knowledge_base.count() == 0:
        raise RuntimeError(
            "Chroma knowledge base is empty. Run scripts/ingest_corpus.py first."
        )
    generator = OllamaGenerator(ollama_client, resolved.generation_model)
    return build_graph(rag_answerer=RagAnswerer(knowledge_base, generator))
