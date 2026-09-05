from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config import Settings
from llm.embeddings import embedding_dimension, get_embeddings, validate_collection_dimension
from llm.ollama import get_chat_model
from rag import ChromaKnowledgeBase, LexicalIndex, RagAnswerer, chroma_client
from services.chat import AgentService
from storage.database import Database
from storage.retail_repositories import OrderRepository, ProductRepository
from workflow.agent_graph import build_agent_graph
from workflow.tools import ToolRegistry

RETRIEVAL_CONFIG = Path("config/retrieval.json")


@dataclass(frozen=True, slots=True)
class Runtime:
    settings: Settings
    service: AgentService
    database: Database
    knowledge_base: ChromaKnowledgeBase


def build_runtime(settings: Settings | None = None) -> Runtime:
    resolved = (settings or Settings.from_env()).with_retrieval_config(RETRIEVAL_CONFIG)
    database = Database(resolved.retail_db)
    database.initialize()
    embedder = get_embeddings(model_name=resolved.embedding_model)
    model = get_chat_model(
        model_name=resolved.generation_model,
        base_url=resolved.ollama_base_url,
        request_timeout_seconds=resolved.request_timeout_seconds,
        max_output_tokens=resolved.answer_tokens,
        max_parallel_generations=resolved.max_parallel_generations,
    )
    knowledge_base = ChromaKnowledgeBase(
        chroma_client(resolved),
        resolved.collection_name,
        embedder,
        LexicalIndex(resolved.lexical_db),
        top_k=resolved.top_k,
    )
    validate_collection_dimension(
        knowledge_base.collection,
        embedding_dimension(embedder),
        model_name=resolved.embedding_model,
    )
    if knowledge_base.count() == 0:
        raise RuntimeError("Knowledge base is empty. Run scripts/ingest_corpus.py first.")
    rag_answerer = RagAnswerer(
        knowledge_base,
        model,
        minimum_similarity=resolved.minimum_similarity,
        max_selected_hits=resolved.max_selected_hits,
        max_evidence_chars=resolved.max_evidence_chars,
        answer_tokens=resolved.answer_tokens,
    )
    registry = ToolRegistry(ProductRepository(database), OrderRepository(database), rag_answerer)
    graph = build_agent_graph(model=model, registry=registry)
    return Runtime(
        settings=resolved,
        service=AgentService(graph),
        database=database,
        knowledge_base=knowledge_base,
    )
