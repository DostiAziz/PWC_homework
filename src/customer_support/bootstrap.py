from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from customer_support.config import Settings
from customer_support.llm.embeddings import get_embeddings
from customer_support.llm.ollama import get_chat_model
from customer_support.rag.answer import RagAnswerer
from customer_support.rag.lexical import LexicalIndex
from customer_support.rag.store import ChromaKnowledgeBase, chroma_client
from customer_support.services.chat import AgentService
from customer_support.storage.database import Database
from customer_support.storage.retail_repositories import OrderRepository, ProductRepository
from customer_support.workflow.agent_graph import build_agent_graph
from customer_support.workflow.tools import ToolRegistry

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
    )
    knowledge_base = ChromaKnowledgeBase(
        chroma_client(resolved),
        resolved.collection_name,
        embedder,
        LexicalIndex(resolved.lexical_db),
        top_k=resolved.top_k,
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
