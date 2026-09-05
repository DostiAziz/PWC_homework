from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import ollama

from pwc_support.config import Settings
from pwc_support.llm.ollama import OllamaGateway
from pwc_support.rag.answer import RagAnswerer
from pwc_support.rag.lexical import LexicalIndex
from pwc_support.rag.store import ChromaKnowledgeBase, chroma_client
from pwc_support.services.chat import AgentService
from pwc_support.storage.database import Database
from pwc_support.storage.retail_repositories import OrderRepository, ProductRepository
from pwc_support.workflow.agent_graph import build_agent_graph
from pwc_support.workflow.tools import ToolRegistry

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
    client = ollama.Client(
        host=resolved.ollama_base_url,
        timeout=resolved.request_timeout_seconds,
    )
    model = OllamaGateway(
        client,
        generation_model=resolved.generation_model,
        embedding_model=resolved.embedding_model,
        num_ctx=resolved.num_ctx,
        schema_tokens=resolved.schema_tokens,
        max_parallel_generations=resolved.max_parallel_generations,
    )
    knowledge_base = ChromaKnowledgeBase(
        chroma_client(resolved),
        resolved.collection_name,
        model,
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
