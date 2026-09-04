from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ollama

from pwc_support.adapters.simulated_mailbox import SimulatedMailbox
from pwc_support.agents.classifier import OllamaIntentClassifier
from pwc_support.agents.order import OllamaOrderPlanner, build_order_agent
from pwc_support.agents.product import build_product_agent
from pwc_support.agents.rag import build_rag_agent
from pwc_support.agents.returns import build_return_refund_agent
from pwc_support.config import Settings
from pwc_support.llm.ollama import OllamaEmbedder, OllamaGenerator
from pwc_support.rag.answer import RagAnswerer
from pwc_support.rag.lexical import LexicalIndex
from pwc_support.rag.store import ChromaKnowledgeBase, chroma_client
from pwc_support.services.review import OutboxDispatcher, ReviewService
from pwc_support.storage.database import Database
from pwc_support.storage.repositories import (
    CaseRepository,
    MailboxRepository,
    OutboxRepository,
    ReviewRepository,
)
from pwc_support.storage.retail_repositories import (
    OrderRepository,
    RefundRepository,
    ReturnRepository,
)
from pwc_support.workflow.graph import build_graph
from pwc_support.workflow.retail_actions import RetailApprovalService
from pwc_support.workflow.retail_tools import DefaultOrderToolbox
from pwc_support.workflow.tools import CaseTool, MailboxTool, Toolbox

RETRIEVAL_CONFIG = Path("config/retrieval.json")


@dataclass(frozen=True, slots=True)
class Runtime:
    """Every collaborator the application service needs, constructed explicitly."""

    settings: Settings
    graph: Any
    database: Database
    cases: CaseRepository
    reviews: ReviewRepository
    mailbox: SimulatedMailbox
    knowledge_base: ChromaKnowledgeBase
    review_service: ReviewService
    mailbox_repository: MailboxRepository


def build_runtime(
    settings: Settings | None = None,
) -> Runtime:
    """Construct the local Ollama, Chroma, SQLite and LangGraph runtime."""
    resolved = (settings or Settings.from_env()).with_retrieval_config(RETRIEVAL_CONFIG)
    ollama_client = ollama.Client(host=resolved.ollama_base_url)
    knowledge_base = ChromaKnowledgeBase(
        chroma_client(resolved),
        resolved.collection_name,
        OllamaEmbedder(ollama_client, resolved.embedding_model),
        LexicalIndex(resolved.lexical_db),
        top_k=resolved.top_k,
    )
    if knowledge_base.count() == 0:
        raise RuntimeError("Chroma knowledge base is empty. Run scripts/ingest_corpus.py first.")
    generator = OllamaGenerator.from_connection(
        host=resolved.ollama_base_url,
        model=resolved.generation_model,
        request_timeout_seconds=resolved.request_timeout_seconds,
        num_ctx=resolved.num_ctx,
        schema_tokens=resolved.schema_tokens,
        max_parallel_generations=resolved.max_parallel_generations,
    )
    database = Database(resolved.operations_db)
    database.initialize()
    retail_database = Database(
        resolved.retail_db_path or (resolved.data_dir / "state" / "retail.sqlite3")
    )
    retail_database.initialize()
    cases = CaseRepository(database)
    reviews = ReviewRepository(database)
    mailbox = SimulatedMailbox(resolved.mailbox_path, database=database)
    classifier = OllamaIntentClassifier(generator, settings=resolved)
    order_agent = build_order_agent(
        tools=DefaultOrderToolbox(OrderRepository(retail_database)),
        planner=OllamaOrderPlanner(generator),
        max_steps=resolved.max_planned_tasks,
    )
    product_agent = build_product_agent(tools=None, planner=None)
    return_refund_agent = build_return_refund_agent(tools=None)
    rag_answerer_instance = RagAnswerer(
        knowledge_base,
        generator,
        minimum_similarity=resolved.minimum_similarity,
        max_selected_hits=resolved.max_selected_hits,
        max_evidence_chars=resolved.max_evidence_chars,
        answer_tokens=resolved.answer_tokens,
    )
    rag_agent = build_rag_agent(rag_answerer=rag_answerer_instance)

    graph = build_graph(
        classifier=classifier,
        order_agent=order_agent,
        product_agent=product_agent,
        return_refund_agent=return_refund_agent,
        rag_agent=rag_agent,
        conversation_memory_repo=None,  # Loaded by ClientSupportService for now
        toolbox=Toolbox(case_tool=CaseTool(cases), mailbox_tool=MailboxTool(mailbox)),
        max_planned_tasks=resolved.max_planned_tasks,
    )
    mailbox_repository = MailboxRepository(database)
    dispatcher = OutboxDispatcher(OutboxRepository(database), mailbox_repository, cases)
    retail_approvals = RetailApprovalService(
        ReturnRepository(retail_database, resolved.return_window_days),
        RefundRepository(retail_database),
    )
    return Runtime(
        settings=resolved,
        graph=graph,
        database=database,
        cases=cases,
        reviews=reviews,
        mailbox=mailbox,
        knowledge_base=knowledge_base,
        review_service=ReviewService(
            reviews, dispatcher, retail_approvals, frozenset(resolved.reviewer_ids)
        ),
        mailbox_repository=mailbox_repository,
    )
