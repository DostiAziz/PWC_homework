from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ollama

from pwc_support.adapters.simulated_mailbox import SimulatedMailbox
from pwc_support.config import Settings
from pwc_support.llm.ollama import OllamaEmbedder, OllamaGenerator
from pwc_support.rag.answer import RagAnswerer
from pwc_support.rag.lexical import LexicalIndex
from pwc_support.rag.store import ChromaKnowledgeBase, chroma_client
from pwc_support.storage.database import Database
from pwc_support.storage.repositories import CaseRepository, ReviewRepository, OutboxRepository, MailboxRepository
from pwc_support.storage.retail_repositories import ProductRepository, OrderRepository, RefundRepository, ReturnRepository
from pwc_support.workflow.retail_tools import build_retail_tools
from pwc_support.workflow.retail_actions import RetailApprovalService
from pwc_support.services.review import OutboxDispatcher, ReviewService
from pwc_support.workflow.graph import build_graph
from pwc_support.workflow.risk_classifier import OllamaSemanticRiskClassifier
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
    dispatcher: OutboxDispatcher


def build_runtime(
    settings: Settings | None = None,
    *,
    checkpointer: Any = None,
    enable_interrupt: bool = True,
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
        raise RuntimeError(
            "Chroma knowledge base is empty. Run scripts/ingest_corpus.py first."
        )
    generator = OllamaGenerator(ollama_client, resolved.generation_model)
    database = Database(resolved.operations_db)
    database.initialize()
    retail_database = Database(resolved.retail_db_path or (resolved.data_dir / "state" / "retail.sqlite3"))
    retail_database.initialize()
    cases = CaseRepository(database)
    reviews = ReviewRepository(database)
    mailbox = SimulatedMailbox(resolved.mailbox_path, database=database)
    risk_classifier = (
        OllamaSemanticRiskClassifier.from_settings(resolved)
        if resolved.semantic_classifier_model.strip()
        else None
    )
    graph = build_graph(
        risk_classifier=risk_classifier,
        retail_tools=build_retail_tools(ProductRepository(retail_database), OrderRepository(retail_database), "CUS-1001", ReturnRepository(retail_database, resolved.return_window_days), resolved.refund_auto_approval_limit),
        rag_answerer=RagAnswerer(
            knowledge_base,
            generator,
            minimum_similarity=resolved.minimum_similarity,
            max_selected_hits=resolved.max_selected_hits,
            max_evidence_chars=resolved.max_evidence_chars,
            answer_tokens=resolved.answer_tokens,
        ),
        toolbox=Toolbox(case_tool=CaseTool(cases), mailbox_tool=MailboxTool(mailbox)),
        enable_interrupt=False,
        checkpointer=checkpointer,
        max_planned_tasks=resolved.max_planned_tasks,
    )
    dispatcher = OutboxDispatcher(OutboxRepository(database), MailboxRepository(database), cases)
    retail_approvals = RetailApprovalService(ReturnRepository(retail_database, resolved.return_window_days), RefundRepository(retail_database))
    return Runtime(
        settings=resolved,
        graph=graph,
        database=database,
        cases=cases,
        reviews=reviews,
        mailbox=mailbox,
        knowledge_base=knowledge_base,
        review_service=ReviewService(reviews, dispatcher, retail_approvals, frozenset(resolved.reviewer_ids)),
        dispatcher=dispatcher,
    )
