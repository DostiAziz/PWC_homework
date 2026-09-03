"""In-memory doubles so workflow tests exercise the real graph without Ollama or Chroma."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pwc_support.domain.models import (
    CaseRecord,
    CaseRequest,
    CaseStatus,
    RagRequest,
    RetrievalBatch,
    RetrievalHit,
)
from pwc_support.rag.answer import RagAnswerer
from pwc_support.workflow.tools import CaseTool, MailboxTool, Toolbox

SERVICES_HIT = RetrievalHit(
    source_id="pwc-global-services",
    chunk_id="chunk-services-1",
    title="PwC global services",
    text="PwC describes assurance, tax, and advisory services for business clients.",
    heading="Services",
    similarity=0.82,
)


class FakeKnowledgeBase:
    """Return evidence only for questions the fake corpus actually covers."""

    def __init__(self, *, hits: tuple[RetrievalHit, ...] = (SERVICES_HIT,)) -> None:
        self.hits = hits
        self.requests: list[RagRequest] = []

    def retrieve(self, request: RagRequest) -> RetrievalBatch:
        self.requests.append(request)
        question = request.question.casefold()
        if any(term in question for term in ("service", "pwc", "advisory", "tax")):
            return RetrievalBatch(hits=self.hits)
        return RetrievalBatch(hits=())


class FakeGenerator:
    def __init__(
        self, template: str = "PwC provides assurance, tax and advisory work. [S1]"
    ) -> None:
        self.template = template
        self.calls: list[str] = []

    def text(
        self, *, system: str, user: str, max_tokens: int = 512, temperature: float = 0.2
    ) -> str:
        self.calls.append(user)
        return self.template


class InMemoryCaseStore:
    def __init__(self) -> None:
        self.records: dict[str, CaseRecord] = {}

    def create(self, request: CaseRequest, *, case_id: str) -> CaseRecord:
        record = self.records.get(case_id) or CaseRecord(
            case_id=case_id,
            conversation_id=request.conversation_id,
            client_id=request.client_id,
            category=request.category,
            status=CaseStatus.PENDING_REVIEW,
            summary=request.summary,
            version=1,
        )
        self.records[case_id] = record
        return record

    def find(self, case_id: str) -> CaseRecord | None:
        return self.records.get(case_id)

    def update_status(self, case_id: str, status: CaseStatus) -> CaseRecord:
        record = self.records[case_id]
        updated = record.model_copy(update={"status": status, "version": record.version + 1})
        self.records[case_id] = updated
        return updated

    def seed(self, case_id: str, *, conversation_id: UUID) -> CaseRecord:
        return self.create(
            CaseRequest(
                conversation_id=conversation_id,
                client_id="client-1",
                category="general_enquiry",
                summary="Existing tracked enquiry",
            ),
            case_id=case_id,
        )


class InMemoryMailbox:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    def send(self, *, thread_id: str, recipient: str, subject: str, body: str) -> Any:
        message = {
            "thread_id": thread_id,
            "recipient": recipient,
            "subject": subject,
            "body": body,
            "provider_message_id": f"message-{len(self.sent) + 1}",
        }
        self.sent.append(message)
        return type("SentMessage", (), message)


def fake_rag_answerer(
    knowledge_base: FakeKnowledgeBase | None = None,
    generator: FakeGenerator | None = None,
) -> RagAnswerer:
    return RagAnswerer(knowledge_base or FakeKnowledgeBase(), generator or FakeGenerator())


def fake_toolbox(
    store: InMemoryCaseStore | None = None, mailbox: InMemoryMailbox | None = None
) -> Toolbox:
    return Toolbox(
        case_tool=CaseTool(store or InMemoryCaseStore()),
        mailbox_tool=MailboxTool(mailbox or InMemoryMailbox()),
    )
