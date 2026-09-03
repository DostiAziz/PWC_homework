from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    StringConstraints,
    field_validator,
    model_validator,
)


class Channel(StrEnum):
    CHAT = "chat"
    SIMULATED_EMAIL = "simulated_email"


class OutcomeStatus(StrEnum):
    ANSWERED = "answered"
    CLARIFICATION_REQUIRED = "clarification_required"
    PENDING_REVIEW = "pending_review"
    UNABLE_TO_ANSWER = "unable_to_answer"
    FAILED = "failed"


class Route(StrEnum):
    PLAN = "plan"
    GREETING = "greeting"
    CLARIFY = "clarify"
    UNSUPPORTED = "unsupported"
    REVIEW = "review"


class ReviewCategory(StrEnum):
    CONFIDENTIALITY = "confidentiality"
    LEGAL_REGULATORY = "legal_regulatory"
    COMPLAINT_ESCALATION = "complaint_escalation"
    EXTERNAL_ACTION = "external_action"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    PROFESSIONAL_JUDGEMENT = "professional_judgement"


class TaskKind(StrEnum):
    KNOWLEDGE_QUERY = "knowledge_query"
    CASE_LOOKUP = "case_lookup"


class TaskStatus(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"
    SKIPPED = "skipped"


class ReviewDecisionKind(StrEnum):
    APPROVE = "approve"
    EDIT = "edit"
    REJECT = "reject"
    REQUEST_REVISION = "request_revision"
    TAKE_OWNERSHIP = "take_ownership"


class CaseStatus(StrEnum):
    OPEN = "open"
    PENDING_REVIEW = "pending_review"
    RESOLVED = "resolved"
    HUMAN_OWNED = "human_owned"
    REJECTED = "rejected"


class DomainModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class IncomingMessage(DomainModel):
    message_id: UUID
    conversation_id: UUID
    channel: Channel
    provider: Literal["local_chat", "simulated_email"]
    provider_message_id: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    provider_thread_id: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    sender_id: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    recipient_id: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    subject: Annotated[str, StringConstraints(max_length=300)] | None = None
    body: Annotated[str, StringConstraints(min_length=1, max_length=8000)]
    language: Literal["en"]
    received_at: datetime
    metadata: dict[str, str] = Field(default_factory=dict)

    @field_validator("body")
    @classmethod
    def normalize_body(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("body cannot be blank")
        return normalized

    @field_validator("received_at")
    @classmethod
    def require_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("received_at must include a timezone")
        return value.astimezone(UTC)


class Citation(DomainModel):
    source_id: str
    chunk_id: str
    marker: str
    title: str
    canonical_url: HttpUrl | None = None
    heading: str
    excerpt: str
    similarity: float = Field(ge=-1.0, le=1.0)


class RetrievalHit(DomainModel):
    source_id: str
    chunk_id: str
    title: str
    text: str
    heading: str
    canonical_url: HttpUrl | None = None
    similarity: float = Field(ge=-1.0, le=1.0)
    language: str = "en"
    source_status: str = "active"
    source_type: str = "public_summary"


class RetrievalBatch(DomainModel):
    hits: tuple[RetrievalHit, ...] = ()
    used_filter_fallback: bool = False
    evidence_conflict: bool = False


class RagRequest(DomainModel):
    question: Annotated[str, StringConstraints(min_length=1, max_length=4000)]
    language: Literal["en"] = "en"
    sector: str | None = None
    service_line: str | None = None
    territory: str | None = None
    triage_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class RagResult(DomainModel):
    status: Literal["answered", "insufficient_evidence"]
    answer: str = ""
    citations: tuple[Citation, ...] = ()
    hits: tuple[RetrievalHit, ...] = ()
    used_filter_fallback: bool = False
    evidence_conflict: bool = False


class EvidenceBundle(DomainModel):
    """Sources retrieved for a specialist to read, with no generated reply.

    The escalation path retrieves evidence but never asks the model to draft an answer,
    so this carries the selected sources on their own.
    """

    citations: tuple[Citation, ...] = ()
    hits: tuple[RetrievalHit, ...] = ()
    used_filter_fallback: bool = False


class PlannedTask(DomainModel):
    task_id: str
    kind: TaskKind
    input: Annotated[str, StringConstraints(min_length=1, max_length=2000)]


class ProposedAction(DomainModel):
    action_type: Literal[
        "create_case", "update_case", "schedule", "send_document", "deliver_email"
    ]
    description: Annotated[str, StringConstraints(min_length=1, max_length=1000)]
    requires_review: bool = True


class WorkPlan(DomainModel):
    tasks: tuple[PlannedTask, ...] = Field(min_length=1, max_length=4)
    proposed_actions: tuple[ProposedAction, ...] = ()


class TaskResult(DomainModel):
    task_id: str
    status: TaskStatus
    payload: dict[str, Any] = Field(default_factory=dict)
    error_code: str | None = None


class TriageDecision(DomainModel):
    intent: str
    sector: str | None = None
    service_line: str | None = None
    territory: str | None = None
    missing_fields: tuple[str, ...] = ()
    review_categories: frozenset[ReviewCategory] = frozenset()
    confidence: float = Field(ge=0.0, le=1.0)
    route: Route


class DraftReply(DomainModel):
    text: Annotated[str, StringConstraints(min_length=1, max_length=1500)]
    citation_markers: tuple[str, ...] = ()
    factual: bool = True
    response_version: int = Field(default=1, ge=1)
    source: Literal["assistant", "human", "system"] = "assistant"


class Verification(DomainModel):
    route: Literal["release", "review", "revise"]
    error_code: str | None = None
    citations: tuple[Citation, ...] = ()
    reason: str | None = None


class CaseRecord(DomainModel):
    case_id: str
    conversation_id: UUID
    client_id: str
    category: str
    status: CaseStatus
    summary: str
    version: int = Field(ge=1)


class CaseLookupResult(DomainModel):
    found: bool
    case: CaseRecord | None = None


class CaseRequest(DomainModel):
    conversation_id: UUID
    client_id: str
    category: str
    summary: Annotated[str, StringConstraints(min_length=1, max_length=1000)]


class ReviewRequest(DomainModel):
    review_id: UUID
    case_id: str
    run_id: UUID
    categories: frozenset[ReviewCategory]
    original_message: str
    proposed_reply: DraftReply | None = None
    proposed_actions: tuple[ProposedAction, ...] = ()
    # Background sources retrieved for the specialist. Empty when the enquiry was
    # escalated after drafting, because the draft carries its own citations.
    evidence: tuple[Citation, ...] = ()
    # The paused run's LangGraph checkpoint namespace. Persisting it is what lets a
    # review raised in one session be resumed from another.
    checkpoint_id: str | None = None
    response_version: int = Field(ge=1)
    status: Literal["pending", "decided"] = "pending"


class ReviewDecision(DomainModel):
    review_id: UUID
    kind: ReviewDecisionKind
    reviewer_id: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    response_version: int = Field(ge=1)
    edited_text: str | None = None
    note: Annotated[str, StringConstraints(max_length=1000)] = ""

    @model_validator(mode="after")
    def validate_edit(self) -> ReviewDecision:
        if self.kind is ReviewDecisionKind.EDIT and not self.edited_text:
            raise ValueError("edited_text is required for edit")
        return self


class ClientOutcome(DomainModel):
    conversation_id: UUID
    case_id: str | None = None
    review_id: UUID | None = None
    status: OutcomeStatus
    message: str
    citations: tuple[Citation, ...] = ()
    workflow_run_id: UUID


class OperationalEvent(DomainModel):
    run_id: UUID
    conversation_id: UUID
    node: str
    event_type: str
    duration_ms: float | None = Field(default=None, ge=0)
    details: dict[str, Any] = Field(default_factory=dict)
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ConversationView(DomainModel):
    conversation_id: UUID
    messages: tuple[IncomingMessage, ...] = ()
    outcome: ClientOutcome | None = None
