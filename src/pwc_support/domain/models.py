from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
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
    OTHER_SENSITIVE_RISK = "other_sensitive_risk"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    CITATION_VERIFICATION_FAILURE = "citation_verification_failure"


class TaskKind(StrEnum):
    KNOWLEDGE_QUERY = "knowledge_query"
    CASE_LOOKUP = "case_lookup"


class TaskStatus(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"
    SKIPPED = "skipped"


class ReviewDecisionKind(StrEnum):
    SEND_RESPONSE = "send_response"
    APPROVE = "approve"
    EDIT = "edit"
    REJECT = "reject"
    REQUEST_REVISION = "request_revision"
    TAKE_OWNERSHIP = "take_ownership"
    APPROVE_REFUND = "approve_refund"
    REJECT_REFUND = "reject_refund"
    APPROVE_RETURN = "approve_return"
    REJECT_RETURN = "reject_return"
    APPROVE_CANCELLATION = "approve_cancellation"
    REJECT_CANCELLATION = "reject_cancellation"
    REQUEST_INFORMATION = "request_information"
    OFFER_REPLACEMENT = "offer_replacement"


class DomainModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CaseStatus(StrEnum):
    OPEN = "open"
    PENDING_REVIEW = "pending_review"
    RESOLVED = "resolved"
    HUMAN_OWNED = "human_owned"
    REJECTED = "rejected"
    DELIVERY_PENDING = "delivery_pending"
    DELIVERY_FAILED = "delivery_failed"


class RetailIntent(StrEnum):
    PRODUCT_SEARCH = "product_search"
    PRODUCT_RECOMMENDATION = "product_recommendation"
    ORDER_STATUS = "order_status"
    RETURN_REQUEST = "return_request"
    REFUND_REQUEST = "refund_request"
    GENERAL_POLICY = "general_policy"


class RetailActionRisk(StrEnum):
    NONE = "none"
    FINANCIAL = "financial"
    IRREVERSIBLE = "irreversible"
    FRAUD_SUSPECTED = "fraud_suspected"
    POLICY_EXCEPTION = "policy_exception"


class ProductSummary(DomainModel):
    product_id: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    name: str
    category: str
    price: Decimal = Field(ge=0)
    currency: Annotated[str, StringConstraints(pattern=r"^[A-Z]{3}$")] = "EUR"
    stock: int = Field(ge=0)
    attributes: dict[str, Any] = Field(default_factory=dict)
    active_offer: dict[str, Any] | None = None


class OrderSummary(DomainModel):
    order_id: str
    customer_id: str
    status: str
    total: Decimal = Field(ge=0)
    currency: str = "EUR"
    items: tuple[dict[str, Any], ...] = ()
    delivered_at: datetime | None = None


class ReturnEligibility(DomainModel):
    eligible: bool
    reason: str
    deadline: datetime | None = None
    refund_amount: Decimal = Field(ge=0)
    risk_flags: tuple[RetailActionRisk, ...] = ()


class ReturnRequest(DomainModel):
    return_id: str
    order_id: str
    item_id: str
    reason: str
    status: str
    idempotency_key: str


class RoutingSnapshot(DomainModel):
    input_fingerprint: str
    provider_message_id: str
    policy_version: str
    deterministic_match: bool
    matched_rule_ids: tuple[str, ...] = ()
    pre_retrieval_route: str
    snapshot_hash: str


class InboundClaim(DomainModel):
    provider: str
    provider_message_id: str
    payload_hash: str
    claim_token: str
    lease_expires_at: datetime
    attempt_count: int = Field(ge=1)


class MessageKind(StrEnum):
    AUTOMATIC_ANSWER = "automatic_answer"
    CASE_ACKNOWLEDGEMENT = "case_acknowledgement"
    REVIEWED_RESPONSE = "reviewed_response"
    SERVICE_FAILURE = "service_failure"


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


class OutboundMessage(DomainModel):
    delivery_key: str
    kind: MessageKind
    case_id: str | None = None
    response_version: int | None = None
    recipient: str
    thread_id: str
    subject: str
    body: str
    payload_hash: str

    @model_validator(mode="after")
    def validate_reviewed_response(self) -> OutboundMessage:
        if self.kind is MessageKind.REVIEWED_RESPONSE:
            if self.case_id is None or self.response_version is None:
                raise ValueError("reviewed responses require case_id and response_version")
        elif self.response_version is not None:
            raise ValueError("response_version is only valid for reviewed responses")
        return self


class DeliveryReceipt(DomainModel):
    delivery_key: str
    provider_message_id: str
    delivered_at: datetime


class InboundClaimResult(DomainModel):
    status: Literal["claimed", "completed", "in_progress"]
    claim: InboundClaim | None = None
    outcome: ClientOutcome | None = None


class ReviewDecisionResult(DomainModel):
    decision_id: UUID
    review_id: UUID
    kind: ReviewDecisionKind
    case_status: CaseStatus
    outbox_key: str | None = None
    replayed: bool = False


class PlannedTask(DomainModel):
    task_id: str
    kind: TaskKind
    input: Annotated[str, StringConstraints(min_length=1, max_length=2000)]


class ProposedAction(DomainModel):
    action_type: Literal[
        "create_case",
        "update_case",
        "schedule",
        "send_document",
        "deliver_email",
        "retail_return",
        "retail_refund",
        "retail_cancellation",
        "offer_replacement",
    ]
    description: Annotated[str, StringConstraints(min_length=1, max_length=1000)]
    requires_review: bool = True
    details: dict[str, Any] = Field(default_factory=dict)


class WorkPlan(DomainModel):
    tasks: tuple[PlannedTask, ...] = Field(min_length=1, max_length=4)
    proposed_actions: tuple[ProposedAction, ...] = ()


class TaskResult(DomainModel):
    task_id: str
    status: TaskStatus
    payload: dict[str, Any] = Field(default_factory=dict)
    error_code: str | None = None


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
    delivery_recipient: str | None = None
    delivery_thread_id: str | None = None
    assigned_reviewer: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    delivery_status: CaseStatus | None = None


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
    response_version: int = Field(ge=1)
    status: Literal["pending", "decided"] = "pending"
    evidence_state: Literal["selected", "insufficient", "conflicting", "unavailable"] = "selected"
    routing_provenance: RoutingSnapshot | None = None
    delivery_recipient: str | None = None
    delivery_thread_id: str | None = None
    delivery_subject: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


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


# --- Agentic specialist routing contracts -----------------------------------
#
# These typed contracts are the shared vocabulary between the intent classifier,
# the specialist subgraphs (OrderAgent, ProductAgent, ReturnRefundAgent, RagAgent),
# and the supervisor workflow that dispatches tasks and joins their results. They
# intentionally live alongside the other domain models rather than under
# `agents/`, so storage-layer repositories can return them without importing
# from the agents package. `pwc_support.agents.contracts` re-exports them.


class SupportIntent(StrEnum):
    PRODUCT_SEARCH = "product_search"
    PRODUCT_RECOMMENDATION = "product_recommendation"
    ORDER_STATUS = "order_status"
    ORDER_CANCELLATION = "order_cancellation"
    RETURN_REQUEST = "return_request"
    REFUND_REQUEST = "refund_request"
    KNOWLEDGE_QUERY = "knowledge_query"


class SpecialistName(StrEnum):
    ORDER = "order"
    PRODUCT = "product"
    RETURN_REFUND = "return_refund"
    RAG = "rag"


class SpecialistStatus(StrEnum):
    """Terminal/clarification vocabulary shared by every specialist subgraph.

    `ORDER_NOT_FOUND` is order-specific: OrderAgent keeps its existing not-found
    status while sharing every other value with ReturnRefundAgent.
    """

    NEEDS_INFORMATION = "needs_information"
    CORRECTION_REQUESTED = "correction_requested"
    REVIEW_REQUIRED = "review_required"
    COMPLETED = "completed"
    UNSUPPORTED = "unsupported"
    TOOL_UNAVAILABLE = "tool_unavailable"
    FAILED = "failed"
    ORDER_NOT_FOUND = "order_not_found"


class PolicyFinding(DomainModel):
    rule_id: Annotated[str, StringConstraints(min_length=1, max_length=100)]
    passed: bool
    message: Annotated[str, StringConstraints(min_length=1, max_length=500)]


class OrderInvestigation(DomainModel):
    """Customer-scoped, verified order facts collected before any policy decision."""

    order_id: str
    customer_id: str
    status: str
    total: Decimal = Field(ge=0)
    currency: Annotated[str, StringConstraints(pattern=r"^[A-Z]{3}$")] = "EUR"
    payment_status: str
    fulfilment_status: str
    shipped_at: datetime | None = None
    version: int = Field(ge=1)
    items: tuple[dict[str, Any], ...] = ()


class ReturnInvestigation(DomainModel):
    """Customer-scoped, verified order-and-item facts for a return or refund request."""

    order_id: str
    customer_id: str
    item_id: str
    order_status: str
    item_description: str
    quantity: int = Field(ge=1)
    unit_price: Decimal = Field(ge=0)
    currency: Annotated[str, StringConstraints(pattern=r"^[A-Z]{3}$")] = "EUR"
    delivered_at: datetime | None = None
    return_window_expires_at: datetime | None = None
    eligible: bool
    risk_flags: tuple[RetailActionRisk, ...] = ()
    version: int = Field(ge=1)


class CancellationReviewPacket(DomainModel):
    """Side-effect-free reviewer handoff for an order cancellation request."""

    conversation_id: UUID
    client_id: str
    original_request: Annotated[str, StringConstraints(min_length=1, max_length=2000)]
    order: OrderInvestigation
    policy_findings: tuple[PolicyFinding, ...] = ()
    missing_information: tuple[str, ...] = ()
    recommended_action: Annotated[str, StringConstraints(min_length=1, max_length=200)]


class ReturnReviewPacket(DomainModel):
    """Side-effect-free reviewer handoff for a return request."""

    conversation_id: UUID
    client_id: str
    original_request: Annotated[str, StringConstraints(min_length=1, max_length=2000)]
    investigation: ReturnInvestigation
    policy_findings: tuple[PolicyFinding, ...] = ()
    missing_information: tuple[str, ...] = ()
    recommended_action: Annotated[str, StringConstraints(min_length=1, max_length=200)]


class RefundReviewPacket(DomainModel):
    """Side-effect-free reviewer handoff for a refund request."""

    conversation_id: UUID
    client_id: str
    original_request: Annotated[str, StringConstraints(min_length=1, max_length=2000)]
    investigation: ReturnInvestigation
    refund_amount: Decimal = Field(ge=0)
    policy_findings: tuple[PolicyFinding, ...] = ()
    missing_information: tuple[str, ...] = ()
    recommended_action: Annotated[str, StringConstraints(min_length=1, max_length=200)]


class CancellationExecutionResult(DomainModel):
    """Result of applying an approved cancellation, distinguishing replayed approvals."""

    order_id: str
    review_id: str
    status: str
    replayed: bool = False


class OfferSummary(DomainModel):
    """A database-computed active offer, priced in code rather than by a model."""

    product_id: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    offer_id: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    list_price: Decimal = Field(ge=0)
    discount_percent: Decimal = Field(ge=0, le=100)
    effective_price: Decimal = Field(ge=0)
    currency: Annotated[str, StringConstraints(pattern=r"^[A-Z]{3}$")] = "EUR"


class ProductRecommendation(DomainModel):
    """A single catalogue match with a database-derived reason, not an invented one."""

    product: ProductSummary
    match_reason: Annotated[str, StringConstraints(min_length=1, max_length=300)]
    rank: int = Field(ge=1)


class RoutingTask(DomainModel):
    """One planner-identified task, routed to exactly one specialist."""

    task_id: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    intent: SupportIntent
    specialist: SpecialistName
    user_text: Annotated[str, StringConstraints(min_length=1, max_length=2000)]
    entities: dict[str, str] = Field(default_factory=dict)
    depends_on: tuple[str, ...] = ()

    @field_validator("entities")
    @classmethod
    def bound_entities(cls, value: dict[str, str]) -> dict[str, str]:
        if len(value) > 20:
            raise ValueError("entities cannot contain more than 20 keys")
        for key, entity_value in value.items():
            if not (1 <= len(key) <= 100):
                raise ValueError("entity keys must be 1-100 characters")
            if len(entity_value) > 500:
                raise ValueError("entity values cannot exceed 500 characters")
        return value


class RoutingDecision(DomainModel):
    """The classifier's task plan, or a decision-level clarification/unsupported outcome."""

    tasks: tuple[RoutingTask, ...] = Field(default_factory=tuple, max_length=8)
    active_task_id: str | None = None
    clarification_required: bool = False
    clarification_reason: str | None = None
    unsupported: bool = False
    classifier_model: str = ""
    classifier_prompt_version: str = ""


class SpecialistResult(DomainModel):
    """The typed value every specialist subgraph returns to the supervisor."""

    specialist: SpecialistName
    task_id: str
    status: SpecialistStatus
    customer_message: str | None = None
    facts: dict[str, Any] = Field(default_factory=dict)
    citations: tuple[Citation, ...] = ()
    review_packet: CancellationReviewPacket | ReturnReviewPacket | RefundReviewPacket | None = None
    missing_information: tuple[str, ...] = ()
    events: tuple[dict[str, Any], ...] = ()


class ConversationMemory(DomainModel):
    """Durable, per-conversation specialist state used to resume clarification loops."""

    conversation_id: UUID
    client_id: str
    active_specialist: SpecialistName | None = None
    active_task_id: str | None = None
    state_json: dict[str, Any] = Field(default_factory=dict)
    version: int = Field(ge=0)
    updated_at: datetime

    @model_validator(mode="after")
    def require_active_task_with_specialist(self) -> ConversationMemory:
        if self.active_specialist is not None and self.active_task_id is None:
            raise ValueError("active_specialist requires an active_task_id")
        return self
