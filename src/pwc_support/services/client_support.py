from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, cast
from uuid import UUID, uuid4

from pwc_support.domain.models import (
    Channel,
    Citation,
    ClientOutcome,
    DraftReply,
    OperationalEvent,
    OutcomeStatus,
    ProposedAction,
    ReviewCategory,
    ReviewRequest,
    RoutingSnapshot,
)
from pwc_support.storage.repositories import InboundRepository


@dataclass(frozen=True, slots=True)
class WorkflowRun:
    """One workflow execution: the client outcome plus the evidence of how it ran."""

    outcome: ClientOutcome
    # The channel's own thread (an email conversation); stable across messages.
    thread_id: str
    review_request: dict[str, Any] | None
    events: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    rag_results: list[dict[str, Any]] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)

    @property
    def visited_nodes(self) -> list[str]:
        return [str(event["node"]) for event in self.events]

    @property
    def total_duration_ms(self) -> float:
        return round(sum(float(event.get("duration_ms") or 0.0) for event in self.events), 2)


class ClientSupportService:
    """Translate channel input into the workflow contract and persist what it produced."""

    def __init__(
        self,
        graph: Any,
        *,
        reviews: Any = None,
        database: Any = None,
        review_service: Any = None,
    ) -> None:
        self.graph = graph
        self.reviews = reviews
        self.database = database
        self.review_service = review_service
        self.inbound = InboundRepository(database) if database is not None else None
        self._identity_runs: dict[tuple[str, str], WorkflowRun] = {}

    def submit(
        self,
        *,
        body: str,
        client_id: str,
        channel: Channel = Channel.CHAT,
        conversation_id: UUID | None = None,
        thread_id: str | None = None,
        subject: str | None = None,
        provider_message_id: str | None = None,
    ) -> WorkflowRun:
        conversation = conversation_id or uuid4()
        run_id = uuid4()
        thread = thread_id or str(conversation)
        provider = "local_chat" if channel is Channel.CHAT else "simulated_email"
        identity = provider_message_id or f"message-{uuid4()}"
        cached = self._identity_runs.get((provider, identity))
        if cached is not None:
            return cached
        claim = None
        inbound = self.inbound
        if inbound is not None:
            claim_result = inbound.claim(
                provider,
                identity,
                hashlib.sha256(body.encode()).hexdigest(),
                lease_seconds=300,
                now=datetime.now(UTC),
            )
            if claim_result.status == "completed":
                if claim_result.outcome is None:
                    raise RuntimeError("completed inbound message has no stored outcome")
                result = self._restore_cached_run(claim_result.outcome, thread_id=thread)
                self._identity_runs[(provider, identity)] = result
                return result
            if claim_result.status == "in_progress" or claim_result.claim is None:
                raise RuntimeError("message is already being processed")
            claim = claim_result.claim
        state = self.graph.invoke(
            {
                "message": {
                    "body": body,
                    "channel": channel.value,
                    "sender_id": client_id,
                    "subject": subject,
                    "conversation_id": str(conversation),
                    "thread_id": thread,
                },
                "run_id": str(run_id),
                "conversation_id": str(conversation),
                "thread_id": thread,
                "client_id": client_id,
                "channel": channel.value,
                "inbound_message_id": identity,
            },
        )
        result = self._finish(
            state,
            conversation=conversation,
            run_id=run_id,
            thread_id=thread,
        )
        self._identity_runs[(provider, identity)] = result
        if claim is not None and inbound is not None:
            snapshot = state.get("routing_snapshot")
            if snapshot is not None:
                inbound.record_routing_snapshot(
                    claim, RoutingSnapshot.model_validate(snapshot)
                )
            inbound.complete(claim, result.outcome.model_dump(mode="json"))
        return result

    def pending_reviews(self) -> list[ReviewRequest]:
        return [] if self.reviews is None else list(self.reviews.list_pending())

    def decide_review(self, **kwargs: Any) -> Any:
        if self.review_service is not None:
            return self.review_service.decide(**kwargs)
        if self.reviews is None:
            raise RuntimeError("review persistence is not configured")
        return self.reviews.decide(**kwargs)

    def _finish(
        self,
        state: dict[str, Any],
        *,
        conversation: UUID,
        run_id: UUID,
        thread_id: str,
    ) -> WorkflowRun:
        review_request = state.get("review_request")
        events = list(state.get("events", []))
        self._persist_events(events, conversation=conversation, run_id=run_id)
        if (
            review_request is not None
            and str(state.get("outcome", {}).get("status")) == OutcomeStatus.PENDING_REVIEW.value
        ):
            self._persist_review(review_request, run_id=run_id)
            outcome = ClientOutcome(
                conversation_id=conversation,
                case_id=review_request.get("case_id"),
                review_id=UUID(str(review_request["review_id"])),
                status=OutcomeStatus.PENDING_REVIEW,
                message=(
                    "Thank you. This enquiry needs a PwC specialist, so it is awaiting review."
                ),
                workflow_run_id=run_id,
            )
            return WorkflowRun(
                outcome=outcome,
                thread_id=thread_id,
                review_request=review_request,
                events=events,
                tool_calls=list(state.get("tool_calls", [])),
                rag_results=list(state.get("rag_results", [])),
                state=state,
            )
        delivery = state.get("delivery", {})
        outcome_state = state.get("outcome", {})
        outcome = ClientOutcome(
            conversation_id=conversation,
            case_id=outcome_state.get("case_id"),
            review_id=(
                UUID(str(outcome_state["review_id"])) if outcome_state.get("review_id") else None
            ),
            status=OutcomeStatus(outcome_state.get("status", OutcomeStatus.FAILED.value)),
            message=str(delivery.get("message") or "No response was produced."),
            citations=tuple(
                Citation.model_validate(item) for item in delivery.get("citations", [])
            ),
            workflow_run_id=run_id,
        )
        return WorkflowRun(
            outcome=outcome,
            thread_id=thread_id,
            review_request=review_request,
            events=events,
            tool_calls=list(state.get("tool_calls", [])),
            rag_results=list(state.get("rag_results", [])),
            state=state,
        )

    def _persist_events(
        self, events: list[dict[str, Any]], *, conversation: UUID, run_id: UUID
    ) -> None:
        if self.database is None:
            return
        for event in events:
            self.database.record_event(
                OperationalEvent(
                    run_id=run_id,
                    conversation_id=conversation,
                    node=str(event["node"]),
                    event_type=str(event["event_type"]),
                    duration_ms=event.get("duration_ms"),
                    details=dict(event.get("details", {})),
                )
            )

    def _persist_review(self, request: dict[str, Any], *, run_id: UUID) -> None:
        if self.reviews is None:
            return
        self.reviews.create(
            ReviewRequest(
                review_id=UUID(str(request["review_id"])),
                case_id=str(request.get("case_id") or "UNASSIGNED"),
                run_id=run_id,
                categories=frozenset(
                    ReviewCategory(category) for category in request.get("categories", [])
                ),
                original_message=str(request.get("original_message", "")),
                evidence=tuple(
                    Citation.model_validate(item) for item in request.get("evidence", [])
                ),
                proposed_reply=(
                    DraftReply.model_validate(request["proposed_reply"])
                    if request.get("proposed_reply")
                    else None
                ),
                proposed_actions=tuple(
                    ProposedAction.model_validate(item)
                    for item in request.get("proposed_actions", [])
                ),
                response_version=int(request.get("response_version", 1)),
                evidence_state=cast(
                    Literal["selected", "insufficient", "conflicting", "unavailable"],
                    str(request.get("evidence_state", "selected")),
                ),
                routing_provenance=(
                    RoutingSnapshot.model_validate(request["routing_provenance"])
                    if request.get("routing_provenance")
                    else None
                ),
                delivery_recipient=request.get("delivery_recipient"),
                delivery_thread_id=request.get("delivery_thread_id"),
                delivery_subject=request.get("delivery_subject"),
            )
        )

    def _restore_cached_run(self, outcome: ClientOutcome, *, thread_id: str) -> WorkflowRun:
        events = (
            [
                event.model_dump(mode="json")
                for event in self.database.list_events(outcome.workflow_run_id)
            ]
            if self.database is not None
            else []
        )
        review_request = None
        if self.reviews is not None and outcome.review_id is not None:
            review = self.reviews.find(outcome.review_id)
            review_request = review.model_dump(mode="json") if review is not None else None
        return WorkflowRun(
            outcome=outcome,
            thread_id=thread_id,
            review_request=review_request,
            events=events,
        )
