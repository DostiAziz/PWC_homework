from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from langgraph.types import Command

from pwc_support.domain.models import (
    Channel,
    Citation,
    ClientOutcome,
    OperationalEvent,
    OutcomeStatus,
    ReviewCategory,
    ReviewDecision,
    ReviewRequest,
)
from pwc_support.storage.repositories import InboundRepository


@dataclass(frozen=True, slots=True)
class WorkflowRun:
    """One workflow execution: the client outcome plus the evidence of how it ran."""

    outcome: ClientOutcome
    # The channel's own thread (an email conversation); stable across messages.
    thread_id: str
    # This run's LangGraph checkpoint namespace; unique per run, and what resume targets.
    checkpoint_id: str
    interrupted: bool
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
        mailbox: Any = None,
        review_service: Any = None,
    ) -> None:
        self.graph = graph
        self.reviews = reviews
        self.database = database
        self.mailbox = mailbox
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
        # Each enquiry gets its own checkpoint namespace. Sharing one across a
        # conversation would replay and accumulate the previous enquiry's state.
        checkpoint_id = f"run-{run_id}"
        config = {"configurable": {"thread_id": checkpoint_id}}
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
            },
            config,
        )
        result = self._finish(
            state,
            conversation=conversation,
            run_id=run_id,
            thread_id=thread,
            checkpoint_id=checkpoint_id,
        )
        self._identity_runs[(provider, identity)] = result
        if self.inbound is not None:
            claim = self.inbound.claim(
                provider, identity, str(hash(body)), lease_seconds=300, now=datetime.now(UTC)
            ).claim
            if claim is not None:
                self.inbound.complete(claim, result.outcome.model_dump(mode="json"))
        return result

    def resume(self, *, checkpoint_id: str, decision: ReviewDecision) -> WorkflowRun:
        """Resume the checkpointed run that paused for a specialist decision."""
        config = {"configurable": {"thread_id": checkpoint_id}}
        state = self.graph.invoke(
            Command(resume=decision.model_dump(mode="json")), config
        )
        conversation = UUID(str(state.get("conversation_id") or uuid4()))
        run_id = UUID(str(state.get("run_id") or uuid4()))
        if self.reviews is not None:
            self.reviews.mark_decided(decision.review_id)
        return self._finish(
            state,
            conversation=conversation,
            run_id=run_id,
            thread_id=str(state.get("thread_id") or conversation),
            checkpoint_id=checkpoint_id,
        )

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
        checkpoint_id: str,
    ) -> WorkflowRun:
        interrupts = state.get("__interrupt__") or ()
        review_request = state.get("review_request")
        if interrupts and review_request is None:
            review_request = _interrupt_payload(interrupts)
        events = list(state.get("events", []))
        self._persist_events(events, conversation=conversation, run_id=run_id)
        if review_request is not None and (interrupts or str(state.get("outcome", {}).get("status")) == OutcomeStatus.PENDING_REVIEW.value):
            self._persist_review(review_request, run_id=run_id, checkpoint_id=checkpoint_id)
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
                checkpoint_id=checkpoint_id,
                interrupted=bool(interrupts),
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
                UUID(str(outcome_state["review_id"]))
                if outcome_state.get("review_id")
                else None
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
            checkpoint_id=checkpoint_id,
            interrupted=False,
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

    def _persist_review(
        self, request: dict[str, Any], *, run_id: UUID, checkpoint_id: str
    ) -> None:
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
                # Stored so a later session can resume this paused run from the queue.
                checkpoint_id=checkpoint_id,
                response_version=int(request.get("response_version", 1)),
            )
        )


def _interrupt_payload(interrupts: Any) -> dict[str, Any] | None:
    for item in interrupts:
        value = getattr(item, "value", item)
        if isinstance(value, dict):
            return dict(value)
    return None
