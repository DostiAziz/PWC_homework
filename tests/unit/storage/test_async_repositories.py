from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest

from pwc_support.domain.errors import SupportError
from pwc_support.domain.models import (
    DeliveryReceipt,
    MessageKind,
    OutboundMessage,
    ReviewCategory,
    ReviewDecisionKind,
    ReviewRequest,
    RoutingSnapshot,
)
from pwc_support.storage.database import Database
from pwc_support.storage.repositories import (
    InboundRepository,
    MailboxRepository,
    OutboxRepository,
    ReviewRepository,
)


def _database(tmp_path: Path) -> Database:
    database = Database(tmp_path / "operations.sqlite3")
    database.initialize()
    return database


def _snapshot(message_id: str = "message-1") -> RoutingSnapshot:
    return RoutingSnapshot(
        input_fingerprint="hash-1",
        provider_message_id=message_id,
        policy_version="rules-v1",
        deterministic_match=False,
        pre_retrieval_route="plan",
        snapshot_hash="snapshot-hash-1",
    )


def test_claim_recovery_and_snapshot_winner_are_compare_and_set(tmp_path: Path) -> None:
    database = _database(tmp_path)
    repository = InboundRepository(database)
    now = datetime(2026, 9, 3, tzinfo=UTC)

    first = repository.claim("local_chat", "message-1", "hash-1", lease_seconds=30, now=now)
    assert first.status == "claimed"
    assert first.claim is not None
    persisted = repository.record_routing_snapshot(first.claim, _snapshot())
    assert persisted.snapshot_hash == "snapshot-hash-1"

    active = repository.claim("local_chat", "message-1", "hash-1", lease_seconds=30, now=now)
    assert active.status == "in_progress"

    recovered = repository.claim(
        "local_chat",
        "message-1",
        "hash-1",
        lease_seconds=30,
        now=now + timedelta(seconds=31),
    )
    assert recovered.status == "claimed"
    assert recovered.claim is not None
    assert (
        repository.record_routing_snapshot(recovered.claim, _snapshot()).snapshot_hash
        == "snapshot-hash-1"
    )


def test_stale_claim_cannot_replace_snapshot(tmp_path: Path) -> None:
    database = _database(tmp_path)
    repository = InboundRepository(database)
    now = datetime(2026, 9, 3, tzinfo=UTC)
    first = repository.claim("local_chat", "message-1", "hash-1", lease_seconds=1, now=now)
    assert first.claim is not None
    second = repository.claim(
        "local_chat", "message-1", "hash-1", lease_seconds=30, now=now + timedelta(seconds=2)
    )
    assert second.claim is not None
    with pytest.raises(SupportError) as error:
        repository.record_routing_snapshot(first.claim, _snapshot())
    assert error.value.code.value == "STALE_INBOUND_CLAIM"


def test_outbox_and_mailbox_delivery_key_is_idempotent(tmp_path: Path) -> None:
    database = _database(tmp_path)
    outbox = OutboxRepository(database)
    mailbox = MailboxRepository(database)
    message = OutboundMessage(
        delivery_key="inbound:1:service-failure",
        kind=MessageKind.SERVICE_FAILURE,
        recipient="client@example.test",
        thread_id="thread-1",
        subject="Support unavailable",
        body="Please try again later.",
        payload_hash="payload-1",
    )
    outbox.insert(message)
    claimed = outbox.claim_next(worker_id="worker-1", now=datetime(2026, 9, 3, tzinfo=UTC))
    assert claimed is not None
    receipt = mailbox.deliver(message)
    assert isinstance(receipt, DeliveryReceipt)
    assert mailbox.deliver(message) == receipt
    with pytest.raises(SupportError):
        mailbox.deliver(message.model_copy(update={"body": "different", "payload_hash": "other"}))


def test_review_decision_compare_and_set_replays_identical_decision(tmp_path: Path) -> None:
    database = _database(tmp_path)
    reviews = ReviewRepository(database)
    review_id = uuid4()
    run_id = uuid4()
    reviews.create(
        ReviewRequest(
            review_id=review_id,
            case_id="CASE-1",
            run_id=run_id,
            categories=frozenset({ReviewCategory.CONFIDENTIALITY}),
            original_message="A confidential document leaked.",
            response_version=1,
        )
    )
    first = reviews.decide(
        review_id=review_id,
        decision_id=uuid4(),
        expected_version=1,
        reviewer_id="reviewer-1",
        kind=ReviewDecisionKind.SEND_RESPONSE,
        reviewed_text="A specialist will contact you.",
        reason="reviewed",
    )
    replay = reviews.decide(
        review_id=review_id,
        decision_id=first.decision_id,
        expected_version=1,
        reviewer_id="reviewer-1",
        kind=ReviewDecisionKind.SEND_RESPONSE,
        reviewed_text="A specialist will contact you.",
        reason="reviewed",
    )
    assert replay.replayed is True
    assert replay.outbox_key == first.outbox_key
