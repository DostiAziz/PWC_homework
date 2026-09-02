from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from pwc_support.domain.models import (
    Channel,
    IncomingMessage,
    PlannedTask,
    ReviewCategory,
    ReviewDecision,
    ReviewDecisionKind,
    TaskKind,
    WorkPlan,
)


def test_incoming_message_normalizes_body() -> None:
    message = IncomingMessage(
        message_id=uuid4(),
        conversation_id=uuid4(),
        channel=Channel.CHAT,
        provider="local_chat",
        provider_message_id="m-1",
        provider_thread_id="t-1",
        sender_id="client-1",
        recipient_id="support",
        body="  Hello   there  ",
        language="en",
        received_at=datetime.now(UTC),
    )
    assert message.body == "Hello there"


def test_message_rejects_oversized_body() -> None:
    with pytest.raises(ValidationError):
        IncomingMessage.model_validate(
            {
                "message_id": uuid4(),
                "conversation_id": uuid4(),
                "channel": "chat",
                "provider": "local_chat",
                "provider_message_id": "m-1",
                "provider_thread_id": "t-1",
                "sender_id": "client-1",
                "recipient_id": "support",
                "body": "x" * 8001,
                "language": "en",
                "received_at": datetime.now(UTC),
            }
        )


def test_review_edit_requires_text_and_current_version() -> None:
    with pytest.raises(ValidationError):
        ReviewDecision(
            review_id=uuid4(),
            kind=ReviewDecisionKind.EDIT,
            reviewer_id="r-1",
            response_version=1,
        )


def test_work_plan_is_bounded() -> None:
    task = PlannedTask(
        task_id="t-1",
        kind=TaskKind.KNOWLEDGE_QUERY,
        input="What services are available?",
    )
    plan = WorkPlan(tasks=(task,))
    assert plan.tasks[0].kind.value == "knowledge_query"
    assert ReviewCategory.CONFIDENTIALITY.value == "confidentiality"
