from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from pwc_support.domain.models import (
    CatalogueAction,
    Channel,
    IncomingMessage,
    OrderAction,
    Task,
    TaskKind,
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


def test_catalogue_task_rejects_order_action() -> None:
    with pytest.raises(ValidationError):
        Task(
            task_id="task-1",
            kind=TaskKind.CATALOGUE,
            request="Show offers",
            catalogue_action=CatalogueAction.OFFERS,
            order_action=OrderAction.CANCEL,
        )


def test_order_task_normalizes_order_id() -> None:
    task = Task(
        task_id="task-1",
        kind=TaskKind.ORDER,
        request="Cancel ord-2001",
        order_action=OrderAction.CANCEL,
        order_id="ord-2001",
    )

    assert task.order_id == "ORD-2001"
