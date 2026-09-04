import pytest
from pydantic import ValidationError

from pwc_support.domain.models import (
    MessageKind,
    OutboundMessage,
)


def test_reviewed_response_requires_case_and_version() -> None:
    with pytest.raises(ValidationError):
        OutboundMessage(
            delivery_key="d-1",
            kind=MessageKind.REVIEWED_RESPONSE,
            recipient="client@example.test",
            thread_id="thread-1",
            subject="Response",
            body="Approved response",
            payload_hash="hash",
        )

    message = OutboundMessage(
        delivery_key="d-1",
        kind=MessageKind.REVIEWED_RESPONSE,
        case_id="CASE-1",
        response_version=1,
        recipient="client@example.test",
        thread_id="thread-1",
        subject="Response",
        body="Approved response",
        payload_hash="hash",
    )
    assert message.response_version == 1
