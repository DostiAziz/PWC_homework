from __future__ import annotations

from typing import Any
from uuid import uuid4

from pwc_support.domain.models import Channel, ClientOutcome, OutcomeStatus


class ClientSupportService:
    """Translate client-channel input into the common workflow contract."""

    def __init__(self, graph: Any) -> None:
        self.graph = graph

    def submit(
        self, *, body: str, client_id: str, channel: Channel = Channel.CHAT
    ) -> ClientOutcome:
        conversation_id = uuid4()
        run_id = uuid4()
        result = self.graph.invoke(
            {
                "message": {
                    "body": body,
                    "channel": channel.value,
                    "sender_id": client_id,
                    "conversation_id": str(conversation_id),
                },
                "run_id": str(run_id),
            }
        )
        raw_status = result.get("outcome", {}).get("status", "failed")
        status = OutcomeStatus(raw_status)
        return ClientOutcome(
            conversation_id=conversation_id,
            status=status,
            message=result.get("delivery", {}).get("message", "No response was produced."),
            workflow_run_id=run_id,
        )
