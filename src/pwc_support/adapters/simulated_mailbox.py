from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from pwc_support.domain.models import Channel, IncomingMessage


class SimulatedMailbox:
    """File-backed inbox/outbox adapter with stable email thread identifiers."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def receive(self, *, sender: str, subject: str, body: str) -> IncomingMessage:
        thread_id = f"thread-{uuid4()}"
        message = IncomingMessage(
            message_id=uuid4(), conversation_id=uuid4(), channel=Channel.SIMULATED_EMAIL,
            provider="simulated_email", provider_message_id=f"message-{uuid4()}",
            provider_thread_id=thread_id, sender_id=sender, recipient_id="support@pwc.test",
            subject=subject, body=body, language="en", received_at=datetime.now(UTC),
        )
        self._append("inbox", message.model_dump(mode="json"))
        return message

    def send(self, *, thread_id: str, recipient: str, subject: str, body: str) -> IncomingMessage:
        message = IncomingMessage(
            message_id=uuid4(), conversation_id=uuid4(), channel=Channel.SIMULATED_EMAIL,
            provider="simulated_email", provider_message_id=f"message-{uuid4()}",
            provider_thread_id=thread_id, sender_id="support@pwc.test", recipient_id=recipient,
            subject=subject, body=body, language="en", received_at=datetime.now(UTC),
        )
        self._append("outbox", message.model_dump(mode="json"))
        return message

    def outbox(self) -> list[IncomingMessage]:
        return [IncomingMessage.model_validate(item) for item in self._read().get("outbox", [])]

    def _append(self, key: str, value: dict[str, Any]) -> None:
        payload = self._read()
        payload.setdefault(key, []).append(value)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        return cast(dict[str, Any], json.loads(self.path.read_text(encoding="utf-8")))
