from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from pwc_support.domain.models import Channel, IncomingMessage, MessageKind, OutboundMessage
from pwc_support.storage.database import Database
from pwc_support.storage.repositories import MailboxRepository


class SimulatedMailbox:
    """File-backed inbox/outbox adapter with stable email thread identifiers."""

    def __init__(self, path: Path, database: Database | None = None) -> None:
        self.path = path
        self.database = database

    def receive(self, *, sender: str, subject: str, body: str) -> IncomingMessage:
        thread_id = f"thread-{uuid4()}"
        message = IncomingMessage(
            message_id=uuid4(),
            conversation_id=uuid4(),
            channel=Channel.SIMULATED_EMAIL,
            provider="simulated_email",
            provider_message_id=f"message-{uuid4()}",
            provider_thread_id=thread_id,
            sender_id=sender,
            recipient_id="support@pwc.test",
            subject=subject,
            body=body,
            language="en",
            received_at=datetime.now(UTC),
        )
        self._append("inbox", message.model_dump(mode="json"))
        if self.database is not None:
            with self.database.connect() as connection:
                connection.execute(
                    "INSERT OR IGNORE INTO inbound_messages ("
                    "provider,provider_message_id,payload_hash,message_json,conversation_id,"
                    "provider_thread_id,sender_id,recipient_id,subject,body,status,created_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        message.provider,
                        message.provider_message_id,
                        hashlib.sha256(message.body.encode()).hexdigest(),
                        message.model_dump_json(),
                        str(message.conversation_id),
                        message.provider_thread_id,
                        message.sender_id,
                        message.recipient_id,
                        message.subject,
                        message.body,
                        "received",
                        message.received_at.isoformat(),
                    ),
                )
        return message

    def send(self, *, thread_id: str, recipient: str, subject: str, body: str) -> IncomingMessage:
        message = IncomingMessage(
            message_id=uuid4(),
            conversation_id=uuid4(),
            channel=Channel.SIMULATED_EMAIL,
            provider="simulated_email",
            provider_message_id=f"message-{uuid4()}",
            provider_thread_id=thread_id,
            sender_id="support@pwc.test",
            recipient_id=recipient,
            subject=subject,
            body=body,
            language="en",
            received_at=datetime.now(UTC),
        )
        self._append("outbox", message.model_dump(mode="json"))
        if self.database is not None:
            OutboxMailbox = MailboxRepository(self.database)
            OutboxMailbox.deliver(
                OutboundMessage(
                    delivery_key=f"email:{message.provider_message_id}",
                    kind=MessageKind.AUTOMATIC_ANSWER,
                    recipient=recipient,
                    thread_id=thread_id,
                    subject=subject,
                    body=body,
                    payload_hash=hashlib.sha256(body.encode()).hexdigest(),
                )
            )
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
