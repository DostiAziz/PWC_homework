from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import UUID, uuid4

from pwc_support.domain.models import (
    CaseLookupResult,
    CaseRecord,
    CaseRequest,
    CaseStatus,
)

CASE_REFERENCE = re.compile(r"\b(CASE-[0-9A-Z]{4,12})\b")


def find_case_reference(text: str) -> str | None:
    """Detect an existing case reference so the workflow can look it up rather than guess."""
    match = CASE_REFERENCE.search(text.upper())
    return match.group(1) if match else None


class CaseStore(Protocol):
    def create(self, request: CaseRequest, *, case_id: str) -> CaseRecord: ...
    def find(self, case_id: str) -> CaseRecord | None: ...
    def update_status(self, case_id: str, status: CaseStatus) -> CaseRecord: ...


class MailboxAdapter(Protocol):
    def send(self, *, thread_id: str, recipient: str, subject: str, body: str) -> Any: ...


@dataclass(frozen=True, slots=True)
class ToolCall:
    """Auditable record of one non-retrieval tool invocation."""

    tool: str
    operation: str
    ok: bool
    detail: dict[str, Any]

    def as_event(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "operation": self.operation,
            "ok": self.ok,
            "detail": self.detail,
        }


class CaseTool:
    """Non-retrieval workflow tool: read and write durable client case records."""

    name = "case_management"

    def __init__(self, store: CaseStore) -> None:
        self.store = store

    def lookup(self, case_id: str) -> tuple[CaseLookupResult, ToolCall]:
        record = self.store.find(case_id)
        result = CaseLookupResult(found=record is not None, case=record)
        return result, ToolCall(
            tool=self.name,
            operation="lookup",
            ok=record is not None,
            detail={"case_id": case_id, "found": record is not None},
        )

    def open_case(
        self,
        *,
        conversation_id: UUID,
        client_id: str,
        category: str,
        summary: str,
    ) -> tuple[CaseRecord, ToolCall]:
        case_id = f"CASE-{uuid4().hex[:8].upper()}"
        record = self.store.create(
            CaseRequest(
                conversation_id=conversation_id,
                client_id=client_id,
                category=category,
                summary=summary[:1000],
            ),
            case_id=case_id,
        )
        return record, ToolCall(
            tool=self.name,
            operation="open_case",
            ok=True,
            detail={"case_id": record.case_id, "status": record.status.value},
        )

    def close_case(self, case_id: str, status: CaseStatus) -> ToolCall:
        try:
            record = self.store.update_status(case_id, status)
        except KeyError:
            return ToolCall(
                tool=self.name,
                operation="close_case",
                ok=False,
                detail={"case_id": case_id, "error": "unknown_case"},
            )
        return ToolCall(
            tool=self.name,
            operation="close_case",
            ok=True,
            detail={
                "case_id": record.case_id,
                "status": record.status.value,
                "version": record.version,
            },
        )


class MailboxTool:
    """Non-retrieval workflow tool: deliver the final reply on the simulated channel."""

    name = "simulated_mailbox"

    def __init__(self, mailbox: MailboxAdapter) -> None:
        self.mailbox = mailbox

    def deliver(self, *, thread_id: str, recipient: str, subject: str, body: str) -> ToolCall:
        try:
            sent = self.mailbox.send(
                thread_id=thread_id, recipient=recipient, subject=subject, body=body
            )
        except Exception as error:  # the adapter is file-backed and can fail on disk
            return ToolCall(
                tool=self.name,
                operation="deliver",
                ok=False,
                detail={"thread_id": thread_id, "error": type(error).__name__},
            )
        return ToolCall(
            tool=self.name,
            operation="deliver",
            ok=True,
            detail={
                "thread_id": thread_id,
                "provider_message_id": str(getattr(sent, "provider_message_id", "")),
                "recipient": recipient,
            },
        )


@dataclass(frozen=True, slots=True)
class Toolbox:
    """The workflow's non-retrieval tools, injected once at graph construction."""

    case_tool: CaseTool | None = None
    mailbox_tool: MailboxTool | None = None
