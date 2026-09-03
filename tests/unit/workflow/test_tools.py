from uuid import uuid4

from pwc_support.domain.models import CaseStatus
from pwc_support.workflow.tools import CaseTool, MailboxTool, find_case_reference
from tests.fakes import InMemoryCaseStore, InMemoryMailbox


def test_case_reference_detection_ignores_ordinary_text() -> None:
    assert find_case_reference("Please look at case-abcd1234 again") == "CASE-ABCD1234"
    assert find_case_reference("What services does PwC provide?") is None


def test_case_tool_opens_looks_up_and_closes_a_case() -> None:
    tool = CaseTool(InMemoryCaseStore())

    record, opened = tool.open_case(
        conversation_id=uuid4(), client_id="client-1", category="general_enquiry",
        summary="Question about services",
    )
    found, lookup = tool.lookup(record.case_id)
    closed = tool.close_case(record.case_id, CaseStatus.RESOLVED)

    assert opened.ok and lookup.ok and closed.ok
    assert found.case is not None and found.case.case_id == record.case_id
    assert closed.detail["status"] == "resolved"
    assert closed.detail["version"] == 2


def test_case_tool_reports_an_unknown_case_instead_of_raising() -> None:
    tool = CaseTool(InMemoryCaseStore())

    result, call = tool.lookup("CASE-MISSING1")

    assert result.found is False
    assert call.ok is False
    assert tool.close_case("CASE-MISSING1", CaseStatus.RESOLVED).detail["error"] == "unknown_case"


def test_mailbox_tool_records_a_delivery_on_the_existing_thread() -> None:
    mailbox = InMemoryMailbox()

    call = MailboxTool(mailbox).deliver(
        thread_id="thread-1", recipient="client@example.test",
        subject="Re: Services", body="Here is the published information.",
    )

    assert call.ok is True
    assert call.detail["thread_id"] == "thread-1"
    assert mailbox.sent[0]["body"] == "Here is the published information."


def test_mailbox_tool_reports_a_delivery_failure_without_crashing_the_run() -> None:
    class BrokenMailbox:
        def send(self, **_: object) -> object:
            raise OSError("disk full")

    call = MailboxTool(BrokenMailbox()).deliver(
        thread_id="t", recipient="r", subject="s", body="b"
    )

    assert call.ok is False
    assert call.detail["error"] == "OSError"
