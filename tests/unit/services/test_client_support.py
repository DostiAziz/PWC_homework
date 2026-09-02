from pwc_support.services.client_support import ClientSupportService
from pwc_support.workflow.graph import build_graph


def test_client_service_returns_pending_outcome_for_sensitive_message() -> None:
    service = ClientSupportService(build_graph(enable_interrupt=False))

    outcome = service.submit(
        body="We have a confidential document exposure.", client_id="client-1"
    )

    assert outcome.status.value == "pending_review"
    assert "specialist" in outcome.message


def test_client_service_returns_answer_for_general_question() -> None:
    service = ClientSupportService(build_graph())

    outcome = service.submit(body="What services does PwC provide?", client_id="client-1")

    assert outcome.status.value == "answered"
    assert outcome.conversation_id
