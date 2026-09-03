from pathlib import Path
from uuid import UUID, uuid4

from pwc_support.domain.models import (
    Channel,
    ReviewDecisionKind,
)
from pwc_support.services.client_support import ClientSupportService
from pwc_support.services.review import OutboxDispatcher, ReviewService
from pwc_support.storage.database import Database
from pwc_support.storage.repositories import (
    CaseRepository,
    MailboxRepository,
    OutboxRepository,
    ReviewRepository,
)
from pwc_support.workflow.graph import build_graph
from pwc_support.workflow.tools import CaseTool, MailboxTool, Toolbox
from tests.fakes import InMemoryMailbox, fake_rag_answerer, fake_toolbox


def _service(tmp_path: Path, *, mailbox: InMemoryMailbox | None = None) -> ClientSupportService:
    database = Database(tmp_path / "operations.sqlite3")
    database.initialize()
    cases = CaseRepository(database)
    reviews = ReviewRepository(database)
    dispatcher = OutboxDispatcher(OutboxRepository(database), MailboxRepository(database), cases)
    graph = build_graph(
        rag_answerer=fake_rag_answerer(),
        toolbox=Toolbox(
            case_tool=CaseTool(cases),
            mailbox_tool=MailboxTool(mailbox) if mailbox is not None else None,
        ),
    )
    return ClientSupportService(
        graph,
        reviews=reviews,
        database=database,
        review_service=ReviewService(
            reviews, dispatcher, allowed_reviewer_ids=frozenset({"specialist-1"})
        ),
    )


def test_general_question_is_answered_and_its_run_is_observable(tmp_path: Path) -> None:
    run = _service(tmp_path).submit(body="What services does PwC provide?", client_id="client-1")

    assert run.outcome.status.value == "answered"
    assert run.outcome.citations[0].marker == "[S1]"
    assert run.visited_nodes[:3] == ["intake", "triage", "plan_work"]
    assert run.total_duration_ms >= 0.0


def test_completed_provider_message_is_not_processed_twice_after_restart(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "operations.sqlite3")
    database.initialize()

    first_service = ClientSupportService(
        build_graph(rag_answerer=fake_rag_answerer(), toolbox=fake_toolbox()),
        database=database,
    )
    second_service = ClientSupportService(
        build_graph(rag_answerer=fake_rag_answerer(), toolbox=fake_toolbox()),
        database=database,
    )

    first = first_service.submit(
        body="What services does PwC provide?",
        client_id="client-1",
        provider_message_id="provider-message-1",
    )
    second = second_service.submit(
        body="What services does PwC provide?",
        client_id="client-1",
        provider_message_id="provider-message-1",
    )

    assert second.outcome == first.outcome
    assert len(database.list_events(first.outcome.workflow_run_id)) == len(first.events)


def test_review_delivery_preserves_email_recipient_thread_and_subject(tmp_path: Path) -> None:
    database = Database(tmp_path / "operations.sqlite3")
    database.initialize()
    cases = CaseRepository(database)
    reviews = ReviewRepository(database)
    dispatcher = OutboxDispatcher(OutboxRepository(database), MailboxRepository(database), cases)
    review_service = ReviewService(
        reviews, dispatcher, allowed_reviewer_ids=frozenset({"specialist-1"})
    )
    graph = build_graph(
        rag_answerer=fake_rag_answerer(),
        toolbox=Toolbox(case_tool=CaseTool(cases)),
    )
    service = ClientSupportService(
        graph,
        reviews=reviews,
        database=database,
        review_service=review_service,
    )

    run = service.submit(
        body="We have a confidential document exposure.",
        client_id="alice@example.test",
        channel=Channel.SIMULATED_EMAIL,
        conversation_id=uuid4(),
        thread_id="email-thread-42",
        subject="Confidentiality concern",
        provider_message_id="provider-message-2",
    )
    review = service.pending_reviews()[0]

    service.decide_review(
        review_id=review.review_id,
        decision_id=uuid4(),
        expected_version=review.response_version,
        reviewer_id="specialist-1",
        kind=ReviewDecisionKind.SEND_RESPONSE,
        reviewed_text="A specialist will contact you shortly.",
    )

    with database.connect() as connection:
        message = connection.execute(
            "SELECT recipient, thread_id, subject FROM outbox_messages WHERE review_id = ?",
            (str(run.outcome.review_id),),
        ).fetchone()
    assert message is not None
    assert dict(message) == {
        "recipient": "alice@example.test",
        "thread_id": "email-thread-42",
        "subject": "Confidentiality concern",
    }


def test_sensitive_message_queues_and_persists_a_pending_review(tmp_path: Path) -> None:
    service = _service(tmp_path)

    run = service.submit(body="We have a confidential document exposure.", client_id="client-1")

    assert run.outcome.status.value == "pending_review"
    pending = service.pending_reviews()
    assert len(pending) == 1
    assert pending[0].review_id == run.outcome.review_id


def test_specialist_approval_is_delivered_asynchronously(tmp_path: Path) -> None:
    service = _service(tmp_path)
    queued = service.submit(body="I want to make a complaint.", client_id="client-1")
    assert queued.outcome.review_id is not None
    review = service.pending_reviews()[0]

    result = service.decide_review(
        review_id=review.review_id,
        decision_id=uuid4(),
        expected_version=review.response_version,
        reviewer_id="specialist-1",
        kind=ReviewDecisionKind.SEND_RESPONSE,
        reviewed_text="A complaints specialist will call you today.",
    )

    assert result.case_status.value == "delivery_pending"
    assert service.pending_reviews() == []


def test_specialist_rejection_closes_the_case_without_delivery(tmp_path: Path) -> None:
    service = _service(tmp_path)
    queued = service.submit(body="Please confirm our regulatory position.", client_id="client-1")
    assert queued.outcome.review_id is not None
    review = service.pending_reviews()[0]

    result = service.decide_review(
        review_id=review.review_id,
        decision_id=uuid4(),
        expected_version=review.response_version,
        reviewer_id="specialist-1",
        kind=ReviewDecisionKind.REJECT,
    )

    assert result.case_status.value == "rejected"


def test_email_submission_keeps_one_thread_and_delivers_through_the_mailbox(
    tmp_path: Path,
) -> None:
    mailbox = InMemoryMailbox()
    service = _service(tmp_path, mailbox=mailbox)
    conversation = uuid4()

    service.submit(
        body="What services does PwC provide?",
        client_id="client@example.test",
        channel=Channel.SIMULATED_EMAIL,
        conversation_id=conversation,
        thread_id="thread-7",
        subject="Services enquiry",
    )

    assert mailbox.sent[0]["thread_id"] == "thread-7"
    assert mailbox.sent[0]["subject"] == "Services enquiry"


def test_node_events_are_persisted_for_every_run(tmp_path: Path) -> None:
    database = Database(tmp_path / "operations.sqlite3")
    database.initialize()
    graph = build_graph(rag_answerer=fake_rag_answerer(), toolbox=fake_toolbox())
    service = ClientSupportService(graph, database=database)

    run = service.submit(body="What services does PwC provide?", client_id="client-1")
    stored = database.list_events(UUID(str(run.outcome.workflow_run_id)))

    assert [event.node for event in stored] == run.visited_nodes


def test_second_enquiry_does_not_inherit_the_first_run_state(tmp_path: Path) -> None:
    """Each provider message gets an independent workflow run."""
    service = _service(tmp_path)
    conversation = uuid4()

    first = service.submit(
        body="What services does PwC provide?",
        client_id="client-1",
        conversation_id=conversation,
    )
    second = service.submit(
        body="Which industries does PwC publish information about?",
        client_id="client-1",
        conversation_id=conversation,
    )

    assert first.outcome.workflow_run_id != second.outcome.workflow_run_id
    assert first.outcome.conversation_id == second.outcome.conversation_id == conversation
    assert second.visited_nodes == first.visited_nodes
    assert len(second.rag_results) == 1
    assert set(second.state["task_results"]) == {"knowledge-1"}


def test_review_run_reports_the_complete_async_path(tmp_path: Path) -> None:
    service = _service(tmp_path)

    run = service.submit(body="We have a confidential data breach.", client_id="client-1")

    assert run.visited_nodes == [
        "intake",
        "triage",
        "gather_evidence",
        "human_review",
        "finalise_case",
    ]


def test_pending_review_is_decidable_from_a_new_service_instance(tmp_path: Path) -> None:
    """The reviewer page is stateless: the durable queue carries delivery metadata."""
    service = _service(tmp_path)
    queued_run = service.submit(
        body="Our confidential PwC advisory report may have leaked.", client_id="client-1"
    )

    queued = service.pending_reviews()[0]

    assert [citation.source_id for citation in queued.evidence] == ["pwc-global-services"]

    restarted = _service(tmp_path)
    result = restarted.decide_review(
        review_id=queued.review_id,
        decision_id=uuid4(),
        expected_version=queued.response_version,
        reviewer_id="specialist-1",
        kind=ReviewDecisionKind.SEND_RESPONSE,
        reviewed_text="A specialist will contact you today.",
    )

    assert result.case_status is not None
    assert queued_run.outcome.review_id == result.review_id
    assert restarted.pending_reviews() == []
