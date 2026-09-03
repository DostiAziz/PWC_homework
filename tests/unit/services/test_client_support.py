from pathlib import Path
from uuid import UUID, uuid4

from langgraph.checkpoint.memory import InMemorySaver

from pwc_support.domain.models import Channel, ReviewDecision, ReviewDecisionKind
from pwc_support.services.client_support import ClientSupportService
from pwc_support.storage.database import Database
from pwc_support.storage.repositories import ReviewRepository
from pwc_support.workflow.graph import build_graph
from tests.fakes import InMemoryMailbox, fake_rag_answerer, fake_toolbox


def _service(tmp_path: Path, *, mailbox: InMemoryMailbox | None = None) -> ClientSupportService:
    database = Database(tmp_path / "operations.sqlite3")
    database.initialize()
    graph = build_graph(
        rag_answerer=fake_rag_answerer(),
        toolbox=fake_toolbox(mailbox=mailbox),
        enable_interrupt=True,
        checkpointer=InMemorySaver(),
    )
    return ClientSupportService(
        graph, reviews=ReviewRepository(database), database=database, mailbox=mailbox
    )


def test_general_question_is_answered_and_its_run_is_observable(tmp_path: Path) -> None:
    run = _service(tmp_path).submit(body="What services does PwC provide?", client_id="client-1")

    assert run.outcome.status.value == "answered"
    assert run.outcome.citations[0].marker == "[S1]"
    assert run.visited_nodes[:3] == ["intake", "triage", "plan_work"]
    assert run.total_duration_ms >= 0.0


def test_sensitive_message_pauses_and_persists_a_pending_review(tmp_path: Path) -> None:
    service = _service(tmp_path)

    run = service.submit(body="We have a confidential document exposure.", client_id="client-1")

    assert run.interrupted is True
    assert run.outcome.status.value == "pending_review"
    pending = service.pending_reviews()
    assert len(pending) == 1
    assert pending[0].review_id == run.outcome.review_id


def test_specialist_approval_resumes_the_checkpointed_run(tmp_path: Path) -> None:
    service = _service(tmp_path)
    paused = service.submit(body="I want to make a complaint.", client_id="client-1")
    assert paused.outcome.review_id is not None

    resumed = service.resume(
        checkpoint_id=paused.checkpoint_id,
        decision=ReviewDecision(
            review_id=paused.outcome.review_id,
            kind=ReviewDecisionKind.EDIT,
            reviewer_id="specialist-1",
            response_version=1,
            edited_text="A complaints specialist will call you today.",
        ),
    )

    assert resumed.outcome.status.value == "answered"
    assert resumed.outcome.message == "A complaints specialist will call you today."
    assert service.pending_reviews() == []


def test_specialist_rejection_closes_the_run_without_an_answer(tmp_path: Path) -> None:
    service = _service(tmp_path)
    paused = service.submit(body="Please confirm our regulatory position.", client_id="client-1")
    assert paused.outcome.review_id is not None

    resumed = service.resume(
        checkpoint_id=paused.checkpoint_id,
        decision=ReviewDecision(
            review_id=paused.outcome.review_id,
            kind=ReviewDecisionKind.REJECT,
            reviewer_id="specialist-1",
            response_version=1,
        ),
    )

    assert resumed.outcome.status.value == "unable_to_answer"


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
    """One checkpoint namespace per run: otherwise each reply replays the previous trace."""
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

    assert first.checkpoint_id != second.checkpoint_id
    assert first.outcome.conversation_id == second.outcome.conversation_id == conversation
    assert second.visited_nodes == first.visited_nodes
    assert len(second.rag_results) == 1
    assert set(second.state["task_results"]) == {"knowledge-1"}


def test_paused_run_reports_only_the_nodes_it_actually_reached(tmp_path: Path) -> None:
    service = _service(tmp_path)

    run = service.submit(body="We have a confidential data breach.", client_id="client-1")

    # The run pauses inside human_review, so the trace stops at the evidence gathered
    # for the specialist. Nothing downstream of the interrupt is reported as visited.
    assert run.visited_nodes == ["intake", "triage", "gather_evidence"]

