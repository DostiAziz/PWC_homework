from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from pwc_support.storage.checkpoints import sqlite_checkpointer
from pwc_support.workflow.graph import build_graph
from tests.fakes import fake_rag_answerer, fake_toolbox


def test_sensitive_case_pauses_and_resumes_after_review() -> None:
    graph = build_graph(
        rag_answerer=fake_rag_answerer(),
        toolbox=fake_toolbox(),
        enable_interrupt=True,
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "review-thread-1"}}

    paused = graph.invoke({"message": {"body": "We have a confidential data breach."}}, config)

    assert paused["__interrupt__"]
    assert paused["__interrupt__"][0].value["categories"] == ["confidentiality"]

    resumed = graph.invoke(
        Command(resume={"kind": "edit", "edited_text": "A specialist will contact you."}),
        config,
    )

    assert resumed["outcome"]["status"] == "answered"
    assert resumed["delivery"]["message"] == "A specialist will contact you."
    assert resumed["review_decision"]["kind"] == "edit"


def test_sqlite_checkpoints_survive_a_restarted_process(tmp_path: Path) -> None:
    path = tmp_path / "checkpoints.sqlite3"
    config = {"configurable": {"thread_id": "durable-thread-1"}}

    with sqlite_checkpointer(path) as saver:
        graph = build_graph(
            rag_answerer=fake_rag_answerer(),
            toolbox=fake_toolbox(),
            enable_interrupt=True,
            checkpointer=saver,
        )
        paused = graph.invoke({"message": {"body": "I want to escalate a complaint."}}, config)
        assert paused["__interrupt__"]

    with sqlite_checkpointer(path) as saver:
        graph = build_graph(
            rag_answerer=fake_rag_answerer(),
            toolbox=fake_toolbox(),
            enable_interrupt=True,
            checkpointer=saver,
        )
        resumed = graph.invoke(Command(resume={"kind": "approve"}), config)

    assert resumed["outcome"]["status"] == "pending_review"
    assert resumed["review_decision"]["kind"] == "approve"


def test_requesting_a_revision_never_approves_the_draft_it_rejected() -> None:
    graph = build_graph(
        rag_answerer=fake_rag_answerer(),
        toolbox=fake_toolbox(),
        enable_interrupt=True,
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "revision-thread-1"}}
    graph.invoke({"message": {"body": "I want to escalate a complaint."}}, config)

    resumed = graph.invoke(Command(resume={"kind": "request_revision"}), config)

    assert resumed["outcome"]["status"] == "pending_review"
    assert resumed["review_decision"]["kind"] == "request_revision"
