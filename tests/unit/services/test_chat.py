from typing import Any

from pwc_support.domain.models import Task, TaskKind, TraceEvent
from pwc_support.services.chat import ChatService


class FakeGraph:
    def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
        return {
            **state,
            "tasks": (Task(task_id="task-1", kind=TaskKind.KNOWLEDGE, request=state["message"]),),
            "response": "Grounded answer. [S1]",
            "citations": (),
            "events": (TraceEvent(node="respond", event_type="completed", duration_ms=1.0),),
        }


def test_submit_projects_terminal_graph_state_to_chat_reply() -> None:
    reply = ChatService(FakeGraph()).submit(
        body="How long is shipping?",
        customer_id="CUS-1001",
    )

    assert reply.message == "Grounded answer. [S1]"
    assert [task.kind for task in reply.tasks] == [TaskKind.KNOWLEDGE]
    assert reply.total_duration_ms >= 0


def test_submit_exposes_failure_without_inventing_an_answer() -> None:
    class FailingGraph:
        def invoke(self, _: dict[str, Any]) -> dict[str, Any]:
            raise RuntimeError("offline")

    reply = ChatService(FailingGraph()).submit(body="question", customer_id="CUS-1001")

    assert reply.message == "The support workflow is unavailable. Please try again."
    assert reply.citations == ()
    assert reply.pending_cancellation is None
