from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from pwc_support.workflow.graph import build_graph


def test_sensitive_case_pauses_and_resumes_after_review() -> None:
    graph = build_graph(enable_interrupt=True, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "review-thread-1"}}

    paused = graph.invoke(
        {"message": {"body": "We have a confidential data breach."}}, config
    )

    assert paused["__interrupt__"]
    resumed = graph.invoke(
        Command(resume={"decision": "approve", "reply": "A specialist will contact you."}),
        config,
    )

    assert resumed["outcome"]["status"] == "answered"
    assert resumed["delivery"]["message"] == "A specialist will contact you."
