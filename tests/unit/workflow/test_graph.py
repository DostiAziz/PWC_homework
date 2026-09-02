from uuid import uuid4

from pwc_support.workflow.graph import build_graph


def test_main_graph_has_eight_meaningful_nodes() -> None:
    graph = build_graph().get_graph()

    assert {
        "intake", "triage", "plan_work", "case_tools", "compose_reply",
        "verify_response", "human_review", "finalise_case",
    } <= set(graph.nodes)


def test_general_question_reaches_finalisation() -> None:
    result = build_graph().invoke(
        {
            "message": {"body": "What services does PwC provide?"},
            "run_id": str(uuid4()),
        }
    )

    assert result["outcome"]["status"] == "answered"
    assert result["delivery"]["message"]
