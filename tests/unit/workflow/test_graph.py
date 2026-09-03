from uuid import uuid4

from pwc_support.workflow.graph import (
    build_graph,
    normalize_markers,
    renumber_citations,
    split_questions,
)
from tests.fakes import (
    FakeGenerator,
    FakeKnowledgeBase,
    InMemoryCaseStore,
    InMemoryMailbox,
    fake_rag_answerer,
    fake_toolbox,
)


def test_main_graph_exposes_every_named_workflow_node() -> None:
    graph = build_graph().get_graph()

    assert {
        "intake", "triage", "plan_work", "execute_task", "case_tools", "compose_reply",
        "verify_response", "respond_directly", "human_review", "finalise_case",
    } <= set(graph.nodes)


def test_general_question_is_answered_with_citations_from_the_rag_subgraph() -> None:
    result = build_graph(rag_answerer=fake_rag_answerer(), toolbox=fake_toolbox()).invoke(
        {"message": {"body": "What services does PwC provide?"}, "run_id": str(uuid4())}
    )

    assert result["outcome"]["status"] == "answered"
    assert result["verification"]["route"] == "release"
    assert result["delivery"]["citations"][0]["source_id"] == "pwc-global-services"
    assert "[S1]" in result["delivery"]["message"]


def test_greeting_is_answered_deterministically_without_retrieval_or_review() -> None:
    knowledge_base = FakeKnowledgeBase()
    result = build_graph(
        rag_answerer=fake_rag_answerer(knowledge_base), toolbox=fake_toolbox()
    ).invoke({"message": {"body": "hi"}})

    assert result["outcome"]["status"] == "answered"
    assert knowledge_base.requests == []
    assert [event["node"] for event in result["events"]] == [
        "intake", "triage", "respond_directly", "finalise_case"
    ]


def test_unsupported_benign_question_abstains_instead_of_escalating() -> None:
    result = build_graph(rag_answerer=fake_rag_answerer(), toolbox=fake_toolbox()).invoke(
        {"message": {"body": "How do I bake a sourdough loaf at home?"}}
    )

    assert result["outcome"]["status"] == "unable_to_answer"
    assert result["verification"]["route"] == "revise"
    assert "review_request" not in result


def test_short_message_asks_for_clarification() -> None:
    result = build_graph(rag_answerer=fake_rag_answerer(), toolbox=fake_toolbox()).invoke(
        {"message": {"body": "insurance?"}}
    )

    assert result["outcome"]["status"] == "clarification_required"


def test_multi_part_enquiry_fans_out_into_independent_tasks() -> None:
    knowledge_base = FakeKnowledgeBase()
    result = build_graph(
        rag_answerer=fake_rag_answerer(knowledge_base), toolbox=fake_toolbox()
    ).invoke(
        {
            "message": {
                "body": (
                    "What services does PwC provide to banks? "
                    "Which tax capabilities does PwC describe?"
                )
            }
        }
    )

    assert result["expected_task_ids"] == ["knowledge-1", "knowledge-2"]
    assert set(result["task_results"]) == {"knowledge-1", "knowledge-2"}
    assert len(knowledge_base.requests) == 2
    assert [citation["marker"] for citation in result["delivery"]["citations"]] == ["[S1]", "[S2]"]


def test_known_case_reference_adds_a_case_lookup_task_and_reuses_the_case() -> None:
    store = InMemoryCaseStore()
    store.seed("CASE-ABCD1234", conversation_id=uuid4())
    result = build_graph(rag_answerer=fake_rag_answerer(), toolbox=fake_toolbox(store)).invoke(
        {"message": {"body": "About CASE-ABCD1234, what services does PwC provide?"}}
    )

    assert "case-1" in result["task_results"]
    assert result["case_id"] == "CASE-ABCD1234"
    assert store.records["CASE-ABCD1234"].status.value == "resolved"


def test_email_channel_reply_is_delivered_through_the_mailbox_tool() -> None:
    mailbox = InMemoryMailbox()
    result = build_graph(
        rag_answerer=fake_rag_answerer(), toolbox=fake_toolbox(mailbox=mailbox)
    ).invoke(
        {
            "message": {"body": "What services does PwC provide?", "subject": "Services"},
            "channel": "simulated_email",
            "thread_id": "thread-42",
            "client_id": "client@example.test",
        }
    )

    assert mailbox.sent[0]["thread_id"] == "thread-42"
    assert mailbox.sent[0]["body"] == result["delivery"]["message"]


def test_verification_rejects_a_draft_that_cites_an_unknown_source() -> None:
    result = build_graph(
        rag_answerer=fake_rag_answerer(generator=FakeGenerator("Invented claim. [S7]")),
        toolbox=fake_toolbox(),
    ).invoke({"message": {"body": "What services does PwC provide?"}})

    assert result["verification"]["error_code"] == "MODEL_OUTPUT_INVALID"
    assert result["outcome"]["status"] == "pending_review"


def test_graph_without_a_retrieval_runtime_never_fabricates_an_answer() -> None:
    result = build_graph().invoke({"message": {"body": "What services does PwC provide?"}})

    assert result["outcome"]["status"] == "unable_to_answer"
    assert result["task_results"]["knowledge-1"].error_code == "RETRIEVAL_UNAVAILABLE"


def test_split_questions_only_decomposes_substantive_parts() -> None:
    assert split_questions("One question only?", limit=4) == ["One question only?"]
    assert split_questions("What does PwC do? Which sectors are covered?", limit=4) == [
        "What does PwC do?",
        "Which sectors are covered?",
    ]
    assert split_questions("What does PwC do? Why?", limit=4) == ["What does PwC do? Why?"]
    many = "What is one? What is two? What is three? What is four?"
    assert len(split_questions(many, limit=2)) == 2


def test_renumbering_gives_merged_answers_one_marker_series() -> None:
    citation = {
        "source_id": "s", "chunk_id": "c", "marker": "[S1]", "title": "t",
        "heading": "h", "excerpt": "e", "similarity": 0.5, "canonical_url": None,
    }
    text, citations = renumber_citations([
        {"answer": "First. [S1]", "citations": [citation]},
        {"answer": "Second. [S1]", "citations": [citation]},
    ])

    assert "[S1]" in text and "[S2]" in text
    assert [item.marker for item in citations] == ["[S1]", "[S2]"]


def test_unattributed_draft_is_withheld_rather_than_sent_to_the_client() -> None:
    result = build_graph(
        rag_answerer=fake_rag_answerer(
            generator=FakeGenerator("PwC does many things, take my word for it.")
        ),
        toolbox=fake_toolbox(),
    ).invoke({"message": {"body": "What services does PwC provide?"}})

    assert result["verification"]["error_code"] == "MODEL_OUTPUT_INVALID"
    assert result["outcome"]["status"] == "unable_to_answer"


def test_lookalike_citation_brackets_are_normalised_not_rejected() -> None:
    """A local model that writes 【S1】 is still citing its source correctly."""
    result = build_graph(
        rag_answerer=fake_rag_answerer(
            generator=FakeGenerator("PwC provides assurance and tax work.\u3010S1\u3011")
        ),
        toolbox=fake_toolbox(),
    ).invoke({"message": {"body": "What services does PwC provide?"}})

    assert result["outcome"]["status"] == "answered"
    assert "[S1]" in result["delivery"]["message"]
    assert "\u3010" not in result["delivery"]["message"]


def test_normalize_markers_leaves_ordinary_text_alone() -> None:
    assert normalize_markers("Plain answer [S1] with brackets.") == (
        "Plain answer [S1] with brackets."
    )
    assert normalize_markers("Answer \uff3bS2\uff3d here.") == "Answer [S2] here."


def test_escalated_enquiry_is_tracked_as_a_case_even_without_planning() -> None:
    store = InMemoryCaseStore()
    result = build_graph(
        rag_answerer=fake_rag_answerer(), toolbox=fake_toolbox(store)
    ).invoke({"message": {"body": "We may have exposed a confidential client document."}})

    case_id = result["case_id"]
    assert case_id is not None
    assert result["review_request"]["case_id"] == case_id
    assert store.records[case_id].category == "confidentiality"
    assert store.records[case_id].status.value == "pending_review"
