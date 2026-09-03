from pwc_support.domain.models import RagRequest, RetrievalBatch, RetrievalHit
from pwc_support.rag.subgraph import build_rag_graph, to_rag_result
from tests.fakes import FakeGenerator, FakeKnowledgeBase


def test_rag_subgraph_compiles_the_four_required_nodes() -> None:
    graph = build_rag_graph(FakeKnowledgeBase(), FakeGenerator()).get_graph()

    assert {
        "prepare_query",
        "retrieve_candidates",
        "select_evidence",
        "answer_with_citations",
    } <= set(graph.nodes)


def test_subgraph_answers_from_evidence_and_reports_per_node_timing() -> None:
    graph = build_rag_graph(FakeKnowledgeBase(), FakeGenerator())

    state = graph.invoke({"request": RagRequest(question="What services are available?")})

    assert state["status"] == "answered"
    assert state["citations"][0].marker == "[S1]"
    assert set(state["node_timings"]) == {
        "prepare_query",
        "retrieve_candidates",
        "select_evidence",
        "answer_with_citations",
    }


def test_subgraph_skips_generation_when_no_evidence_clears_the_threshold() -> None:
    generator = FakeGenerator()
    graph = build_rag_graph(FakeKnowledgeBase(), generator)

    state = graph.invoke({"request": RagRequest(question="How do I bake bread at home?")})

    assert state["status"] == "insufficient_evidence"
    assert generator.calls == []
    assert to_rag_result(state).status == "insufficient_evidence"


def test_weak_similarity_is_discarded_before_generation() -> None:
    weak = RetrievalHit(
        source_id="s", chunk_id="c", title="t", text="weak", heading="h", similarity=0.2
    )
    generator = FakeGenerator()
    graph = build_rag_graph(FakeKnowledgeBase(hits=(weak,)), generator, minimum_similarity=0.45)

    state = graph.invoke({"request": RagRequest(question="What services does PwC provide?")})

    assert state["status"] == "insufficient_evidence"
    assert generator.calls == []


def test_selection_respects_the_maximum_hit_budget() -> None:
    hits = tuple(
        RetrievalHit(
            source_id=f"s{index}",
            chunk_id=f"c{index}",
            title="t",
            text="evidence",
            heading="h",
            similarity=0.9,
        )
        for index in range(6)
    )

    class ManyHits(FakeKnowledgeBase):
        def retrieve(self, request: RagRequest) -> RetrievalBatch:
            return RetrievalBatch(hits=hits)

    graph = build_rag_graph(ManyHits(), FakeGenerator(), max_selected_hits=2)
    state = graph.invoke({"request": RagRequest(question="What services does PwC provide?")})

    assert len(state["selected"]) == 2
