from pwc_support.domain.models import RagRequest, RetrievalBatch, RetrievalHit
from pwc_support.rag.answer import RagAnswerer, prepare_query


class FakeKnowledgeBase:
    def retrieve(self, request: RagRequest, *, top_k: int = 6) -> RetrievalBatch:
        return RetrievalBatch(
            hits=(
                RetrievalHit(
                    source_id="source-1",
                    chunk_id="chunk-1",
                    title="Services",
                    text="PwC provides consulting services.",
                    heading="Services",
                    similarity=0.8,
                ),
            )
        )


class FakeGenerator:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def text(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> str:
        self.calls.append(user)
        return "PwC provides consulting services. [S1]"


def test_rag_answerer_returns_grounded_answer_and_citation() -> None:
    result = RagAnswerer(FakeKnowledgeBase(), FakeGenerator()).answer(
        RagRequest(question="What services are available?")
    )

    assert result.status == "answered"
    assert result.citations[0].marker == "[S1]"
    assert result.citations[0].source_id == "source-1"


def test_query_preparation_resolves_customer_support_pronouns() -> None:
    assert prepare_query("What services do you provide?") == (
        "What services do you provide? PwC business services"
    )
    assert prepare_query("What services does PwC provide?") == "What services does PwC provide?"


def test_gather_evidence_selects_sources_without_calling_the_model() -> None:
    generator = FakeGenerator()
    answerer = RagAnswerer(FakeKnowledgeBase(), generator)

    bundle = answerer.gather_evidence(RagRequest(question="Was our report exposed?"))

    assert [citation.source_id for citation in bundle.citations] == ["source-1"]
    assert bundle.citations[0].marker == "[S1]"
    assert generator.calls == []
