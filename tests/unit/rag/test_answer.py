from pwc_support.domain.models import RagRequest, RetrievalBatch, RetrievalHit
from pwc_support.rag.answer import RagAnswerer


class FakeKnowledgeBase:
    def retrieve(self, request: RagRequest, *, top_k: int = 6) -> RetrievalBatch:
        return RetrievalBatch(hits=(RetrievalHit(
            source_id="source-1", chunk_id="chunk-1", title="Services",
            text="PwC provides consulting services.", heading="Services",
            similarity=0.8,
        ),))


class FakeGenerator:
    def text(self, *, system: str, user: str, max_tokens: int = 512) -> str:
        return "PwC provides consulting services. [S1]"


def test_rag_answerer_returns_grounded_answer_and_citation() -> None:
    result = RagAnswerer(FakeKnowledgeBase(), FakeGenerator()).answer(
        RagRequest(question="What services are available?")
    )

    assert result.status == "answered"
    assert result.citations[0].marker == "[S1]"
    assert result.citations[0].source_id == "source-1"
