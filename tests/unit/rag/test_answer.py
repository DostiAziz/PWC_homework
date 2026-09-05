from domain.models import RagRequest, RetrievalBatch, RetrievalHit
from rag import RagAnswerer, prepare_query


class FakeKnowledgeBase:
    def __init__(self, hits: tuple[RetrievalHit, ...] | None = None) -> None:
        self.hits = (
            hits
            if hits is not None
            else (
                RetrievalHit(
                    source_id="shipping-and-orders",
                    chunk_id="shipping-and-orders-delivery-times-0",
                    title="Shipping and order support",
                    heading="Delivery times",
                    text="Standard delivery normally takes three to five business days.",
                    similarity=0.9,
                ),
            )
        )

    def retrieve(self, request: RagRequest, *, top_k: int = 6) -> RetrievalBatch:
        return RetrievalBatch(hits=self.hits)


class FakeGenerator:
    def __init__(
        self, reply: str = "Standard delivery takes three to five business days. [S1]"
    ) -> None:
        self.calls: list[str] = []
        self.reply = reply

    def text(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        self.calls.append(user)
        return self.reply


def test_prepare_query_normalizes_whitespace_without_adding_domain_claims() -> None:
    assert prepare_query("  How   long is shipping? ") == "How long is shipping?"


def test_rag_answer_uses_selected_retail_evidence() -> None:
    hit = RetrievalHit(
        source_id="shipping-and-orders",
        chunk_id="shipping-and-orders-delivery-times-0",
        title="Shipping and order support",
        heading="Delivery times",
        text="Standard delivery normally takes three to five business days.",
        similarity=0.9,
    )
    answerer = RagAnswerer(
        FakeKnowledgeBase(hits=(hit,)),
        FakeGenerator("Standard delivery takes three to five business days. [S1]"),
    )

    result = answerer.answer(RagRequest(question="How long is shipping?"))

    assert result.status == "answered"
    assert result.citations[0].source_id == "shipping-and-orders"


def test_rag_answer_without_a_valid_marker_is_withheld() -> None:
    hit = RetrievalHit(
        source_id="shipping-and-orders",
        chunk_id="shipping-1",
        title="Shipping and order support",
        heading="Delivery times",
        text="Standard delivery normally takes three to five business days.",
        similarity=0.9,
    )
    answerer = RagAnswerer(FakeKnowledgeBase(hits=(hit,)), FakeGenerator("Three days."))

    result = answerer.answer(RagRequest(question="How long is shipping?"))

    assert result.status == "insufficient_evidence"
    assert result.citations == ()
