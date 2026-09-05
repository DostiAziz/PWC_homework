from __future__ import annotations

from customer_support.domain.models import RagRequest, RetrievalBatch, RetrievalHit

RETAIL_HIT = RetrievalHit(
    source_id="shipping-and-orders",
    chunk_id="chunk-shipping-1",
    title="Shipping and Orders",
    text="Standard shipping takes three to five business days.",
    heading="Standard Delivery",
    similarity=0.82,
)


class FakeKnowledgeBase:
    """Return evidence only for questions the fake corpus actually covers."""

    def __init__(self, *, hits: tuple[RetrievalHit, ...] = (RETAIL_HIT,)) -> None:
        self.hits = hits
        self.requests: list[RagRequest] = []

    def retrieve(self, request: RagRequest) -> RetrievalBatch:
        self.requests.append(request)
        question = request.question.casefold()
        terms = ("service", "pwc", "advisory", "tax", "shipping", "order", "delivery")
        if any(term in question for term in terms):
            return RetrievalBatch(hits=self.hits)
        return RetrievalBatch(hits=())


class FakeGenerator:
    def __init__(
        self, template: str = "Standard shipping takes three to five business days. [S1]"
    ) -> None:
        self.template = template
        self.calls: list[str] = []

    def text(
        self, *, system: str, user: str, max_tokens: int = 512, temperature: float = 0.2
    ) -> str:
        self.calls.append(user)
        return self.template
