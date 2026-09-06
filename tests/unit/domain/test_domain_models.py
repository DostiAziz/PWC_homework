from decimal import Decimal

from domain import (
    CancellationPreview,
    CancellationResult,
    ChatReply,
    Citation,
    DomainModel,
    OfferSummary,
    OrderSummary,
    PendingCancellation,
    ProductSummary,
    RagRequest,
    RagResult,
    RetrievalBatch,
    RetrievalHit,
    TimingSpan,
)
from domain.chat import ChatReply as ChatReplyDirect
from domain.chat import TimingSpan as TimingSpanDirect
from domain.models import OrderSummary as OrderSummaryCompat
from domain.rag import Citation as CitationDirect
from domain.rag import RagRequest as RagRequestDirect
from domain.rag import RagResult as RagResultDirect
from domain.rag import RetrievalBatch as RetrievalBatchDirect
from domain.rag import RetrievalHit as RetrievalHitDirect
from domain.retail import CancellationPreview as CancellationPreviewDirect
from domain.retail import CancellationResult as CancellationResultDirect
from domain.retail import OfferSummary as OfferSummaryDirect
from domain.retail import OrderSummary as OrderSummaryDirect
from domain.retail import PendingCancellation as PendingCancellationDirect
from domain.retail import ProductSummary as ProductSummaryDirect


def test_domain_reexports():
    assert OrderSummary is OrderSummaryDirect is OrderSummaryCompat
    assert ProductSummary is ProductSummaryDirect
    assert OfferSummary is OfferSummaryDirect
    assert CancellationPreview is CancellationPreviewDirect
    assert PendingCancellation is PendingCancellationDirect
    assert CancellationResult is CancellationResultDirect
    assert Citation is CitationDirect
    assert RetrievalHit is RetrievalHitDirect
    assert RetrievalBatch is RetrievalBatchDirect
    assert RagRequest is RagRequestDirect
    assert RagResult is RagResultDirect
    assert TimingSpan is TimingSpanDirect
    assert ChatReply is ChatReplyDirect
    assert issubclass(ProductSummary, DomainModel)


def test_model_instantiation():
    product = ProductSummary(
        product_id="PRD-1",
        name="T-Shirt",
        category="Apparel",
        price=Decimal("19.99"),
        currency="EUR",
        stock=10,
    )
    assert product.price == Decimal("19.99")

    order = OrderSummary(
        order_id="ORD-1",
        customer_id="CUS-1",
        status="paid",
        fulfilment_status="processing",
        total=Decimal("19.99"),
        currency="EUR",
        version=1,
    )
    assert order.order_id == "ORD-1"
