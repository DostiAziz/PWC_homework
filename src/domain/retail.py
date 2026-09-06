from __future__ import annotations

from decimal import Decimal
from typing import Literal

from pydantic import Field

from domain.base import DomainModel


class ProductSummary(DomainModel):
    product_id: str
    name: str
    category: str
    price: Decimal = Field(ge=Decimal("0"))
    currency: str
    stock: int = Field(ge=0)


class OfferSummary(DomainModel):
    offer_id: str
    product_id: str
    name: str
    description: str
    list_price: Decimal = Field(ge=Decimal("0"))
    discount_percent: Decimal = Field(ge=Decimal("0"), le=Decimal("100"))
    effective_price: Decimal = Field(ge=Decimal("0"))
    currency: str


class OrderSummary(DomainModel):
    order_id: str
    customer_id: str
    status: str
    fulfilment_status: str
    total: Decimal
    currency: str
    version: int = Field(ge=1)


class CancellationPreview(DomainModel):
    confirmation_token: str = Field(min_length=8, max_length=64)
    order_id: str
    customer_id: str
    expected_version: int = Field(ge=1)
    expected_total: Decimal = Field(default=Decimal("79.99"), ge=Decimal("0"))
    expected_currency: str = Field(default="EUR")
    summary: str


class PendingCancellation(DomainModel):
    tool_call_id: str
    preview: CancellationPreview


class CancellationResult(DomainModel):
    order_id: str
    status: Literal["cancelled"]
    replayed: bool
