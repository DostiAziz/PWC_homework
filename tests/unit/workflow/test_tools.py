from pwc_support.rag.answer import RagAnswerer
from pwc_support.storage.database import Database
from pwc_support.storage.retail_repositories import OrderRepository, ProductRepository
from pwc_support.workflow.tools import TOOL_SCHEMAS, ToolRegistry
from tests.fakes import FakeGenerator, FakeKnowledgeBase


def _registry(retail_db: Database) -> ToolRegistry:
    rag = RagAnswerer(FakeKnowledgeBase(), FakeGenerator())
    return ToolRegistry(ProductRepository(retail_db), OrderRepository(retail_db), rag)


def test_schemas_expose_five_tools() -> None:
    names = {s["function"]["name"] for s in TOOL_SCHEMAS}
    assert names == {
        "search_products",
        "list_offers",
        "get_order_status",
        "search_policies",
        "cancel_order",
    }


def test_list_offers_returns_discounted_price(retail_db: Database) -> None:
    out = _registry(retail_db).run("list_offers", {"category": "jackets"}, customer_id="CUS-1001")
    assert "116.99" in out.content
    assert out.requires_confirmation is False


def test_get_order_status_is_customer_scoped(retail_db: Database) -> None:
    reg = _registry(retail_db)
    mine = reg.run("get_order_status", {"order_id": "ORD-5001"}, customer_id="CUS-1001")
    assert "shipped" in mine.content.lower()
    other = reg.run("get_order_status", {"order_id": "ORD-3001"}, customer_id="CUS-1001")
    assert "could not find" in other.content.lower()


def test_search_policies_returns_citations(retail_db: Database) -> None:
    out = _registry(retail_db).run(
        "search_policies", {"question": "how long is shipping?"}, customer_id="CUS-1001"
    )
    assert "[S1]" in out.content
    assert len(out.citations) >= 1


def test_cancel_order_defers_to_confirmation(retail_db: Database) -> None:
    out = _registry(retail_db).run(
        "cancel_order", {"order_id": "ORD-2001"}, customer_id="CUS-1001"
    )
    assert out.requires_confirmation is True
    assert out.order_id == "ORD-2001"


def test_unknown_tool_returns_safe_message(retail_db: Database) -> None:
    out = _registry(retail_db).run("nope", {}, customer_id="CUS-1001")
    assert "unknown" in out.content.lower()
