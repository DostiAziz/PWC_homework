from pathlib import Path

from pwc_support.storage.database import Database
from pwc_support.storage.retail_repositories import OrderRepository, ProductRepository, ReturnRepository
from pwc_support.workflow.retail_tools import build_retail_action_tools, build_retail_tools
from tests.unit.storage.test_retail_repositories import _db


def test_retail_tools_expose_inventory_offer_and_return_schemas(tmp_path: Path) -> None:
    db = _db(tmp_path)
    tools = build_retail_tools(ProductRepository(db), OrderRepository(db), "C-1", ReturnRepository(db))
    names = {tool.name for tool in tools}
    assert {"search_products", "get_product", "check_inventory", "get_active_offer", "lookup_order", "evaluate_product_return"} <= names
    assert {tool.name for tool in build_retail_action_tools(ReturnRepository(db))} == {"request_product_return"}
    assert tools[2].invoke({"product_id": "P-1"})["quantity"] == 3
