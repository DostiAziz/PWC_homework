from pathlib import Path


def test_client_surface_uses_one_chat_input() -> None:
    source = Path("app.py").read_text(encoding="utf-8")
    assert "st.chat_input" in source
    assert "render_retail_workspace" not in source
    assert "Order ID" not in source
    assert "Item ID" not in source
    assert "Reason" not in source


def test_client_examples_are_natural_language() -> None:
    source = Path("app.py").read_text(encoding="utf-8")
    assert "Can you tell me if my order has shipped?" in source
    assert "What products are on offer?" in source
