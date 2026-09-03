from pathlib import Path


def test_streamlit_exposes_retail_workspace_controls() -> None:
    source = Path("app.py").read_text(encoding="utf-8")
    assert "Retail self-service" in source
    assert "product-search-form" in source
    assert "recommendation-form" in source
    assert "order-status-form" in source
    assert "return-form" in source
