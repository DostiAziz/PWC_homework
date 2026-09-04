from pathlib import Path


def test_ui_is_one_natural_language_chat() -> None:
    source = Path("app.py").read_text(encoding="utf-8")

    assert source.count("st.chat_input") == 1
    assert "st.tabs" not in source
    assert "Simulated email" not in source
    assert "Human review" not in source
    assert "Order ID" not in source


def test_ui_keeps_pending_cancellation_in_session() -> None:
    source = Path("app.py").read_text(encoding="utf-8")

    assert '"pending_cancellation"' in source
    assert "reply.pending_cancellation" in source
