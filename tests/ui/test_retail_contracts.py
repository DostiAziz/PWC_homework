from pathlib import Path


def test_ui_is_one_natural_language_chat() -> None:
    source = Path("app.py").read_text(encoding="utf-8")

    assert source.count("st.chat_input") == 1
    assert "st.tabs" not in source
    assert "Simulated email" not in source
    assert "Human review" not in source


def test_ui_routes_confirmation_via_resume() -> None:
    source = Path("app.py").read_text(encoding="utf-8")

    assert '"awaiting_confirmation"' in source
    assert "reply.awaiting_confirmation" in source
    assert "runtime.service.resume" in source
    assert "thread_id" in source
