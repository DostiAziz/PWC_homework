"""One-chat Streamlit UI for the local retail support prototype."""

from __future__ import annotations

import uuid
from typing import Any

import streamlit as st

from bootstrap import Runtime, build_runtime
from domain.models import ChatReply, Citation

st.set_page_config(page_title="Retail support", page_icon="💬")


@st.cache_resource
def get_runtime() -> Runtime:
    return build_runtime()


def render_citations(citations: tuple[Citation, ...]) -> None:
    if not citations:
        return
    with st.expander(f"Sources ({len(citations)})"):
        for citation in citations:
            st.markdown(f"- {citation.marker} **{citation.title}**, {citation.heading}")
            st.caption(citation.excerpt)


def render_trace(reply: ChatReply) -> None:
    label = f"Trace: {len(reply.steps)} steps, {reply.total_duration_ms:.0f} ms"
    with st.expander(label):
        if reply.steps:
            st.caption("Tools used: " + ", ".join(reply.steps))
        if reply.awaiting_confirmation:
            st.info(f"⏸ Awaiting confirmation: {reply.preview}")


def initialise_session() -> None:
    defaults: dict[str, Any] = {
        "messages": [],
        "customer_id": "CUS-1001",
        "thread_id": str(uuid.uuid4()),
        "awaiting_confirmation": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


st.title("Retail customer support")
st.caption("Local agentic RAG prototype using Ollama, LangGraph, Chroma, and SQLite")
st.markdown(
    "Try: `What jackets are on offer?`, `Where is ORD-5001?`, or `How long is standard shipping?`"
)
initialise_session()

try:
    runtime = get_runtime()
except Exception as error:
    st.error(
        "The local runtime is unavailable. Start Ollama, seed retail data, and ingest the corpus."
    )
    st.code(str(error))
    st.stop()

TOOL_LABELS: dict[str, str] = {
    "search_knowledge_base": "Searching knowledge base...",
    "search_products": "Searching product catalog...",
    "list_offers": "Checking active offers...",
    "get_order_status": "Checking order status...",
    "cancel_order": "Processing order cancellation...",
}


def get_step_label(step: str) -> str:
    return TOOL_LABELS.get(step, f"Running {step}...")


st.sidebar.caption(f"Demo customer: {st.session_state.customer_id}")
st.sidebar.caption(f"Generation: {runtime.settings.generation_model}")
st.sidebar.caption(f"Indexed chunks: {runtime.knowledge_base.count()}")

for entry in st.session_state.messages:
    with st.chat_message(entry["role"]):
        st.write(entry["content"])
        if entry.get("reply") is not None:
            render_citations(entry["reply"].citations)
            render_trace(entry["reply"])

if question := st.chat_input("Ask about products, offers, policies, or your order"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        status_box = st.status("Thinking...", expanded=True)
        text_placeholder = st.empty()
        streamed_text = ""
        last_reply: ChatReply | None = None

        if st.session_state.awaiting_confirmation:
            stream = runtime.service.stream_resume(
                thread_id=st.session_state.thread_id,
                customer_id=st.session_state.customer_id,
                decision=question,
            )
            # runtime.service.resume route
        else:
            stream = runtime.service.stream_submit(
                thread_id=st.session_state.thread_id,
                body=question,
                customer_id=st.session_state.customer_id,
            )

        for event in stream:
            if event.kind == "step":
                status_box.write(get_step_label(event.step))
            elif event.kind == "token":
                status_box.update(label="Generating response...", state="running", expanded=False)
                streamed_text += event.token
                text_placeholder.markdown(streamed_text)
            elif event.kind in ("done", "interrupt", "error"):
                last_reply = event.reply

        status_box.update(label="Complete", state="complete", expanded=False)
        reply = last_reply or ChatReply(
            message=streamed_text or "No response received.",
            status="unavailable",
        )
        if reply.message and reply.message != streamed_text:
            text_placeholder.markdown(reply.message)
        render_citations(reply.citations)
        render_trace(reply)

    st.session_state.awaiting_confirmation = reply.awaiting_confirmation
    st.session_state.messages.append(
        {"role": "assistant", "content": reply.message, "reply": reply}
    )
    st.rerun()

