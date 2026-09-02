from __future__ import annotations

from typing import Any

import streamlit as st

from pwc_support.bootstrap import build_runtime
from pwc_support.workflow.graph import build_graph

st.set_page_config(page_title="PwC Client Support Prototype", page_icon="💬")
st.title("PwC client support")
st.caption("Prototype: public-information assistance with controlled specialist review")


@st.cache_resource
def get_runtime() -> Any:
    return build_runtime()


def process_message(body: str) -> dict[str, Any]:
    try:
        result = get_runtime().invoke({"message": {"body": body}})
        st.session_state.runtime_error = None
        return result
    except Exception as error:
        st.session_state.runtime_error = str(error)
        return build_graph().invoke({"message": {"body": body}})

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_reviews" not in st.session_state:
    st.session_state.pending_reviews = []
if "runtime_error" not in st.session_state:
    st.session_state.runtime_error = None

client_tab, email_tab, review_tab = st.tabs(["Client chat", "Simulated email", "Human review"])

with client_tab:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message.get("status"):
                st.caption(f"Status: {message['status']}")
            if message.get("citations"):
                with st.expander("Sources"):
                    for citation in message["citations"]:
                        label = f"{citation['marker']} {citation['title']}"
                        if citation.get("canonical_url"):
                            st.markdown(f"- [{label}]({citation['canonical_url']})")
                        else:
                            st.markdown(f"- {label}")

    question = st.chat_input("Ask a general question about PwC services")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        result = process_message(question)
        outcome = result.get("outcome", {})
        status = outcome.get("status", "unknown")
        response = result.get("delivery", {}).get("message", "No response was produced.")
        rag = result.get("rag_results", [{}])[0]
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response,
                "status": status,
                "citations": rag.get("citations", []),
            }
        )
        if status == "pending_review":
            st.session_state.pending_reviews.append({"question": question, "response": response})
        st.rerun()

if st.session_state.runtime_error:
    st.sidebar.warning(
        "Local Ollama/Chroma runtime unavailable. Showing deterministic fallback. "
        "Run the runtime check and corpus ingestion scripts."
    )

with email_tab:
    st.subheader("Simulated inbox")
    email_body = st.text_area("Email body", key="email_body")
    if st.button("Process simulated email") and email_body.strip():
        result = process_message(email_body)
        st.success(result.get("delivery", {}).get("message", "No response was produced."))
        st.caption(f"Status: {result.get('outcome', {}).get('status', 'unknown')}")
        for citation in result.get("rag_results", [{}])[0].get("citations", []):
            st.markdown(f"- [{citation['title']}]({citation.get('canonical_url', '')})")

with review_tab:
    st.subheader("Pending specialist reviews")
    if not st.session_state.pending_reviews:
        st.info("No pending reviews in this demo session.")
    for index, review in enumerate(st.session_state.pending_reviews):
        with st.container(border=True):
            st.write(review["question"])
            st.caption(review["response"])
            edited = st.text_area(
                "Specialist response",
                value="A specialist will contact you.",
                key=f"review-{index}",
            )
            if st.button("Approve response", key=f"approve-{index}"):
                st.session_state.messages.append(
                    {"role": "assistant", "content": edited, "status": "answered"}
                )
                st.session_state.pending_reviews.pop(index)
                st.rerun()
