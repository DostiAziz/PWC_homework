from __future__ import annotations

import streamlit as st

from pwc_support.workflow.graph import build_graph

st.set_page_config(page_title="PwC Client Support Prototype", page_icon="💬")
st.title("PwC client support")
st.caption("Prototype: public-information assistance with controlled specialist review")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message.get("status"):
            st.caption(f"Status: {message['status']}")

question = st.chat_input("Ask a general question about PwC services")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    result = build_graph().invoke({"message": {"body": question}})
    outcome = result.get("outcome", {})
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result.get("delivery", {}).get("message", "No response was produced."),
            "status": outcome.get("status", "unknown"),
        }
    )
    st.rerun()
