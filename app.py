"""Streamlit entry point: client chat, simulated email, and specialist review."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import streamlit as st

from pwc_support.bootstrap import Runtime, build_runtime
from pwc_support.config import Settings
from pwc_support.domain.models import (
    Channel,
    ReviewDecisionKind,
    ReviewRequest,
)
from pwc_support.services.client_support import ClientSupportService, WorkflowRun
from pwc_support.storage.checkpoints import sqlite_checkpointer

st.set_page_config(page_title="PwC Client Support Prototype", page_icon="💬", layout="wide")


@dataclass(frozen=True, slots=True)
class Application:
    runtime: Runtime
    service: ClientSupportService
    # Holding the stack keeps the SQLite checkpoint connection open for the app's life.
    # Without this reference it is garbage collected and every resume fails.
    stack: ExitStack


@st.cache_resource
def get_application() -> Application:
    """Open the local runtime once, keeping the SQLite checkpoint store alive."""
    stack = ExitStack()
    settings = Settings.from_env()
    saver = stack.enter_context(sqlite_checkpointer(settings.checkpoints_db))
    runtime = build_runtime(settings, checkpointer=saver, enable_interrupt=True)
    service = ClientSupportService(
        runtime.graph,
        reviews=runtime.reviews,
        database=runtime.database,
        mailbox=runtime.mailbox,
        review_service=runtime.review_service,
    )
    return Application(runtime=runtime, service=service, stack=stack)


def render_run_details(run: WorkflowRun) -> None:
    """Expose the agent's actual execution: nodes, routing, timing, and evidence."""
    with st.expander(
        f"Agent trace — {len(run.events)} nodes, {run.total_duration_ms:.0f} ms", expanded=False
    ):
        triage = run.state.get("triage", {})
        verification = run.state.get("verification", {})
        columns = st.columns(3)
        columns[0].metric("Triage route", str(triage.get("route", "—")))
        columns[1].metric("Verification", str(verification.get("route", "—")))
        columns[2].metric("Total latency", f"{run.total_duration_ms:.0f} ms")
        if triage.get("review_categories"):
            st.caption(f"Review categories: {', '.join(triage['review_categories'])}")
        if verification.get("reason"):
            st.caption(f"Verification note: {verification['reason']}")

        st.markdown("**Node execution**")
        st.dataframe(
            [
                {
                    "node": event["node"],
                    "event": event["event_type"],
                    "ms": event.get("duration_ms"),
                    "details": ", ".join(
                        f"{key}={value}" for key, value in event.get("details", {}).items()
                    ),
                }
                for event in run.events
            ],
            hide_index=True,
            width="stretch",
        )

        plan = run.state.get("plan", {}).get("tasks", [])
        if plan:
            st.markdown("**Decomposed tasks**")
            st.dataframe(
                [
                    {
                        "task_id": task["task_id"],
                        "kind": task["kind"],
                        "input": task["input"],
                        "status": _task_status(run, task["task_id"]),
                    }
                    for task in plan
                ],
                hide_index=True,
                width="stretch",
            )

        if run.tool_calls:
            st.markdown("**Tool calls**")
            st.dataframe(
                [
                    {
                        "tool": call["tool"],
                        "operation": call["operation"],
                        "ok": call["ok"],
                        "detail": ", ".join(
                            f"{key}={value}" for key, value in call["detail"].items()
                        ),
                    }
                    for call in run.tool_calls
                ],
                hide_index=True,
                width="stretch",
            )

        excerpts = [
            (result["task_id"], hit)
            for result in run.rag_results
            for hit in result.get("hits", [])
        ]
        if excerpts:
            st.markdown("**Retrieved evidence**")
            for task_id, hit in excerpts:
                st.caption(
                    f"{task_id} · {hit['title']} — {hit['heading']} "
                    f"(similarity {hit['similarity']:.2f})"
                )
                st.text(hit["text"][:400])


def _task_status(run: WorkflowRun, task_id: str) -> str:
    result = run.state.get("task_results", {}).get(task_id)
    if result is None:
        return "—"
    return f"{result.status.value}{f' ({result.error_code})' if result.error_code else ''}"


def render_citations(citations: list[dict[str, Any]]) -> None:
    if not citations:
        return
    with st.expander(f"Sources ({len(citations)})"):
        for citation in citations:
            label = f"{citation['marker']} {citation['title']} — {citation['heading']}"
            if citation.get("canonical_url"):
                st.markdown(f"- [{label}]({citation['canonical_url']})")
            else:
                st.markdown(f"- {label} *(synthetic source)*")
            st.caption(citation["excerpt"][:300])


def render_client_chat(service: ClientSupportService) -> None:
    st.caption(
        "Ask a general question about publicly described PwC services. Answers are grounded "
        "in the local corpus and cited; sensitive matters are routed to a specialist."
    )
    for entry in st.session_state.messages:
        with st.chat_message(entry["role"]):
            st.write(entry["content"])
            if entry.get("status"):
                st.caption(f"Status: {entry['status']}")
            render_citations(entry.get("citations", []))
            if entry.get("run") is not None:
                render_run_details(entry["run"])

    question = st.chat_input("Ask a general question about PwC services")
    if not question:
        return
    st.session_state.messages.append({"role": "user", "content": question})
    with st.spinner("Running the support workflow locally…"):
        run = service.submit(
            body=question,
            client_id=st.session_state.client_id,
            conversation_id=st.session_state.conversation_id,
        )
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": run.outcome.message,
            "status": run.outcome.status.value,
            "citations": [citation.model_dump(mode="json") for citation in run.outcome.citations],
            "run": run,
        }
    )
    st.rerun()


def render_email(service: ClientSupportService, runtime: Runtime) -> None:
    st.subheader("Simulated inbox")
    st.caption(
        "This channel writes through the file-backed `SimulatedMailbox`. It does not connect "
        "to Gmail, Outlook, or any real mail system."
    )
    sender = st.text_input("From", value="client@example.test", key="email_sender")
    subject = st.text_input("Subject", value="Question about PwC services", key="email_subject")
    body = st.text_area("Email body", key="email_body")
    if st.button("Deliver to support", type="primary") and body.strip():
        received = runtime.mailbox.receive(sender=sender, subject=subject, body=body)
        with st.spinner("Running the support workflow locally…"):
            run = service.submit(
                body=body,
                client_id=sender,
                channel=Channel.SIMULATED_EMAIL,
                conversation_id=received.conversation_id,
                thread_id=received.provider_thread_id,
                subject=subject,
            )
        st.success(f"Status: {run.outcome.status.value}")
        st.write(run.outcome.message)
        render_citations([citation.model_dump(mode="json") for citation in run.outcome.citations])
        render_run_details(run)

    outbox = runtime.mailbox.outbox()
    st.markdown(f"**Outbox ({len(outbox)})**")
    for message in reversed(outbox[-10:]):
        with st.container(border=True):
            st.caption(
                f"thread {message.provider_thread_id} → {message.recipient_id} · "
                f"{message.received_at:%Y-%m-%d %H:%M:%S}"
            )
            st.write(message.body)


def render_review(service: ClientSupportService) -> None:
    st.subheader("Pending specialist reviews")
    st.caption("Reviews are durable SQLite cases. Decisions are applied asynchronously and delivered through the simulated mailbox.")
    pending = service.pending_reviews()
    if not pending:
        st.info("No pending reviews. Sensitive enquiries appear here as soon as they pause.")
        return
    for review in pending:
        _render_review_card(service, review)


def _render_review_card(service: ClientSupportService, review: ReviewRequest) -> None:
    key = str(review.review_id)
    with st.container(border=True):
        st.markdown(f"**Case {review.case_id}** · review `{key[:8]}`")
        st.caption(f"Categories: {', '.join(sorted(review.categories)) or 'none'}")
        st.write(review.original_message)
        if review.evidence:
            with st.expander(f"Background sources ({len(review.evidence)})", expanded=True):
                st.caption(
                    "Retrieved for this enquiry. No reply was drafted — a sensitive "
                    "enquiry is never answered by the model."
                )
                for citation in review.evidence:
                    label = f"{citation.title} — {citation.heading}"
                    if citation.canonical_url:
                        st.markdown(f"**[{label}]({citation.canonical_url})**")
                    else:
                        st.markdown(f"**{label}** *(synthetic source)*")
                    st.caption(f"similarity {citation.similarity:.2f}")
                    st.text(citation.excerpt[:400])
        proposed = ""
        kind = st.selectbox(
            "Decision",
            [
                ReviewDecisionKind.SEND_RESPONSE.value,
                ReviewDecisionKind.TAKE_OWNERSHIP.value,
                ReviewDecisionKind.REJECT.value,
            ],
            key=f"kind-{key}",
        )
        edited = st.text_area(
            "Reply to the client",
            value=proposed,
            key=f"text-{key}",
        )
        reviewer = st.text_input("Reviewer", value="specialist-1", key=f"reviewer-{key}")
        if st.button("Submit decision", key=f"submit-{key}", type="primary"):
            result = service.decide_review(
                review_id=review.review_id,
                decision_id=uuid4(),
                expected_version=review.response_version,
                reviewer_id=reviewer,
                kind=ReviewDecisionKind(kind),
                reviewed_text=edited or None,
                reason="reviewer decision",
            )
            st.success(f"Decision recorded: {result.case_status.value}")
            st.rerun()


def initialise_session() -> None:
    defaults: dict[str, Any] = {
        "messages": [],
        "conversation_id": uuid4(),
        "client_id": "demo-client",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


st.title("PwC client support")
st.caption("Prototype: public-information assistance with controlled specialist review")
initialise_session()

try:
    application = get_application()
except Exception as error:  # surfaced to the operator, never hidden behind a fake answer
    st.error(
        "The local runtime is unavailable, so this prototype cannot answer anything. "
        "It will not fall back to an ungrounded response."
    )
    st.code(str(error))
    st.markdown(
        "Start Ollama, pull `gpt-oss:20b` and `nomic-embed-text`, then run:\n\n"
        "```bash\n"
        "PYTHONPATH=src .venv/bin/python scripts/check_runtime.py\n"
        "PYTHONPATH=src .venv/bin/python scripts/ingest_corpus.py\n"
        "```"
    )
    st.stop()

service = application.service
settings = application.runtime.settings
st.sidebar.subheader("Local runtime")
st.sidebar.write(f"Generation: `{settings.generation_model}`")
st.sidebar.write(f"Embeddings: `{settings.embedding_model}`")
st.sidebar.write(f"Chunks indexed: {application.runtime.knowledge_base.count()}")
st.sidebar.write(f"Conversation: `{str(st.session_state.conversation_id)[:8]}`")

client_tab, email_tab, review_tab = st.tabs(["Client chat", "Simulated email", "Human review"])
with client_tab:
    render_client_chat(service)
with email_tab:
    render_email(service, application.runtime)
with review_tab:
    render_review(service)
