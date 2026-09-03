from __future__ import annotations

import time
from typing import Annotated, Any, Literal, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph

from pwc_support.domain.models import (
    Citation,
    EvidenceBundle,
    RagRequest,
    RagResult,
    RetrievalBatch,
    RetrievalHit,
)


def merge_timings(
    left: dict[str, float] | None, right: dict[str, float] | None
) -> dict[str, float]:
    """Accumulate per-node latency instead of letting the last node overwrite it."""
    return {**(left or {}), **(right or {})}


class KnowledgeBase(Protocol):
    def retrieve(self, request: RagRequest) -> RetrievalBatch: ...


class Generator(Protocol):
    def text(
        self, *, system: str, user: str, max_tokens: int = 512, temperature: float = 0.2
    ) -> str: ...


class RagState(TypedDict, total=False):
    """State of the retrieval subgraph, kept separate from the support workflow."""

    request: RagRequest
    prepared_question: str
    candidates: tuple[RetrievalHit, ...]
    selected: tuple[RetrievalHit, ...]
    citations: tuple[Citation, ...]
    used_filter_fallback: bool
    evidence_conflict: bool
    status: Literal["answered", "insufficient_evidence"]
    answer: str
    node_timings: Annotated[dict[str, float], merge_timings]


ANSWER_SYSTEM_PROMPT = (
    "You answer client questions for a professional-services support desk using only the "
    "supplied evidence. Never add facts that are absent from the evidence. Keep every "
    "citation marker such as [S1] immediately after the sentence it supports. If the "
    "evidence does not answer the question, say so plainly."
)


def prepare_query(question: str) -> str:
    """Resolve support-desk pronouns so retrieval sees the organisation, not the reader."""
    normalized = question.casefold()
    refers_to_assistant = " you" in f" {normalized}" or " your" in f" {normalized}"
    if refers_to_assistant and "pwc" not in normalized:
        return f"{question} PwC business services"
    return question


def build_rag_graph(
    knowledge_base: KnowledgeBase,
    generator: Generator,
    *,
    minimum_similarity: float = 0.45,
    max_selected_hits: int = 4,
    max_evidence_chars: int = 6000,
    answer_tokens: int = 512,
    generate: bool = True,
    checkpointer: Any = False,
) -> Any:
    """Compile the dedicated four-node retrieval-augmented-generation subgraph.

    The subgraph is stateless by default (`checkpointer=False`). It never interrupts, and
    the main graph fans several knowledge tasks onto the same instance in one superstep,
    which would otherwise collide in a single inherited checkpoint namespace.

    With `generate=False` the same retrieval and selection run, but the graph stops after
    `select_evidence`. The escalation path uses that mode to put sources in front of a
    specialist without ever asking the model to draft an answer to a sensitive enquiry.
    """

    def _timed(name: str, started: float) -> dict[str, float]:
        return {name: round((time.perf_counter() - started) * 1000, 2)}

    def prepare_query_node(state: RagState) -> dict[str, Any]:
        started = time.perf_counter()
        request = state["request"]
        return {
            "prepared_question": prepare_query(request.question),
            "node_timings": _timed("prepare_query", started),
        }

    def retrieve_candidates_node(state: RagState) -> dict[str, Any]:
        started = time.perf_counter()
        request = state["request"].model_copy(
            update={"question": state["prepared_question"]}
        )
        batch = knowledge_base.retrieve(request)
        return {
            "candidates": batch.hits,
            "used_filter_fallback": batch.used_filter_fallback,
            "evidence_conflict": batch.evidence_conflict,
            "node_timings": _timed("retrieve_candidates", started),
        }

    def select_evidence_node(state: RagState) -> dict[str, Any]:
        started = time.perf_counter()
        candidates = state.get("candidates", ())
        kept: list[RetrievalHit] = []
        budget = max_evidence_chars
        for hit in candidates:
            if hit.similarity < minimum_similarity:
                continue
            if len(kept) >= max_selected_hits or budget - len(hit.text) < 0:
                break
            budget -= len(hit.text)
            kept.append(hit)
        selected = tuple(kept)
        citations = tuple(
            Citation(
                source_id=hit.source_id,
                chunk_id=hit.chunk_id,
                marker=f"[S{index}]",
                title=hit.title,
                canonical_url=hit.canonical_url,
                heading=hit.heading,
                excerpt=hit.text[:500],
                similarity=hit.similarity,
            )
            for index, hit in enumerate(selected, start=1)
        )
        return {
            "selected": selected,
            "citations": citations,
            "status": "answered" if selected else "insufficient_evidence",
            "node_timings": _timed("select_evidence", started),
        }

    def route_after_selection(state: RagState) -> str:
        return "answer_with_citations" if state.get("selected") else END

    def answer_with_citations_node(state: RagState) -> dict[str, Any]:
        started = time.perf_counter()
        selected = state["selected"]
        citations = state["citations"]
        evidence = "\n\n".join(
            f"{citation.marker} {hit.title} — {hit.heading}: {hit.text}"
            for citation, hit in zip(citations, selected, strict=True)
        )
        answer = generator.text(
            system=ANSWER_SYSTEM_PROMPT,
            user=f"Question: {state['request'].question}\nEvidence:\n{evidence}",
            max_tokens=answer_tokens,
            temperature=0.0,
        )
        return {"answer": answer, "node_timings": _timed("answer_with_citations", started)}

    builder: StateGraph[RagState, None, RagState, RagState] = StateGraph(RagState)
    builder.add_node("prepare_query", prepare_query_node)
    builder.add_node("retrieve_candidates", retrieve_candidates_node)
    builder.add_node("select_evidence", select_evidence_node)
    builder.add_edge(START, "prepare_query")
    builder.add_edge("prepare_query", "retrieve_candidates")
    builder.add_edge("retrieve_candidates", "select_evidence")
    if generate:
        builder.add_node("answer_with_citations", answer_with_citations_node)
        builder.add_conditional_edges("select_evidence", route_after_selection)
        builder.add_edge("answer_with_citations", END)
    else:
        # Evidence-only: the generation node is not compiled in at all, so no code path
        # can reach the model from an escalated enquiry.
        builder.add_edge("select_evidence", END)
    return builder.compile(checkpointer=checkpointer)


def to_evidence_bundle(state: dict[str, Any]) -> EvidenceBundle:
    """Project an evidence-only run onto the sources a specialist should read."""
    return EvidenceBundle(
        citations=tuple(state.get("citations", ())),
        hits=tuple(state.get("selected", ())),
        used_filter_fallback=bool(state.get("used_filter_fallback", False)),
    )


def to_rag_result(state: dict[str, Any]) -> RagResult:
    """Project the subgraph's terminal state onto the workflow's RAG contract."""
    status = state.get("status", "insufficient_evidence")
    if status != "answered":
        return RagResult(
            status="insufficient_evidence",
            hits=tuple(state.get("candidates", ())),
            used_filter_fallback=bool(state.get("used_filter_fallback", False)),
        )
    return RagResult(
        status="answered",
        answer=state.get("answer", ""),
        citations=tuple(state.get("citations", ())),
        hits=tuple(state.get("selected", ())),
        used_filter_fallback=bool(state.get("used_filter_fallback", False)),
        evidence_conflict=bool(state.get("evidence_conflict", False)),
    )
