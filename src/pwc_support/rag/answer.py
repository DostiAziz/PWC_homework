from __future__ import annotations

from typing import Any

from pwc_support.domain.models import EvidenceBundle, RagRequest, RagResult
from pwc_support.rag.subgraph import (
    Generator,
    KnowledgeBase,
    build_rag_graph,
    prepare_query,
    to_evidence_bundle,
    to_rag_result,
)

__all__ = ["RagAnswerer", "prepare_query"]


class RagAnswerer:
    """Invoke the compiled RAG subgraph and expose its result to the main workflow."""

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        generator: Generator,
        *,
        minimum_similarity: float = 0.45,
        max_selected_hits: int = 4,
        max_evidence_chars: int = 6000,
        answer_tokens: int = 512,
    ) -> None:
        self.knowledge_base = knowledge_base
        self.generator = generator
        self.minimum_similarity = minimum_similarity
        self.graph = build_rag_graph(
            knowledge_base,
            generator,
            minimum_similarity=minimum_similarity,
            max_selected_hits=max_selected_hits,
            max_evidence_chars=max_evidence_chars,
            answer_tokens=answer_tokens,
        )
        # The same retrieval and selection, compiled without the generation node, so an
        # escalated enquiry can be researched for a specialist but never answered by the
        # model. Both graphs share one threshold, so the specialist sees exactly the
        # evidence the answering path would have used.
        self.evidence_graph = build_rag_graph(
            knowledge_base,
            generator,
            minimum_similarity=minimum_similarity,
            max_selected_hits=max_selected_hits,
            max_evidence_chars=max_evidence_chars,
            answer_tokens=answer_tokens,
            generate=False,
        )

    def answer(self, request: RagRequest) -> RagResult:
        return to_rag_result(self.run(request))

    def run(self, request: RagRequest) -> dict[str, Any]:
        """Return the subgraph's full terminal state, including per-node timings."""
        state: dict[str, Any] = self.graph.invoke({"request": request})
        return state

    def gather_evidence(self, request: RagRequest) -> EvidenceBundle:
        """Retrieve and select sources without generating anything."""
        return to_evidence_bundle(self.evidence_graph.invoke({"request": request}))
