from __future__ import annotations

from typing import Any

from pwc_support.domain.models import Citation, RagRequest, RagResult


def prepare_query(question: str) -> str:
    normalized = question.casefold()
    refers_to_assistant = " you" in f" {normalized}" or " your" in f" {normalized}"
    if refers_to_assistant and "pwc" not in normalized:
        return f"{question} PwC business services"
    return question


class RagAnswerer:
    """Answer only from retrieved evidence and expose its source mapping."""

    def __init__(
        self, knowledge_base: Any, generator: Any, *, minimum_similarity: float = 0.45
    ) -> None:
        self.knowledge_base = knowledge_base
        self.generator = generator
        self.minimum_similarity = minimum_similarity

    def answer(self, request: RagRequest) -> RagResult:
        prepared_request = request.model_copy(
            update={"question": prepare_query(request.question)}
        )
        batch = self.knowledge_base.retrieve(prepared_request)
        hits = tuple(
            hit
            for hit in batch.hits
            if hit.similarity >= self.minimum_similarity or hit.source_type == "lexical"
        )
        if not hits:
            return RagResult(
                status="insufficient_evidence",
                hits=batch.hits,
                used_filter_fallback=batch.used_filter_fallback,
            )
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
            for index, hit in enumerate(hits, start=1)
        )
        evidence = "\n\n".join(
            f"{citation.marker} {hit.title}: {hit.text}"
            for citation, hit in zip(citations, hits, strict=True)
        )
        answer = self.generator.text(
            system="Answer only from the evidence. Keep the citation markers.",
            user=f"Question: {request.question}\nEvidence:\n{evidence}",
        )
        return RagResult(
            status="answered",
            answer=answer,
            citations=citations,
            hits=hits,
            used_filter_fallback=batch.used_filter_fallback,
            evidence_conflict=batch.evidence_conflict,
        )
