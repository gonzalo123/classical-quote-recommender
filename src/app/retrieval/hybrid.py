"""Hybrid retrieval combining semantic and lexical signals."""

from __future__ import annotations

from app.domain.models import CorpusChunk, InputAnalysis, RetrievedCandidate
from app.retrieval.lexical import LexicalRetriever
from app.retrieval.semantic import SemanticRetriever
from settings import (
    LEXICAL_CANDIDATE_COUNT,
    LEXICAL_WEIGHT,
    SEMANTIC_CANDIDATE_COUNT,
    SEMANTIC_WEIGHT,
)


class HybridRetriever:
    """Combine semantic and lexical retrieval into a shared candidate set."""

    def __init__(
        self,
        semantic_retriever: SemanticRetriever,
        lexical_retriever: LexicalRetriever,
        chunks_by_id: dict[str, CorpusChunk],
    ) -> None:
        self.semantic_retriever = semantic_retriever
        self.lexical_retriever = lexical_retriever
        self.chunks_by_id = chunks_by_id

    def _build_query(self, text: str, analysis: InputAnalysis) -> str:
        parts = [
            text,
            analysis.summary,
            analysis.main_theme,
            " ".join(analysis.secondary_themes),
            analysis.tone,
            analysis.intent,
            analysis.dominant_emotion,
            analysis.recommended_quote_type,
        ]
        return " ".join(part for part in parts if part).strip()

    def retrieve(self, text: str, analysis: InputAnalysis, top_k: int) -> list[RetrievedCandidate]:
        query = self._build_query(text, analysis)
        semantic_hits = self.semantic_retriever.search(
            query,
            top_k=max(top_k, SEMANTIC_CANDIDATE_COUNT),
        )
        lexical_hits = self.lexical_retriever.search(
            query,
            list(self.chunks_by_id.values()),
            top_k=max(top_k, LEXICAL_CANDIDATE_COUNT),
        )

        merged: dict[str, RetrievedCandidate] = {}
        for chunk, score in semantic_hits:
            merged[chunk.id] = RetrievedCandidate(chunk=chunk, semantic_score=max(score, 0.0))

        for chunk, score in lexical_hits:
            candidate = merged.get(chunk.id, RetrievedCandidate(chunk=chunk))
            candidate.lexical_score = score
            merged[chunk.id] = candidate

        for candidate in merged.values():
            candidate.hybrid_score = (
                SEMANTIC_WEIGHT * candidate.semantic_score
                + LEXICAL_WEIGHT * candidate.lexical_score
            )

        ordered = sorted(merged.values(), key=lambda item: item.hybrid_score, reverse=True)
        return ordered[:top_k]
