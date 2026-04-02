"""Lexical retrieval based on RapidFuzz with a difflib fallback."""

from __future__ import annotations

from difflib import SequenceMatcher

try:  # pragma: no branch - one branch depends on installed extras
    from rapidfuzz import fuzz
except ImportError:  # pragma: no cover - only used in minimal environments
    class _FallbackFuzz:
        @staticmethod
        def token_set_ratio(left: str, right: str) -> float:
            left_tokens = set(left.lower().split())
            right_tokens = set(right.lower().split())
            if not left_tokens or not right_tokens:
                return 0.0
            overlap = " ".join(sorted(left_tokens & right_tokens))
            left_joined = " ".join(sorted(left_tokens))
            right_joined = " ".join(sorted(right_tokens))
            return max(
                SequenceMatcher(None, overlap, left_joined).ratio(),
                SequenceMatcher(None, overlap, right_joined).ratio(),
            ) * 100.0

        @staticmethod
        def partial_ratio(left: str, right: str) -> float:
            return SequenceMatcher(None, left.lower(), right.lower()).ratio() * 100.0

    fuzz = _FallbackFuzz()

from app.domain.models import CorpusChunk


class LexicalRetriever:
    """Score chunks with lightweight lexical similarity."""

    def score(self, query: str, chunk: CorpusChunk) -> float:
        haystack = " ".join([chunk.text, chunk.work, chunk.author, chunk.reference]).strip()
        return max(
            fuzz.token_set_ratio(query, haystack),
            fuzz.partial_ratio(query, haystack),
        ) / 100.0

    def search(
        self,
        query: str,
        chunks: list[CorpusChunk],
        top_k: int,
    ) -> list[tuple[CorpusChunk, float]]:
        scored = [(chunk, self.score(query, chunk)) for chunk in chunks]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]
