"""Heuristic reranker for quote candidates."""

from __future__ import annotations

from difflib import SequenceMatcher

try:  # pragma: no branch - one branch depends on installed extras
    from rapidfuzz import fuzz
except ImportError:  # pragma: no cover - only used in minimal environments
    class _FallbackFuzz:
        @staticmethod
        def token_set_ratio(left: str, right: str) -> float:
            return SequenceMatcher(None, left.lower(), right.lower()).ratio() * 100.0

        @staticmethod
        def partial_ratio(left: str, right: str) -> float:
            return SequenceMatcher(None, left.lower(), right.lower()).ratio() * 100.0

    fuzz = _FallbackFuzz()

from app.domain.models import InputAnalysis, RetrievedCandidate
from app.retrieval.quote_extractor import extract_compact_quote

TONE_KEYWORDS = {
    "conciliatory": {"measured", "together", "shared", "yielded", "counsel", "fairness"},
    "courteous": {"courtesy", "measured", "discipline", "fairness"},
    "resolute": {"action", "purpose", "deed", "command"},
    "reflective": {"mind", "wiser", "tested", "understanding"},
    "urgent": {"now", "swift", "immediate", "action"},
}

INTENT_KEYWORDS = {
    "negotiate": {"middle", "together", "shared", "yielded", "common"},
    "reply": {"answer", "reply", "word", "speaker"},
    "thank": {"grace", "courtesy", "gift", "shared"},
    "persuade": {"useful", "deed", "discipline", "action"},
    "apologize": {"healed", "understanding", "room", "measured"},
}


class RuleBasedReranker:
    """Compute richer scores after hybrid retrieval."""

    def _keyword_fit(self, keywords: set[str], text: str) -> float:
        lowered = text.lower()
        if not keywords:
            return 0.0
        hits = sum(1 for keyword in keywords if keyword in lowered)
        return min(hits / max(len(keywords), 1), 1.0)

    def _clarity_fit(
        self,
        word_count: int,
        sentence_count: int,
        has_clear_boundaries: bool,
    ) -> float:
        score = 0.0
        if 8 <= word_count <= 24:
            score += 0.65
        elif 6 <= word_count <= 32:
            score += 0.50
        elif word_count <= 40:
            score += 0.30
        else:
            score += 0.10

        if sentence_count == 1:
            score += 0.20
        elif sentence_count == 2:
            score += 0.15
        else:
            score -= 0.10

        score += 0.15 if has_clear_boundaries else -0.20
        return max(0.0, min(score, 1.0))

    def rerank(
        self,
        input_text: str,
        analysis: InputAnalysis,
        candidates: list[RetrievedCandidate],
    ) -> list[RetrievedCandidate]:
        """Rerank candidates with rhetorical heuristics."""

        theme_query = " ".join([analysis.main_theme, *analysis.secondary_themes, analysis.intent])
        tone_keywords = TONE_KEYWORDS.get(analysis.tone.lower(), set())
        intent_keywords = INTENT_KEYWORDS.get(analysis.intent.lower(), set())

        for candidate in candidates:
            extracted_quote = extract_compact_quote(candidate.chunk.text, input_text, analysis)
            candidate.quote_text = extracted_quote.text
            candidate.quote_length = extracted_quote.word_count
            candidate.quote_sentence_count = extracted_quote.sentence_count
            candidate.quote_has_clear_boundaries = extracted_quote.has_clear_boundaries
            candidate_text = candidate.quote_text or candidate.chunk.text
            source_text = candidate.chunk.text
            candidate.thematic_fit = max(
                fuzz.token_set_ratio(theme_query, candidate_text) / 100.0,
                fuzz.token_set_ratio(theme_query, source_text) / 100.0,
                fuzz.partial_ratio(input_text, candidate_text) / 100.0,
                fuzz.partial_ratio(input_text, source_text) / 100.0,
            )
            candidate.tonal_fit = max(
                self._keyword_fit(tone_keywords, candidate_text),
                self._keyword_fit(tone_keywords, source_text),
            )
            candidate.clarity_fit = self._clarity_fit(
                candidate.quote_length or candidate.chunk.length,
                candidate.quote_sentence_count,
                candidate.quote_has_clear_boundaries,
            )
            candidate.rhetorical_fit = max(
                self._keyword_fit(intent_keywords, candidate_text),
                self._keyword_fit(intent_keywords, source_text),
                fuzz.token_set_ratio(analysis.recommended_quote_type, candidate_text) / 100.0,
                fuzz.token_set_ratio(analysis.recommended_quote_type, source_text) / 100.0,
            )
            candidate.rerank_score = (
                0.45 * candidate.hybrid_score
                + 0.20 * candidate.thematic_fit
                + 0.15 * candidate.tonal_fit
                + 0.10 * candidate.clarity_fit
                + 0.10 * candidate.rhetorical_fit
            )

        return sorted(candidates, key=lambda item: item.rerank_score, reverse=True)
