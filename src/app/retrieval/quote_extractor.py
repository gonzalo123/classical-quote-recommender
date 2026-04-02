"""Extract compact quote windows from larger retrieved chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass
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

from app.domain.models import InputAnalysis
from settings import QUOTE_MAX_SENTENCES, QUOTE_MAX_WORDS, QUOTE_MIN_WORDS

SENTENCE_RE = re.compile(r'[^.!?]+[.!?]+(?:["\')\]]+)?')
BOUNDARY_RE = re.compile(r'.*[.!?](?:["\')\]]+)?$')
TOKEN_RE = re.compile(r"[A-Za-z']+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "both",
    "but",
    "for",
    "from",
    "i",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "so",
    "still",
    "that",
    "the",
    "their",
    "them",
    "they",
    "this",
    "to",
    "toward",
    "we",
    "with",
}


@dataclass(slots=True)
class ExtractedQuote:
    """Compact quote excerpt derived from an indexed chunk."""

    text: str
    word_count: int
    sentence_count: int
    has_clear_boundaries: bool


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _split_into_sentences(text: str) -> list[str]:
    normalized = _normalize_whitespace(text)
    if not normalized:
        return []
    sentences = [match.group(0).strip() for match in SENTENCE_RE.finditer(normalized)]
    return [sentence for sentence in sentences if sentence]


def _has_clear_boundaries(text: str) -> bool:
    return bool(text) and bool(BOUNDARY_RE.match(text.strip()))


def _starts_like_a_clean_quote(text: str) -> bool:
    cleaned = text.strip()
    if not cleaned:
        return False
    if re.match(r'^(And|But|So|Yet|Or|Nor)\b', cleaned):
        return False
    return cleaned[0].isalnum() or cleaned[0] in {'"', "'"}


def _length_fit(word_count: int) -> float:
    if 8 <= word_count <= 24:
        return 1.0
    if QUOTE_MIN_WORDS <= word_count <= QUOTE_MAX_WORDS:
        return 0.82
    if word_count < QUOTE_MIN_WORDS:
        return 0.45
    if word_count <= max(QUOTE_MAX_WORDS + 10, 40):
        return 0.35
    return 0.15


def _keyword_overlap(query: str, candidate: str) -> float:
    query_tokens = {
        token.lower()
        for token in TOKEN_RE.findall(query)
        if len(token) > 3 and token.lower() not in STOPWORDS
    }
    if not query_tokens:
        return 0.0
    candidate_tokens = {
        token.lower()
        for token in TOKEN_RE.findall(candidate)
        if len(token) > 3 and token.lower() not in STOPWORDS
    }
    if not candidate_tokens:
        return 0.0
    return len(query_tokens & candidate_tokens) / len(query_tokens)


def _window_score(candidate: str, query: str, sentence_count: int) -> float:
    word_count = len(candidate.split())
    relevance = (
        max(
            fuzz.token_set_ratio(query, candidate) / 100.0,
            fuzz.partial_ratio(query, candidate) / 100.0,
        )
        if query
        else 0.0
    )
    structure = 0.15 if _has_clear_boundaries(candidate) else -0.30
    start_bonus = 0.08 if _starts_like_a_clean_quote(candidate) else -0.18
    sentence_bonus = 0.24 if sentence_count == 1 else 0.04 if sentence_count == 2 else -0.20
    overlap_bonus = 1.20 * _keyword_overlap(query, candidate)
    return relevance + _length_fit(word_count) + structure + start_bonus + sentence_bonus + overlap_bonus


def extract_compact_quote(
    text: str,
    input_text: str,
    analysis: InputAnalysis,
) -> ExtractedQuote:
    """Pick one or two complete sentences from a chunk for quoting."""

    normalized = _normalize_whitespace(text)
    if not normalized:
        return ExtractedQuote(text="", word_count=0, sentence_count=0, has_clear_boundaries=False)

    sentences = _split_into_sentences(normalized)
    query = " ".join(
        [
            input_text,
            analysis.summary,
            analysis.main_theme,
            " ".join(analysis.secondary_themes),
            analysis.tone,
            analysis.intent,
            analysis.dominant_emotion,
            analysis.recommended_quote_type,
        ]
    ).strip()

    best_window = ""
    best_score = float("-inf")

    if sentences:
        max_sentences = max(1, QUOTE_MAX_SENTENCES)
        for start in range(len(sentences)):
            for size in range(1, max_sentences + 1):
                window = sentences[start : start + size]
                if not window:
                    continue
                candidate = " ".join(window).strip()
                word_count = len(candidate.split())
                if word_count > max(QUOTE_MAX_WORDS + 12, 40):
                    continue
                score = _window_score(candidate, query, len(window))
                if score > best_score:
                    best_score = score
                    best_window = candidate

    if not best_window:
        trimmed_words = normalized.split()[:QUOTE_MAX_WORDS]
        best_window = " ".join(trimmed_words).strip()
        last_boundary = max(best_window.rfind("."), best_window.rfind("!"), best_window.rfind("?"))
        if last_boundary > 0:
            best_window = best_window[: last_boundary + 1].strip()

    return ExtractedQuote(
        text=best_window or normalized,
        word_count=len((best_window or normalized).split()),
        sentence_count=max(len(_split_into_sentences(best_window or normalized)), 1),
        has_clear_boundaries=_has_clear_boundaries(best_window or normalized),
    )
