"""Core domain models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class LoadedDocument(BaseModel):
    """Document loaded from disk before chunking."""

    source_path: str
    text: str
    author: str
    work: str
    reference_prefix: str = "fragment"
    metadata: dict[str, Any] = Field(default_factory=dict)


class CorpusChunk(BaseModel):
    """Retrievable text chunk."""

    id: str
    text: str
    work: str
    author: str
    reference: str
    length: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class InputAnalysis(BaseModel):
    """Structured input analysis produced by the analyzer agent."""

    summary: str
    main_theme: str
    secondary_themes: list[str] = Field(default_factory=list)
    tone: str
    intent: str
    dominant_emotion: str
    recommended_quote_type: str


class DetectedLanguage(BaseModel):
    """Detected input language."""

    code: str
    name: str


class AnalyzedInput(BaseModel):
    """Detected language plus structured rhetorical analysis."""

    detected_language: DetectedLanguage
    input_analysis: InputAnalysis


class RetrievedCandidate(BaseModel):
    """Candidate quote with intermediate ranking signals."""

    chunk: CorpusChunk
    quote_text: str = ""
    quote_length: int = 0
    quote_sentence_count: int = 0
    quote_has_clear_boundaries: bool = False
    semantic_score: float = 0.0
    lexical_score: float = 0.0
    hybrid_score: float = 0.0
    thematic_fit: float = 0.0
    tonal_fit: float = 0.0
    clarity_fit: float = 0.0
    rhetorical_fit: float = 0.0
    rerank_score: float = 0.0
    why_it_fits: str = ""


class QuoteOption(BaseModel):
    """Public API quote representation."""

    quote_id: str
    text: str
    translated_text: str
    author: str
    work: str
    reference: str
    score: float
    why_it_fits: str


class SelectorChoice(BaseModel):
    """Selector response item."""

    quote_id: str
    why_it_fits: str
    score: float = Field(ge=0.0, le=1.0)


class SelectionResult(BaseModel):
    """Selector decision."""

    recommended_quote_id: str
    ranked_quotes: list[SelectorChoice]
