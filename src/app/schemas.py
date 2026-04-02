"""API schemas."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from app.domain.models import AnalyzedInput, DetectedLanguage, InputAnalysis, QuoteOption


class HealthResponse(BaseModel):
    status: str = "ok"


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=1, description="Input email, message or short text.")


class QuoteRequest(BaseModel):
    text: str = Field(min_length=1, description="Input email, message or short text.")


class IngestRequest(BaseModel):
    input_path: Path | None = Field(
        default=None,
        description="Optional corpus path. Defaults to DATA_RAW_PATH.",
    )


class IngestResponse(BaseModel):
    input_path: str
    documents: int
    chunks: int
    embedding_backend: str
    vector_backend: str
    metadata_path: str
    index_path: str


class AnalyzeResponse(AnalyzedInput):
    pass


class QuoteResponse(BaseModel):
    detected_language: DetectedLanguage
    input_analysis: InputAnalysis
    recommended_quote: QuoteOption
    alternatives: list[QuoteOption]
