"""API routes."""

from __future__ import annotations

from functools import lru_cache

from fastapi import APIRouter, HTTPException

from app.errors import LocalizationUnavailableError
from app.schemas import AnalyzeRequest, AnalyzeResponse, HealthResponse, IngestRequest, QuoteRequest, QuoteResponse
from app.services.quote_service import QuoteService

router = APIRouter()


@lru_cache(maxsize=1)
def get_quote_service() -> QuoteService:
    """Provide a cached quote service instance."""

    return QuoteService()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@router.post("/ingest")
def ingest(request: IngestRequest):
    try:
        return get_quote_service().ingest(request.input_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):
    try:
        return get_quote_service().analyze(request.text)
    except LocalizationUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/quote", response_model=QuoteResponse)
def quote(request: QuoteRequest):
    try:
        return get_quote_service().quote(request.text)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except LocalizationUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/quotes/{quote_id}")
def get_quote(quote_id: str):
    try:
        quote = get_quote_service().get_quote(quote_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if quote is None:
        raise HTTPException(status_code=404, detail=f"Quote '{quote_id}' not found.")
    return quote
