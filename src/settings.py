"""Application settings."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
ENV_DIR = BASE_DIR / "env"
APP_ENV = os.getenv("APP_ENV", "development")
ENV_FILE = ENV_DIR / f"{APP_ENV}.env"

if ENV_FILE.exists():
    load_dotenv(ENV_FILE)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_path(name: str, default: Path) -> Path:
    value = Path(os.getenv(name, str(default)))
    if value.is_absolute():
        return value
    return (PROJECT_ROOT / value).resolve()


APP_NAME = os.getenv("APP_NAME", "classical-quote-recommender")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

DATA_RAW_PATH = _env_path("DATA_RAW_PATH", PROJECT_ROOT / "src" / "data" / "raw")
DATA_PROCESSED_PATH = _env_path(
    "DATA_PROCESSED_PATH",
    PROJECT_ROOT / "src" / "data" / "processed" / "chunks.jsonl",
)
DATA_INDEX_PATH = _env_path("DATA_INDEX_PATH", PROJECT_ROOT / "src" / "data" / "index")

CHUNK_SIZE_WORDS = int(os.getenv("CHUNK_SIZE_WORDS", "90"))
CHUNK_OVERLAP_WORDS = int(os.getenv("CHUNK_OVERLAP_WORDS", "20"))
MIN_CHUNK_WORDS = int(os.getenv("MIN_CHUNK_WORDS", "20"))

EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "sentence-transformers")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "256"))

VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "numpy")

SEMANTIC_WEIGHT = float(os.getenv("SEMANTIC_WEIGHT", "0.72"))
LEXICAL_WEIGHT = float(os.getenv("LEXICAL_WEIGHT", "0.28"))
SEMANTIC_CANDIDATE_COUNT = int(os.getenv("SEMANTIC_CANDIDATE_COUNT", "20"))
LEXICAL_CANDIDATE_COUNT = int(os.getenv("LEXICAL_CANDIDATE_COUNT", "20"))
RERANK_CANDIDATE_COUNT = int(os.getenv("RERANK_CANDIDATE_COUNT", "10"))
SELECTOR_CANDIDATE_POOL = int(os.getenv("SELECTOR_CANDIDATE_POOL", "6"))
FINAL_QUOTE_COUNT = int(os.getenv("FINAL_QUOTE_COUNT", "3"))
QUOTE_MIN_WORDS = int(os.getenv("QUOTE_MIN_WORDS", "6"))
QUOTE_MAX_WORDS = int(os.getenv("QUOTE_MAX_WORDS", "32"))
QUOTE_MAX_SENTENCES = int(os.getenv("QUOTE_MAX_SENTENCES", "2"))

BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "eu.anthropic.claude-sonnet-4-20250514-v1:0")
BEDROCK_PROFILE = os.getenv("BEDROCK_PROFILE", "sandbox")
BEDROCK_REGION = os.getenv("BEDROCK_REGION", "eu-central-1")
BEDROCK_ENDPOINT_URL = os.getenv("BEDROCK_ENDPOINT_URL", "")
BEDROCK_TEMPERATURE = float(os.getenv("BEDROCK_TEMPERATURE", "0.1"))

USE_AGENT_ANALYSIS = _env_bool("USE_AGENT_ANALYSIS", True)
USE_AGENT_SELECTION = _env_bool("USE_AGENT_SELECTION", True)
