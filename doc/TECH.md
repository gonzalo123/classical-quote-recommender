# Classical Quote Recommender — Technical Documentation

## 1. Overview

- **Name:** classical-quote-recommender
- **Type:** FastAPI REST service + Click CLI
- **Stack:** Python 3.12+, FastAPI, Click, Pydantic v2, NumPy, Sentence Transformers, Strands Agents SDK, AWS Bedrock (Claude Sonnet 4), RapidFuzz, Rich
- **Repository:** https://github.com/gonzalo123/nico
- **Package manager:** Poetry
- **Entry points:**
  - API: `src/app/main.py` (FastAPI app)
  - CLI: `src/cli.py` (Click group, installed as `classical-quotes`)

## 2. Directory Structure

```
src/
  app/
    api/
      routes.py            # FastAPI endpoint definitions
    agents/
      analyzer.py          # Rhetorical analysis agent (Bedrock/Strands)
      selector.py          # Quote selection agent (Bedrock/Strands + English heuristic fallback)
      factory.py           # Bedrock model builder
    domain/
      models.py            # Core Pydantic domain models
    embeddings/
      embedder.py          # SentenceTransformer + HashingEmbedder backends
      vector_store.py      # NumPy-based cosine similarity search
    ingestion/
      loader.py            # EPUB/TXT/MD corpus loader (stdlib only)
      chunker.py           # Paragraph-aware word-window chunker
      metadata.py          # JSONL persistence for CorpusChunk
    retrieval/
      semantic.py          # Vector-based nearest-neighbor retriever
      lexical.py           # RapidFuzz/difflib fuzzy text retriever
      hybrid.py            # Merges semantic + lexical with weighted scores
      reranker.py          # Multi-signal rule-based reranker
      quote_extractor.py   # Extracts compact quotable windows from chunks
    services/
      quote_service.py     # Main orchestration service (full pipeline)
      localization_service.py  # Bedrock-backed quote translation
    utils/
      logging.py           # basicConfig wrapper
      rich_cli.py          # Rich terminal rendering helpers
    errors.py              # Custom exceptions
    schemas.py             # API request/response schemas
    main.py                # FastAPI app factory
  commands/
    ingest.py              # CLI: index corpus
    quote.py               # CLI: recommend quotes for input text
    demo.py                # CLI: end-to-end demo with sample inputs
  cli.py                   # Click entry point
  settings.py              # All configuration (env-driven)
  env/
    development.env        # Dev environment variables
    test.env               # Test overrides (hash embedder, no agents)
  data/
    raw/                   # Raw corpus files (EPUB/TXT/MD)
    processed/             # chunks.jsonl after ingestion
    index/                 # vectors.npy + ids.json
tests/
  conftest.py              # Sets APP_ENV=test
  test_chunker.py          # Chunk stability tests
  test_metadata.py         # JSONL round-trip test
  test_health.py           # /health smoke test
  test_loader_epub.py      # EPUB loading from synthetic archive
  test_hybrid.py           # Hybrid retrieval ranking validation
  test_quote_localization.py  # Translation and localization tests
  test_quote_extractor.py  # Compact quote extraction tests
```

## 3. Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| click | ^8.1.8 | CLI framework |
| fastapi | ^0.115.12 | REST API framework |
| httpx | ^0.28.1 | HTTP client (FastAPI test client dependency) |
| numpy | ^2.2.4 | Vector store, matrix operations |
| boto3 | ^1.37.38 | AWS Bedrock client |
| pydantic | ^2.11.3 | Data validation, domain models, structured output |
| pydantic-settings | ^2.8.1 | Settings management |
| python-dotenv | ^1.1.0 | .env file loading |
| rapidfuzz | ^3.13.0 | Fuzzy string matching for lexical retrieval |
| rich | ^14.0.0 | Terminal formatting for CLI output |
| sentence-transformers | ^4.1.0 | Text embedding model (all-MiniLM-L6-v2) |
| strands-agents | ^1.21.0 | AI agent framework for Bedrock integration |
| uvicorn | ^0.34.0 | ASGI server |
| pytest | ^8.3.5 | Test framework (dev dependency) |

## 4. API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/health` | Health check, returns `{"status": "ok"}` |
| POST | `/ingest` | Index the raw corpus. Body: `{"input_path": "/optional/path"}` |
| POST | `/analyze` | Rhetorical analysis only. Body: `{"text": "..."}` |
| POST | `/quote` | Full pipeline: analyze, retrieve, rerank, select, translate. Body: `{"text": "..."}` |
| GET | `/quotes/{quote_id}` | Retrieve a single indexed quote by ID |

## 5. CLI Commands

| Command | Arguments | Description |
|---------|-----------|-------------|
| `classical-quotes ingest` | `--input PATH` (optional, directory) | Index the corpus from raw files into chunks + vector index |
| `classical-quotes quote` | `--text TEXT` (required) | Run the full pipeline on input text, print recommended quotes |
| `classical-quotes demo` | (none) | Run end-to-end demo with two hardcoded sample inputs. Auto-ingests if no index exists |

## 6. Configuration

### Environment Variables

Loaded from `src/env/{APP_ENV}.env` based on the `APP_ENV` variable (defaults to `development`).

| Variable | Description | Default |
|----------|-------------|---------|
| `APP_ENV` | Active environment (`development`, `test`, or any custom value) | `development` |
| `APP_NAME` | Application name | `classical-quote-recommender` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `DATA_RAW_PATH` | Path to raw corpus directory | `src/data/raw` |
| `DATA_PROCESSED_PATH` | Path to processed chunks JSONL | `src/data/processed/chunks.jsonl` |
| `DATA_INDEX_PATH` | Path to vector index directory | `src/data/index` |
| `CHUNK_SIZE_WORDS` | Words per chunk | `90` |
| `CHUNK_OVERLAP_WORDS` | Overlap between chunks | `20` |
| `MIN_CHUNK_WORDS` | Minimum words to keep a chunk | `20` |
| `EMBEDDING_BACKEND` | `sentence-transformers` or `hash` | `sentence-transformers` |
| `EMBEDDING_MODEL` | Sentence Transformers model name | `sentence-transformers/all-MiniLM-L6-v2` |
| `EMBEDDING_DEVICE` | PyTorch device | `cpu` |
| `EMBEDDING_BATCH_SIZE` | Batch size for embedding | `16` |
| `EMBEDDING_DIMENSION` | Vector dimension (used by hash fallback) | `256` |
| `VECTOR_BACKEND` | Vector store backend | `numpy` |
| `SEMANTIC_WEIGHT` | Weight for semantic score in hybrid merge | `0.72` |
| `LEXICAL_WEIGHT` | Weight for lexical score in hybrid merge | `0.28` |
| `SEMANTIC_CANDIDATE_COUNT` | Candidates from semantic search | `20` |
| `LEXICAL_CANDIDATE_COUNT` | Candidates from lexical search | `20` |
| `RERANK_CANDIDATE_COUNT` | Candidates passed to reranker | `10` |
| `SELECTOR_CANDIDATE_POOL` | Candidates shown to LLM selector | `6` |
| `FINAL_QUOTE_COUNT` | Number of quotes returned | `3` |
| `BEDROCK_MODEL_ID` | AWS Bedrock model ID | `eu.anthropic.claude-sonnet-4-20250514-v1:0` |
| `BEDROCK_PROFILE` | AWS CLI profile name | `sandbox` |
| `BEDROCK_REGION` | AWS region | `eu-central-1` |
| `BEDROCK_ENDPOINT_URL` | Custom Bedrock endpoint (optional) | (empty) |
| `BEDROCK_TEMPERATURE` | LLM temperature | `0.1` |
| `USE_AGENT_ANALYSIS` | Enable LLM-based analysis | `true` |
| `USE_AGENT_SELECTION` | Enable LLM-based selection | `true` |

### Environment Files

| File | Purpose |
|------|---------|
| `src/env/development.env` | Development defaults (full stack with Bedrock) |
| `src/env/test.env` | Test overrides: `EMBEDDING_BACKEND=hash`, agents disabled |

## 7. Build, Test and Run

### Setup

```bash
poetry install
```

### Run API

```bash
make run
# or: PYTHONPATH=src poetry run uvicorn app.main:app --reload
```

### Ingest Corpus

```bash
make ingest
# or: poetry run classical-quotes ingest --input ./src/data/raw
```

### Run Demo

```bash
make demo
# or: poetry run classical-quotes demo
```

### Run Tests

```bash
make test
# or: poetry run pytest -q
```

### Clean

```bash
make clean
# Removes caches plus generated chunks/index artifacts
```

### Makefile Targets

| Target | Command |
|--------|---------|
| `test` | `poetry run pytest -q` |
| `clean` | Remove cache, compiled files, and generated processed/index artifacts |
| `ingest` | `poetry run classical-quotes ingest --input ./src/data/raw` |
| `run` | `PYTHONPATH=src poetry run uvicorn app.main:app --reload` |
| `demo` | `poetry run classical-quotes demo` |
