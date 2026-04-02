# Classical Quote Recommender — Business Documentation

## 1. Purpose

This application recommends classical literature quotes that match the rhetorical tone and intent of short texts like emails, Slack messages, or thread replies.

The current corpus covers Homer's Iliad and Odyssey (nearly 5,000 indexed passages). The system analyzes the user's message for theme, tone, emotion, and intent, then finds the classical passage that best fits rhetorically, not just topically.

It is a proof of concept (PoC), not a production product. There is no authentication, rate limiting, or multi-user state.

## 2. Actors

| Actor | Description |
|-------|-------------|
| End user | A person composing an email or message who wants a rhetorically fitting classical quote to include |
| Developer/Operator | Sets up the corpus, runs ingestion, deploys the service |

## 3. Use Cases

### UC-01: Get a Quote Recommendation

- **Actor:** End user
- **Preconditions:** The corpus has been ingested and the index exists
- **Main flow:**
  1. User submits a short text (email draft, message, reply)
  2. System detects the language of the input
  3. System analyzes the text for theme, tone, intent, and emotion
  4. System searches the corpus using enriched semantic + lexical retrieval
  5. System reranks candidates across five signals (hybrid score, thematic fit, tonal fit, clarity fit, rhetorical fit)
  6. System selects the top 3 quotes with explanations of why each fits
  7. If the input language differs from the quote language, system translates the quotes
  8. System returns one recommended quote plus alternatives, each with original text, translation, score, and explanation
- **Alternative flows:**
  - If Bedrock is unavailable: system returns a 503 error explaining that language-aware analysis and localization require Bedrock
- **Postconditions:** User receives 1 recommended quote + up to 2 alternatives

### UC-02: Ingest a Corpus

- **Actor:** Developer/Operator
- **Preconditions:** Raw corpus files (EPUB, TXT, or MD) exist in the input directory
- **Main flow:**
  1. Operator triggers ingestion via API (`POST /ingest`) or CLI (`classical-quotes ingest`)
  2. System loads documents from disk (parsing EPUB containers, front matter, etc.)
  3. System chunks documents into ~90-word passages with 20-word overlap
  4. System embeds all chunks using the configured model
  5. System saves the chunk metadata (JSONL) and vector index (NumPy .npy) to disk
- **Postconditions:** The index is ready for retrieval queries

### UC-03: Analyze Text Only

- **Actor:** End user
- **Preconditions:** None (does not require an ingested corpus)
- **Main flow:**
  1. User submits text via `POST /analyze`
  2. System returns detected language and rhetorical analysis (theme, tone, intent, emotion, recommended quote type)
- **Alternative flows:**
  - If Bedrock is unavailable and input is not English: returns 503
- **Postconditions:** User receives structured analysis without quote recommendations

### UC-04: Run Demo

- **Actor:** Developer/Operator
- **Preconditions:** Raw corpus files exist (index will be built automatically if missing)
- **Main flow:**
  1. Operator runs `classical-quotes demo`
  2. If no index exists, system auto-ingests the default corpus
  3. System runs the full pipeline on two hardcoded sample messages
  4. Results are printed to the terminal with Rich formatting
- **Postconditions:** Operator sees end-to-end output confirming the system works

## 4. Business Rules

| ID | Rule | Description |
|----|------|-------------|
| BR-01 | Quote length constraint | The system favors compact excerpts, usually one or two complete sentences and under roughly 32 words, so the result can be pasted naturally into a short message |
| BR-02 | Three quotes returned | The system always returns exactly 3 ranked quotes (1 recommended + 2 alternatives) |
| BR-03 | Bedrock dependency for language understanding | Language detection, rhetorical analysis, selection, and translation rely on Bedrock. If Bedrock is unavailable, the language-aware flows fail clearly instead of returning misleading output |
| BR-04 | Verbatim quotes only | All quotes come directly from the indexed corpus. The system never generates or paraphrases text |
| BR-05 | Language-aware output | Explanations (`why_it_fits`) and translations are written in the detected language of the user's input |
| BR-06 | Reranking weights | Hybrid retrieval score accounts for 45% of the final rank, thematic fit 20%, tonal fit 15%, clarity fit 10%, rhetorical fit 10% |

## 5. Integration Points

| External System | Integration Type | Flow Description |
|-----------------|------------------|------------------|
| AWS Bedrock (Claude Sonnet 4) | REST API via boto3 + Strands Agents SDK | Used for rhetorical analysis, quote selection, and translation. Configured via AWS profile (`sandbox`) in `eu-central-1` |
| Sentence Transformers (local) | In-process Python library | Loads `all-MiniLM-L6-v2` model locally for text embedding. Falls back to a deterministic hashing embedder if the model cannot be loaded |
