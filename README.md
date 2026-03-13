# EU CRR RAG

A Retrieval-Augmented Generation system for the EU Capital Requirements Regulation (CRR — Regulation (EU) No 575/2013). Ask compliance questions in English, Italian, or Polish; get answers with article-level citations.

## What it does

1. Downloads (or reads from a local file) the consolidated CRR HTML from EUR-Lex
2. Parses and indexes it preserving the full legal hierarchy: PART → TITLE → CHAPTER → SECTION → ARTICLE → Paragraph
3. Supports multilingual ingestion (EN / IT / PL) in a single shared index with per-node language metadata
4. Answers regulatory questions via a FastAPI HTTP API, citing the relevant articles

## Tech stack

| Layer | Technology |
|---|---|
| Parsing | BeautifulSoup4 (primary), LlamaParse (opt-in) |
| Embeddings | `BAAI/bge-m3` (HuggingFace, local — multilingual, dense + sparse) |
| Vector store | Qdrant Cloud (native hybrid dense + sparse search) |
| Retrieval | AutoMergingRetriever + Qdrant native hybrid (dense + sparse) |
| Synthesis | OpenAI GPT-4o |
| API | FastAPI |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Fill in OPENAI_API_KEY, QDRANT_URL, and QDRANT_API_KEY in .env
# Optionally add LLAMA_CLOUD_API_KEY to enable LlamaParse
```

> **Note:** BGE-M3 is ~570 MB and will be downloaded automatically on first use.

## Ingestion

```bash
# First run: --reset creates a fresh Qdrant collection
python -m src.pipelines.ingest_pipeline --reset --language en --file crr_raw_en.html

# Add Italian (no --reset: appends to existing collection)
python -m src.pipelines.ingest_pipeline --language it --file crr_raw_ita.html

# Add Polish
python -m src.pipelines.ingest_pipeline --language pl --file crr_raw_pl.html

# Download directly from EUR-Lex instead of using a local file
python -m src.pipelines.ingest_pipeline --reset --language en
```

> **Warning:** `--reset` drops the **entire** Qdrant collection (all languages). Only use it when starting completely fresh.

Ingestion produces ~742 article-level documents per language. Node hierarchy and docstore are persisted to `docstore/`.

### CLI reference

| Flag | Default | Description |
|---|---|---|
| `--language` | `en` | Language to ingest: `en`, `it`, or `pl` |
| `--file` | — | Path to a local HTML file (skips network download) |
| `--url` | auto-derived | EUR-Lex URL (derived from `--language` if not set) |
| `--reset` | off | Drop and recreate the Qdrant collection before ingesting |

## Running the API

```bash
uvicorn api.main:app --reload
```

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check; reports whether the index is loaded |
| `POST` | `/api/query` | Ask a question |
| `POST` | `/api/ingest` | Trigger ingestion programmatically |

### Example queries

```bash
# English query (auto-detected, or explicit)
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the minimum CET1 requirements?", "preferred_language": "en"}'

# Italian query (language auto-detected from accent characters)
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Quali sono i requisiti minimi CET1?"}'

# Cross-language retrieval (no language filter)
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the risk weights for exposures to institutions under Article 114?"}'
```

Response shape:
```json
{
  "answer": "...",
  "sources": [
    {
      "text": "...",
      "score": 0.87,
      "metadata": {"article": "92", "part": "TWO", "language": "en", ...}
    }
  ],
  "trace_id": "..."
}
```

If `preferred_language` is omitted, the API detects Polish (from `ąćęłńóśźż…`) and Italian (from `àèéìíîòóùú…`) automatically. Undetected queries retrieve across all languages.

## Architecture

```
EurLexIngester          downloads/reads EUR-Lex HTML → LlamaIndex Documents
      ↓                 (language-aware heading classification; stamps language on every node)
HierarchicalIndexer     HierarchicalNodeParser → Qdrant hybrid vector index (dense + sparse)
      ↓                 docstore persisted for AutoMergingRetriever
QueryEngine             AutoMergingRetriever (vector) + Qdrant native hybrid (BGE-M3 dense+sparse)
                        → optional MetadataFilter(language) → GPT-4o synthesis
      ↓
FastAPI                 POST /api/query → {answer, sources, trace_id}
```

**AutoMergingRetriever:** when a child node (paragraph) is retrieved, it automatically merges up to the parent Article/Section to provide full context.

**Hybrid retrieval:** BGE-M3 dense semantic recall + BGE-M3 sparse (lexical) precision, fused natively by Qdrant — no separate BM25 index needed.

**Multilingual:** a single Qdrant collection holds all languages. Each node carries a `language` metadata field; queries can filter by language or retrieve cross-language.

## Directory structure

```
eu-crr-rag/
├── api/
│   └── main.py                    # FastAPI app
├── src/
│   ├── ingestion/
│   │   ├── eurlex_ingest.py       # EUR-Lex downloader + BeautifulSoup parser
│   │   └── language_config.py     # Heading keywords + URL templates for EN/IT/PL
│   ├── indexing/
│   │   ├── bge_m3_sparse.py       # BGE-M3 sparse encoding for Qdrant hybrid search
│   │   ├── index_builder.py       # HierarchicalIndexer
│   │   └── vector_store.py        # Qdrant Cloud wrapper
│   ├── models/
│   │   └── document.py            # DocumentNode dataclass + NodeLevel enum
│   ├── pipelines/
│   │   └── ingest_pipeline.py     # CLI entrypoint
│   └── query/
│       └── query_engine.py        # Hybrid retriever + GPT-4o synthesis
├── .env.example
└── requirements.txt
```

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | Used for GPT-4o synthesis |
| `QDRANT_URL` | Yes | Qdrant Cloud cluster URL (e.g. `https://xyz.qdrant.io`) |
| `QDRANT_API_KEY` | Yes | Qdrant Cloud API key |
| `LLAMA_CLOUD_API_KEY` | No | Enables LlamaParse for richer HTML parsing |
