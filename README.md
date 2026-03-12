# EU CRR RAG

A Retrieval-Augmented Generation system for the EU Capital Requirements Regulation (CRR — Regulation (EU) No 575/2013). Ask compliance questions in plain English; get answers with article-level citations.

## What it does

1. Downloads the consolidated CRR HTML from EUR-Lex
2. Parses and indexes it preserving the full legal hierarchy: PART → TITLE → CHAPTER → SECTION → ARTICLE → Paragraph
3. Answers regulatory questions via a FastAPI HTTP API, citing the relevant articles

## Tech stack

| Layer | Technology |
|---|---|
| Parsing | BeautifulSoup4 (primary), LlamaParse (opt-in) |
| Embeddings | `bge-small-en-v1.5` (HuggingFace, local) |
| Vector store | Chroma (MVP) |
| Retrieval | AutoMergingRetriever + BM25, fused via Reciprocal Rank Fusion |
| Synthesis | OpenAI GPT-4o |
| API | FastAPI |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Fill in OPENAI_API_KEY in .env
# Optionally add LLAMA_CLOUD_API_KEY to enable LlamaParse
```

## Ingestion

```bash
# First run: --reset drops any existing index
python -m src.pipelines.ingest_pipeline --reset

# Custom EUR-Lex URL (e.g. a different consolidation date)
python -m src.pipelines.ingest_pipeline --url <url>
```

Ingestion produces ~740 article-level documents from the 2026-01-01 consolidation. Output is persisted to `chroma_db/`, `docstore/`, and `bm25_index.pkl`.

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

### Example query

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the risk weights for exposures to institutions under Article 114?"}'
```

Response shape:
```json
{
  "answer": "...",
  "sources": [...],
  "trace_id": "..."
}
```

## Architecture

```
EurLexIngester          downloads + parses EUR-Lex HTML → LlamaIndex Documents
      ↓
HierarchicalIndexer     HierarchicalNodeParser → Chroma vector index + BM25 index
      ↓
QueryEngine             AutoMergingRetriever (vector) + BM25Retriever
                        → QueryFusionRetriever (RRF) → GPT-4o synthesis
      ↓
FastAPI                 POST /api/query → {answer, sources, trace_id}
```

**AutoMergingRetriever:** when a child node (paragraph/point) is retrieved, it automatically merges up to the parent Article/Section to provide full context.

**Hybrid retrieval:** BM25 keyword precision + vector semantic recall, combined with Reciprocal Rank Fusion.

## Directory structure

```
eu-crr-rag/
├── api/
│   └── main.py                  # FastAPI app
├── src/
│   ├── ingestion/
│   │   └── eurlex_ingest.py     # EUR-Lex downloader + BeautifulSoup parser
│   ├── indexing/
│   │   ├── index_builder.py     # HierarchicalIndexer
│   │   └── vector_store.py      # Chroma wrapper
│   ├── models/
│   │   └── document.py          # DocumentNode dataclass + NodeLevel enum
│   ├── pipelines/
│   │   └── ingest_pipeline.py   # CLI entrypoint
│   └── query/
│       └── query_engine.py      # Hybrid retriever + GPT-4o synthesis
├── .env.example
└── requirements.txt
```

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | Used for GPT-4o synthesis |
| `LLAMA_CLOUD_API_KEY` | No | Enables LlamaParse for richer HTML parsing |
