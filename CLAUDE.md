# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Status

Scaffolding complete. See `WORKLOG.md` for current state and next steps.

## What This Project Does

A RAG system for the EU Capital Requirements Regulation (CRR – Regulation (EU) No 575/2013). It:
1. Ingests the consolidated CRR HTML from EUR-Lex
2. Parses and indexes it preserving legal hierarchy: PART → TITLE → CHAPTER → SECTION → ARTICLE → Paragraph → Point
3. Answers regulatory compliance questions with citations via a FastAPI HTTP API

## Tech Stack

- Python 3.11+, FastAPI, LlamaIndex, LlamaParse, Chroma (vector store), OpenAI GPT-4o, HuggingFace `bge-small-en-v1.5` embeddings, BeautifulSoup4 + requests (HTML fallback)

## Commands

```bash
pip install -r requirements.txt

# Ingest (first time: --reset drops any existing index)
python -m src.pipelines.ingest_pipeline --reset
python -m src.pipelines.ingest_pipeline --url <custom-EUR-Lex-URL>

# API server
uvicorn api.main:app --reload
```

Env vars (copy `.env.example` → `.env`): `OPENAI_API_KEY` (required), `LLAMA_CLOUD_API_KEY` (optional, enables LlamaParse).

## Architecture

### Data Flow

1. **EurLexIngester** (`src/ingestion/eurlex_ingest.py`) — Downloads CRR HTML from EUR-Lex; parses with LlamaParse (primary) or BeautifulSoup (fallback)
2. **HierarchicalIndexer** (`src/indexing/index_builder.py`) — Applies `HierarchicalNodeParser`, enriches nodes with hierarchy metadata, builds Chroma vector index + BM25 index
3. **VectorStore** (`src/indexing/vector_store.py`) — Chroma client wrapper
4. **QueryEngine** (`src/query/query_engine.py`) — Hybrid retrieval: `AutoMergingRetriever` (vector) + `BM25Retriever` fused via `QueryFusionRetriever` (Reciprocal Rank Fusion), then GPT-4o synthesis
5. **FastAPI app** (`api/main.py`) — Exposes `POST /api/query` and `POST /api/ingest`

### Key Design Decisions

- **AutoMergingRetriever**: when a child node (paragraph/point) is retrieved, it automatically merges up to the parent Article/Section for full context
- **Hybrid retrieval (BM25 + vector)**: RRF fusion balances keyword precision (BM25) with semantic recall (vector)
- **LlamaParse → BeautifulSoup fallback**: EUR-Lex HTML has complex tables and nested lists; BeautifulSoup is the robustness layer
- **Chroma for MVP**: switch to PGVector for production

### DocumentNode Model (`src/models/document.py`)

Fields: `node_id`, `level` (PART/TITLE/CHAPTER/SECTION/ARTICLE/PARAGRAPH/POINT), `part`, `title`, `chapter`, `section`, `article`, `text`, `metadata` (includes `prev_node_id`, `next_node_id`, `child_node_ids`)

### API

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/query` | `{"query": "..."}` → `{answer, sources, trace_id}` |
| POST | `/api/ingest` | Triggers ingestion pipeline |

## Observability

- Phase 1: structured logging (query, retrieved nodes, LLM tokens, response)
- Phase 2 (deferred): LlamaIndex evaluation modules (faithfulness, context relevance, answer correctness)

## Memory
- Every time you identify and fix some issues worth remembering for future iterations, save it in a `lesson_learnt.md` file. If the file does not exist, do create it. 
- Every time we complete some significant task, update make notes in `WORKLOG.md`. If the file doesn't exist, create it. 
- If we decide to add something to the pipeline (e.g. future improvements, features to add etc.), do update the `WORKLOG.md` file with it. This file should be our project memory, and it should facilitate you from picking up the state of the project every time a new instance of Claude Code is launched. 