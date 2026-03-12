
## Project Overview

A Retrieval-Augmented Generation (RAG) system tailored for the EU Capital Requirements Regulation (CRR – Regulation (EU) No 575/2013, latest consolidated version).  
It ingests structured HTML from EUR-Lex, preserves the legal hierarchy (PART → TITLE → CHAPTER → SECTION → ARTICLE → Paragraph → Point), and answers regulatory compliance questions with complete, traceable context using hierarchical node parsing and auto-merging retrieval.

## Context

### System Boundary

The system ingests, indexes, and queries the EU CRR consolidated text from EUR-Lex.  
It does **not** handle other regulations, external data sources, or real-time updates.

### External Actors

| Name     | Type   | Description                                                  |
|----------|--------|--------------------------------------------------------------|
| EndUser  | `user` | Asks compliance questions and receives answers with citations. |

### Information Flows

| From              | To                | Data                              | Protocol              |
|-------------------|-------------------|-----------------------------------|-----------------------|
| EndUser           | QueryEngine       | User query                        | HTTP API              |
| QueryEngine       | VectorStore       | Embeddings + metadata             | Vector Store API      |
| VectorStore       | QueryEngine       | Retrieved nodes                   | Vector Store API      |
| EurLexIngester    | HierarchicalIndexer | Parsed hierarchical nodes     | In-memory / storage   |

## Tech Stack

- Python 3.11+
- **FastAPI** (for API service)
- LlamaIndex (latest stable version)
- LlamaParse (for structured parsing, with fallback)
- Hugging Face **bge-small-en-v1.5** (embeddings)
- Chroma (MVP vector store; PGVector recommended for production)
- OpenAI GPT-4o (synthesis)
- **beautifulsoup4** + **requests** (fallback HTML parsing & downloading)

## Key Design Decisions

- LlamaIndex chosen for strong hierarchical RAG support and ecosystem maturity.
- LlamaParse preferred for parsing EUR-Lex HTML into structured markdown.
- HierarchicalNodeParser to maintain legal hierarchy with rich metadata.
- AutoMergingRetriever to automatically include full parent context (Article/Section) when sub-points are retrieved.
- Hybrid retrieval via **BM25Retriever** + **QueryFusionRetriever** (Reciprocal Rank Fusion) for improved precision and recall.
- Chroma selected for fast MVP setup and seamless LlamaIndex integration.
- FastAPI used for a simple, production-ready HTTP API.
- bge-small-en-v1.5 chosen for high-quality, open-source, local embeddings (strong MTEB performance, no API dependency).

## Directory Structure

```
eu-crr-rag/
├── src/
│   ├── ingestion/
│   │   └── eurlex_ingest.py
│   ├── indexing/
│   │   ├── index_builder.py
│   │   └── vector_store.py          # Chroma client wrapper
│   ├── query/
│   │   └── query_engine.py
│   ├── models/
│   │   └── document.py
│   └── pipelines/
│       └── ingest_pipeline.py       # Main ingestion orchestrator (CLI)
├── api/
│   └── main.py                      # FastAPI application
├── requirements.txt
└── README.md
```

## Components

### EurLexIngester

- **Type:** Subsystem
- **File:** `src/ingestion/eurlex_ingest.py`
- **Purpose:** Downloads the latest consolidated CRR HTML from the official EUR-Lex URL.  
  Attempts parsing with LlamaParse (markdown output). Falls back to BeautifulSoup + custom hierarchical extraction if LlamaParse fails to correctly handle tables, nested lists or legal formatting.
- **Dependencies:** requests, llama_parse, beautifulsoup4

### HierarchicalIndexer

- **Type:** Subsystem
- **File:** `src/indexing/index_builder.py`
- **Purpose:** Takes parsed documents → applies HierarchicalNodeParser → enriches nodes with metadata (part, title, chapter, section, article, level) → builds vector index (Chroma) and BM25 index → persists both.
- **Dependencies:** EurLexIngester, VectorStore

### QueryEngine

- **Type:** Subsystem
- **File:** `src/query/query_engine.py`
- **Purpose:** Receives user query → executes hybrid retrieval (AutoMergingRetriever on vector index + BM25Retriever via QueryFusionRetriever with Reciprocal Rank Fusion) → synthesizes faithful answer with GPT-4o → returns response including source nodes and citations.
- **Dependencies:** VectorStore, BM25Retriever, QueryFusionRetriever

### VectorStore

- **Type:** DataStore
- **File:** `src/indexing/vector_store.py`
- **Purpose:** Wrapper around Chroma client. Manages storage of embeddings, metadata, and supports vector + hybrid retrieval operations.

## Data Models

### DocumentNode

Represents a node in the CRR document hierarchy.

**Key fields:**

- `node_id`: str (unique within index)
- `level`: str (enum-like: "PART", "TITLE", "CHAPTER", "SECTION", "ARTICLE", "PARAGRAPH", "POINT")
- `part`: str | None
- `title`: str | None
- `chapter`: str | None
- `section`: str | None
- `article`: str | None
- `text`: str (content of this node)
- `metadata`: dict (hierarchy fields + `prev_node_id`, `next_node_id`, `child_node_ids`, etc.)

## API Endpoints (FastAPI)

| Method | Path           | Description                                                                 |
|--------|----------------|-----------------------------------------------------------------------------|
| `POST` | `/api/query`   | Accepts user query → returns answer + retrieved context/sources.           |
| `POST` | `/api/ingest`  | (Optional – MVP may use CLI) Triggers full ingestion pipeline.             |

**Example `/api/query` request body:**

```json
{
  "query": "What are the risk weights for exposures to institutions under Article 114?"
}
```

**Response includes:** `answer`, `sources` (list of node excerpts + metadata), `trace_id`

## Ingestion Orchestration

- Primary entrypoint: `src/pipelines/ingest_pipeline.py` (CLI script)
- Usage example:  
  ```bash
  python -m src.pipelines.ingest_pipeline --url "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:02013R0575-20260101"
  ```
- Optional: exposed via `POST /api/ingest` in FastAPI for automation/CI.

## Glossary

| Term                    | Definition                                                                 |
|-------------------------|----------------------------------------------------------------------------|
| **CRR**                 | Capital Requirements Regulation (EU) No 575/2013                           |
| **EUR-Lex**             | Official portal for EU law documents                                       |
| **AutoMergingRetriever**| LlamaIndex retriever that merges child nodes into parents automatically   |
| **QueryFusionRetriever**| Combines multiple retrievers (vector + BM25) with Reciprocal Rank Fusion  |

## Observability & Evaluation (Deferred to Phase 2)

- Phase 1: Basic structured logging (query, retrieved nodes, LLM tokens, response)
- Phase 2: LlamaIndex evaluation modules (faithfulness, context relevance, answer correctness)

## Research Grounding Summary

- Hierarchical parsing + AutoMergingRetriever: widely adopted for structured legal/regulatory documents.
- Hybrid retrieval (BM25 + vector) via QueryFusionRetriever + RRF: current LlamaIndex best practice for precision/recall balance.
- LlamaParse + BeautifulSoup fallback: ensures robust handling of EUR-Lex HTML formatting.
- FastAPI: standard, type-safe choice for Python-based RAG APIs.

