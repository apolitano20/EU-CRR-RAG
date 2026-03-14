# EU CRR RAG

A Retrieval-Augmented Generation system for the EU Capital Requirements Regulation (CRR — Regulation (EU) No 575/2013). Ask compliance questions in English, Italian, or Polish; get answers with article-level citations.

## What it does

1. Downloads (or reads from a local file) the consolidated CRR HTML from EUR-Lex
2. Parses and indexes it preserving the full legal hierarchy: PART → TITLE → CHAPTER → SECTION → ARTICLE
3. Supports multilingual ingestion (EN / IT / PL) in a single shared index with per-node language metadata
4. Answers regulatory questions via a FastAPI HTTP API, citing the relevant articles
5. Serves a two-panel Next.js UI: chat Q&A on the left, article viewer on the right

## Tech stack

| Layer | Technology |
|---|---|
| Parsing | BeautifulSoup4 (primary), LlamaParse (opt-in) |
| Embeddings | `BAAI/bge-m3` (HuggingFace, local — multilingual, dense + sparse) |
| Vector store | Qdrant Cloud (native hybrid dense + sparse search) |
| Reranker | `BAAI/bge-reranker-v2-m3` (opt-in via `USE_RERANKER=true`) |
| Synthesis | OpenAI GPT-4o |
| API | FastAPI |
| Frontend | Next.js 16 + React 19 + TypeScript + Tailwind CSS |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Fill in OPENAI_API_KEY, QDRANT_URL, and QDRANT_API_KEY in .env
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

Ingestion produces ~745 article-level documents per language (741 articles + 4 annexes).

### GPU-accelerated ingestion (recommended)

Open `colab_ingest.ipynb` in Google Colab (T4 runtime). Ingestion takes ~5 min on GPU vs. ~11 hours on CPU.

### CLI reference

| Flag | Default | Description |
|---|---|---|
| `--language` | `en` | Language to ingest: `en`, `it`, or `pl` |
| `--file` | — | Path to a local HTML file (skips network download) |
| `--url` | auto-derived | EUR-Lex URL (derived from `--language` if not set) |
| `--reset` | off | Drop and recreate the Qdrant collection before ingesting |

## Running locally

### Option 1 — Docker Compose (recommended)

```bash
docker compose up
```

API at `http://localhost:8000`, frontend at `http://localhost:3000`.

### Option 2 — Manual

```bash
# Backend
uvicorn api.main:app --reload --port 8080

# Frontend (separate terminal)
cd frontend
npm install
npm run dev
```

Frontend expects the API at `NEXT_PUBLIC_API_URL` (defaults to `http://localhost:8080`).

### Windows quick-launch

```bat
launch.bat
```

Activates `.venv`, kills stale processes, and opens both panels in the browser.

## API

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check; reports whether the index is loaded |
| `POST` | `/api/query` | Ask a compliance question |
| `GET` | `/api/article/{article_id}` | Fetch full article text and metadata for the viewer |
| `POST` | `/api/ingest` | Trigger ingestion programmatically |

### Example queries

```bash
# English query
curl -X POST http://localhost:8080/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the minimum CET1 requirements?", "preferred_language": "en"}'

# Italian query (language auto-detected from accent characters)
curl -X POST http://localhost:8080/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Quali sono i requisiti minimi CET1?"}'

# Fetch a specific article
curl "http://localhost:8080/api/article/92?language=en"
```

Response shape:
```json
{
  "answer": "...",
  "language": "en",
  "sources": [
    {
      "text": "...",
      "score": 0.87,
      "metadata": {"article": "92", "part": "TWO", "language": "en", ...},
      "expanded": false
    }
  ],
  "trace_id": "..."
}
```

If `preferred_language` is omitted, the API detects Polish (from `ąćęłńóśźż…`) and Italian (from `àèéìíîòóùú…`) automatically. Undetected queries retrieve across all languages.

### Structured answer format

Answers follow a consistent four-part structure:

```
**Direct Answer** — 1–3 sentence concise answer
**Key Provisions** — Bullet points referencing specific Articles
**Conditions, Exceptions & Definitions** — Qualifications and carve-outs
**Article References** — Comma-separated list of all cited Articles
```

## Architecture

```
EurLexIngester          downloads/reads EUR-Lex HTML
      ↓                 DOM-based parser: preserves legal hierarchy, tables, cross-refs
                        stamps language + hierarchy metadata on every node
HierarchicalIndexer     article-level documents → Qdrant hybrid vector index (BGE-M3 dense + sparse)
QueryEngine             Qdrant native hybrid retrieval → cross-ref expansion → optional reranker
                        → optional MetadataFilter(language) → GPT-4o synthesis
      ↓
FastAPI                 POST /api/query  →  {answer, language, sources, trace_id}
                        GET  /api/article/{id}  →  full article text + metadata
      ↓
Next.js frontend        Two-panel UI: chat Q&A + article viewer
```

**Hybrid retrieval:** BGE-M3 dense semantic recall + BGE-M3 sparse (lexical) precision, fused natively by Qdrant.

**Cross-reference expansion:** after retrieval, `referenced_articles` metadata drives follow-up fetches of cited articles. Supports multi-hop (depth=2 by default). External regulation refs (e.g. "Article 10 of Regulation (EU) No 1093/2010") are excluded from expansion.

**Direct article lookup:** queries mentioning exactly one article (e.g. "Explain Article 92") bypass vector ranking and use an exact metadata filter.

**Multilingual:** a single Qdrant collection holds all languages. Each node carries a `language` metadata field; queries can filter by language or retrieve cross-language.

## Directory structure

```
eu-crr-rag/
├── api/
│   └── main.py                    # FastAPI app
├── frontend/                      # Next.js 16 + React 19 UI
│   └── src/
│       ├── components/            # Chat panel, article viewer, source chips
│       └── lib/                   # API client, types
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
├── scripts/
│   └── fix_cross_refs.py          # Qdrant metadata patch utilities
├── tests/
│   ├── unit/                      # 169 unit tests (no external deps)
│   └── integration/               # Integration tests (require live Qdrant + OpenAI)
├── colab_ingest.ipynb             # GPU-accelerated ingestion notebook
├── docker-compose.yml
├── Dockerfile
├── launch.bat                     # Windows quick-launch script
├── .env.example
└── requirements.txt
```

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | Used for GPT-4o synthesis |
| `QDRANT_URL` | Yes | Qdrant Cloud cluster URL (e.g. `https://xyz.qdrant.io`) |
| `QDRANT_API_KEY` | Yes | Qdrant Cloud API key |
| `LLAMA_CLOUD_API_KEY` | No | Enables LlamaParse as primary HTML parser |
| `USE_RERANKER` | No | Set to `true` to enable BGE-Reranker-v2-m3 (~550 MB extra RAM; recommended on GPU) |
| `NEXT_PUBLIC_API_URL` | No | Frontend API base URL (default: `http://localhost:8080`) |

## Tests

```bash
# Unit tests (no external services required)
python -m pytest tests/unit/ -v

# Integration tests (requires live Qdrant + OpenAI keys)
python -m pytest tests/integration/ -v
```

CI runs unit tests on every push via GitHub Actions (`.github/workflows/ci.yml`).
