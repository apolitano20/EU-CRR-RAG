# WORKLOG

## 2026-03-12 — Project scaffolding complete

### What was built
Full directory structure and implementation scaffolding from `spec.md`:

| File | Status |
|------|--------|
| `src/models/document.py` | Done — `DocumentNode` dataclass + `NodeLevel` enum |
| `src/ingestion/eurlex_ingest.py` | Done — `EurLexIngester` (LlamaParse + BeautifulSoup fallback) |
| `src/indexing/vector_store.py` | Done — `VectorStore` Chroma wrapper |
| `src/indexing/index_builder.py` | Done — `HierarchicalIndexer` (HierarchicalNodeParser + BM25 persistence) |
| `src/query/query_engine.py` | Done — `QueryEngine` (AutoMergingRetriever + BM25 + QueryFusionRetriever RRF) |
| `src/pipelines/ingest_pipeline.py` | Done — CLI entrypoint |
| `api/main.py` | Done — FastAPI with `/api/query`, `/api/ingest`, `/health` |
| `requirements.txt` | Done |
| `.env.example` | Done |

### Next steps
1. Install dependencies: `pip install -r requirements.txt`
2. Copy `.env.example` → `.env` and fill in `OPENAI_API_KEY` (and optionally `LLAMA_CLOUD_API_KEY`)
3. Run ingestion: `python -m src.pipelines.ingest_pipeline --reset`
4. Start API: `uvicorn api.main:app --reload`
5. Test query: `POST /api/query {"query": "What are the risk weights for exposures to institutions under Article 114?"}`

### Known gaps / future work
- No tests yet (Phase 2)
- LlamaIndex evaluation modules (faithfulness, context relevance) deferred to Phase 2
- Production vector store: swap Chroma → PGVector
- EUR-Lex HTML structure may need tuning in the BeautifulSoup fallback (`_classify_heading` regexes)
- Rate limits and retry logic for LlamaParse not yet implemented

---

## 2026-03-12 — Bug fixes: first working end-to-end version

### Bugs fixed

| # | Bug | Fix |
|---|-----|-----|
| 1 | `AutoMergingRetriever` had no docstore — parent-context merging was broken | `_build_vector_index()` now creates `SimpleDocumentStore` with ALL nodes, passes it in `StorageContext`, persists to `./docstore/`, and `load()` reloads it via `persist_dir` |
| 2 | `Settings.llm` silently reset to `None` — LLM synthesis broke | Moved `_configure_settings()` to run **after** `indexer.load()` in `QueryEngine.load()` |
| 3 | BeautifulSoup parser walked generic `<h1>/<h2>` tags — EUR-Lex uses CSS classes (`sti-art`, `ti-art`, `ti-section-*`, `normal`) so zero articles were parsed | Rewrote `_parse_with_beautifulsoup()` as a class-aware stateful DOM walker |

### Additional changes
- `.gitignore` added (covers `chroma_db/`, `docstore/`, `bm25_index.pkl`, `.env`, Python/IDE/OS artifacts)
- `use_llama_parse` default flipped to `False` — BeautifulSoup is now primary; LlamaParse is opt-in
- `_reset_lower()` helper clears lower-hierarchy metadata when a higher-level heading is encountered
- `_make_document()` helper centralises Document construction

### Verification steps
```bash
pip install -r requirements.txt
python -m src.pipelines.ingest_pipeline --reset   # check logs: article count > 0
uvicorn api.main:app --reload
curl http://localhost:8000/health                  # index_loaded should be true
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the risk weights for exposures to institutions under Article 114?"}'
```

### Known gaps / future work
- No tests yet (Phase 2)
- LlamaIndex evaluation modules (faithfulness, context relevance) deferred to Phase 2
- Production vector store: swap Chroma → PGVector
- Rate limits and retry logic for LlamaParse not yet implemented
- Verify EUR-Lex class names against live HTML after first ingestion run
