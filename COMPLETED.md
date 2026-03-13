# COMPLETED — Project History

This file is the chronological record of all completed work on the EU CRR RAG system.
For open tasks and backlog, see `WORKLOG.md`.

---

## 2026-03-12 — Project scaffolding complete

### What was built
Full directory structure and implementation scaffolding from `spec.md`:

| File | Status |
|------|--------|
| `src/models/document.py` | Done — `DocumentNode` dataclass + `NodeLevel` enum |
| `src/ingestion/eurlex_ingest.py` | Done — `EurLexIngester` (LlamaParse + BeautifulSoup fallback) |
| `src/indexing/vector_store.py` | Done — `VectorStore` Qdrant Cloud wrapper (dense + sparse hybrid) |
| `src/indexing/index_builder.py` | Done — `HierarchicalIndexer` (HierarchicalNodeParser + Qdrant indexing) |
| `src/query/query_engine.py` | Done — `QueryEngine` (AutoMergingRetriever + Qdrant native hybrid, optional language filter) |
| `src/pipelines/ingest_pipeline.py` | Done — CLI entrypoint |
| `api/main.py` | Done — FastAPI with `/api/query`, `/api/ingest`, `/health` |
| `requirements.txt` | Done |
| `.env.example` | Done |

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

---

## 2026-03-12 — Migration: Chroma+BM25 → Qdrant+BGE-M3, Multilingual (EN/IT/PL)

### What changed

| File | Change |
|------|--------|
| `requirements.txt` | Replaced `chromadb`, `llama-index-vector-stores-chroma`, `llama-index-retrievers-bm25` with `qdrant-client`, `llama-index-vector-stores-qdrant`, `FlagEmbedding` |
| `.env.example` | Added `QDRANT_URL` and `QDRANT_API_KEY` |
| `src/ingestion/language_config.py` | **NEW** — `LanguageConfig` dataclass + `LANGUAGE_CONFIGS` for EN/IT/PL |
| `src/ingestion/eurlex_ingest.py` | Added `language` + `local_file` params; heading/article classification now language-aware |
| `src/indexing/bge_m3_sparse.py` | **NEW** — module-level BGE-M3 singleton; `sparse_doc_fn` / `sparse_query_fn` for Qdrant hybrid search |
| `src/indexing/vector_store.py` | Rewritten for Qdrant Cloud: dense (1024-dim) + sparse vectors, `enable_hybrid=True` |
| `src/indexing/index_builder.py` | Removed BM25 index; default embed model → `BAAI/bge-m3` |
| `src/query/query_engine.py` | Removed `QueryFusionRetriever` + `BM25Retriever`; replaced with Qdrant native hybrid |
| `src/pipelines/ingest_pipeline.py` | Added `--language` and `--file` CLI args |
| `api/main.py` | `QueryRequest` + `preferred_language`; language-aware ingestion; `_detect_language()` heuristic |

### Architecture after migration
- **Single Qdrant collection** (`eu_crr`) holds all languages; each node has `language` metadata field
- **BGE-M3** (1024-dim dense + sparse) replaces `bge-small-en-v1.5` (384-dim, English-only)
- **Hybrid search** (dense + sparse) handled natively by Qdrant — BM25 pickle removed
- **AutoMergingRetriever** + docstore unchanged
- **Language filter** applied optionally at query time

---

## 2026-03-12 — Phase 2: Test suite implemented

### What was built

| File | Tests | Status |
|------|-------|--------|
| `pytest.ini` | Config + markers (unit / integration / requires_html) | Done |
| `tests/conftest.py` | Shared fixtures: synthetic EUR-Lex HTML (EN/IT/PL), real-HTML file paths with auto-skip | Done |
| `tests/unit/conftest.py` | `sys.modules` stubs for qdrant_client / FlagEmbedding | Done |
| `tests/unit/test_language_config.py` | 21 tests | Done |
| `tests/unit/test_classify_heading.py` | 38 tests | Done |
| `tests/unit/test_eurlex_ingest.py` | 28 tests | Done |
| `tests/unit/test_document_node.py` | 16 tests | Done |
| `tests/unit/test_api_endpoints.py` | 18 tests | Done |
| `tests/integration/conftest.py` | Session-scoped fixtures; auto-skips when env vars missing | Done |
| `tests/integration/test_vector_store.py` | 8 tests | Done |
| `tests/integration/test_ingest_pipeline.py` | 7 tests | Done |
| `tests/integration/test_query_engine.py` | 12 tests | Done |

**Unit test result: 121/121 passed.**

---

## 2026-03-12 — Major Redesign: Legal-structure-aware parsing + cross-reference expansion

### Motivation
The old system chunked text by token count (128/512/2048) using `HierarchicalNodeParser`, ignoring legal structure. This caused articles to be split mid-sentence, tables flattened, annexes unidentifiable, and cross-references never followed. The EUR-Lex HTML has a well-defined DOM (`<div id="art_N">`, parent IDs encoding full hierarchy) enabling a precise legal-structure parser.

### Changes

| File | Change |
|------|--------|
| `src/models/document.py` | Added `ANNEX` level; removed `PARAGRAPH`/`POINT`; added `article_title`, `annex_id`, `annex_title`, `referenced_articles`, `referenced_external`, `has_table`, `has_formula` fields |
| `src/ingestion/eurlex_ingest.py` | Full rewrite: DOM-based parser using `art_N`/`anx_N` div IDs; `_extract_hierarchy()`, `_extract_structured_text()`, `_table_to_markdown()`, `_extract_cross_references()` |
| `src/indexing/index_builder.py` | Removed `HierarchicalNodeParser`, `SimpleDocumentStore`, docstore persistence |
| `src/query/query_engine.py` | Removed `AutoMergingRetriever`; added `_expand_cross_references()` method |
| `src/indexing/bge_m3_sparse.py` | Thread-safe double-checked locking singleton |
| `src/indexing/vector_store.py` | `item_count` try/except; `reset()` None guard |

### Follow-up: language-aware cross-reference extraction
`_extract_cross_references` uses `self._lang_cfg.article_keyword` for the article pattern (covers EN/IT/PL automatically) and `_EXTERNAL_REF_PATTERNS` dict with per-language directive/regulation regexes.

### Smoke test results (2026-03-12)

| File | Total docs | Articles | Annexes |
|------|-----------|----------|---------|
| `crr_raw_en.html` | 745 | 741 | 4 |
| `crr_raw_ita.html` | 745 | 741 | 4 |

### Architecture after redesign
- **Each Document = one Article or Annex section** (no sub-article chunking)
- **Hierarchy from DOM IDs**: `prt_III.tis_I.cpt_1.sct_1` → `{'part':'III','title':'I','chapter':'1','section':'1'}`
- **Structured text**: numbered paragraphs preserved, grid-list points indented, tables as Markdown, formulas as `[FORMULA]`
- **Cross-reference expansion**: at query time, `referenced_articles` metadata drives follow-up retrieval
- **Removed**: `HierarchicalNodeParser`, `AutoMergingRetriever`, docstore JSON persistence

---

## 2026-03-12 — Formula enrichment via LlamaParse (implemented, not yet ingested)

### Problem
EUR-Lex embeds mathematical formulas as base64 PNG data URIs inside `<img>` tags. The BS4 parser replaced these with `[FORMULA]` placeholders, making the LLM unable to reason about formula content.

### Solution: hybrid BS4 + LlamaParse formula pass

| Step | What happens |
|------|-------------|
| BS4 parse | Handles all legal structure (hierarchy, cross-refs, tables) as before |
| Formula numbering | `_extract_structured_text` emits `[FORMULA_0]`, `[FORMULA_1]` … and records base64 URIs |
| LlamaParse enrichment | For articles with formulas AND `use_llama_parse=True`, article HTML sent to LlamaParse |
| LaTeX substitution | `_extract_latex_from_markdown` finds `$$...$$`, `\[...\]`, `$...$` and substitutes into placeholders |
| Fallback | If LlamaParse fails or returns fewer formulas than expected, `[FORMULA_N]` is preserved |

### Cost estimate
- Formula-containing articles ≈ 50–100 (minority)
- Each triggers one LlamaParse API call (per article, not per formula)
- Remaining ~650 articles use BS4 only (no LlamaParse cost)

---

## 2026-03-12 — Code review fixes (P0 + housekeeping)

### P0 fixes applied
| # | File | Fix |
|---|------|-----|
| 1 | `eurlex_ingest.py` | `_classify_heading` no-op — moot after DOM-based redesign |
| 2 | `api/main.py` | `_detect_language` — added docstring warning about false-positives |
| 3 | `query_engine.py` | Language-filtered engine cached in `_engine_cache` dict |

### P1 — Already fixed in earlier session
- `vector_store.py` `item_count` try/except ✅
- `vector_store.py` `reset()` None guard ✅
- `bge_m3_sparse.py` thread-safe singleton ✅

### P2 — Test quality (all resolved)
| # | File | Fix |
|---|------|-----|
| 7 | `test_api_endpoints.py` | Fixed `test_index_loaded_true_when_loaded` to use `loaded_client` fixture |
| 8 | `tests/unit/conftest.py` | Added prominent warning: run unit and integration suites in separate invocations |
| 9 | `test_vector_store.py` | Merged `TestVectorStoreReset` into single test |
| 10 | `test_eurlex_ingest.py` | Relaxed hardcoded count to `>= 740` |

---

## 2026-03-13 — Reranker, thread-safety, CORS, env validation, OOM fix

### What was done

#### Reranker added
- `RETRIEVAL_TOP_K` raised from 6 → 12 (wider first-stage candidate set)
- `FlagEmbeddingReranker(top_n=6, model="BAAI/bge-reranker-v2-m3")` added as second postprocessor
- Reranker loaded once in `load()`, reused across all engine/cache instances
- `use_reranker=True` flag allows tests to opt out
- New dep: `llama-index-postprocessor-flag-embedding-reranker`

#### OOM fix: BGEm3Embedding singleton wrapper
- `HuggingFaceEmbedding(bge-m3)` in `_configure_settings` was loading a second 570MB copy
- Replaced with `BGEm3Embedding` — thin `BaseEmbedding` subclass delegating to `_get_model()` singleton
- Only one copy of BGE-M3 now lives in memory

#### Thread-safety + API hardening
- `_engine_cache`: double-checked locking prevents duplicate builds
- `TokenCountingHandler`: per-request instead of shared singleton
- `QueryEngine.is_loaded()`: public method replaces private attr accesses
- Env var validation at startup
- CORS middleware added

---

## 2026-03-13 — Re-ingestion complete (Colab T4 GPU)

### Results
| Metric | Value |
|--------|-------|
| EN documents ingested | 745 |
| IT documents ingested | 745 |
| Total Qdrant items | ~1490 |
| Ingest time (T4 GPU) | ~5 min total (vs ~11h CPU) |
| Smoke test | PASS |

### Notes
- Created `colab_ingest.ipynb` — GPU-accelerated ingestion notebook
- Fixed Colab-specific issues (CUDA detection, constructor args, metadata access)
- Pushed repo to GitHub: https://github.com/apolitano20/EU-CRR-RAG

---

## 2026-03-13 — RAG review fixes + `sparse_query_fn` final fix

### RAG review (from `rag_review_qa_for_claude_code.md`)
6 actionable issues identified and addressed:

| # | Issue | Fix | Status |
|---|-------|-----|--------|
| 1 | `SentenceSplitter` silently re-chunks article-level docs | `transformations=[]` in `from_documents()` | ✅ |
| 2 | No custom prompt — LLM can hallucinate | Custom `_LEGAL_QA_TEMPLATE`: cite articles, say "I don't know" | ✅ |
| 3 | No similarity cutoff (`cutoff=0.0`) | `SIMILARITY_CUTOFF = 0.3` | ✅ |
| 4 | `SIMILARITY_TOP_K = 12` too broad | Reduced to 6 | ✅ |
| 5 | No query normalisation for article refs | `_normalise_query()` — "art. 92" → "Article 92" | ✅ |
| 6 | Reranker absent | ✅ Done in separate commit (opt-in via `USE_RERANKER`) | ✅ |

### `sparse_query_fn` — final fix
Must return **batch format** `(list[list[int]], list[list[float]])` — LlamaIndex indexes `[0]` into outer list.

### Qdrant payload indexes
Added `_ensure_payload_indexes()` — keyword indexes for `language`, `article`, and `level`.

### Cross-lingual fallback
`QueryEngine.query()` retries without language filter if zero results above similarity cutoff.

---

## 2026-03-13 — BGEm3Embedding segfault on Windows + reranker made opt-in

### Problem
`BGEm3Embedding` caused `SIGSEGV` on Windows when called from LlamaIndex query pipeline.

### Root cause
OOM only triggered when reranker (~550MB) loaded **simultaneously** with BGE-M3 (~570MB). On GPU machines this is fine.

### Fixes applied
- Reranker made opt-in via `USE_RERANKER` env var (default `false`)
- **Reverted `BGEm3Embedding` → `HuggingFaceEmbedding`** in query path
- Singleton wrapper removed from query path; `HuggingFaceEmbedding("BAAI/bge-m3")` used directly

### Current embedding strategy
- **Indexing**: `bge_m3_sparse.py` singleton for sparse vectors — unchanged
- **Query-time dense**: `HuggingFaceEmbedding("BAAI/bge-m3")` — stable on Windows
- **Reranker**: disabled by default; enable with `USE_RERANKER=true`

---

## 2026-03-13 — GitHub Actions CI

### What was built

| File | Purpose |
|------|---------|
| `.github/workflows/ci.yml` | Unit tests on every push/PR (Python 3.11 + 3.12); dummy env vars for lifespan validation; manual trigger enabled |
| `.github/workflows/integration.yml` | Integration tests — manual trigger only; requires GitHub repo secrets; runs per-file (BGE-M3 OOM workaround) |

### CI fixes applied
- `273100f` — `workflow_dispatch` trigger for manual run button
- `d5dd7d0` — Dummy env vars for unit test job
- `053e6b1` — Separated integration tests into dedicated workflow

### Colab notebook updates (`703435a`)
- Renamed to "GPU Dev Environment"
- `pip install` now reads from `requirements.txt`
- Added integration test cells and ad-hoc query cell

---

## 2026-03-13 — Full Code Review + RAG Checklist Gap Analysis

Reviewed entire codebase (source, tests, config). **58 issues found**: 2 critical, 4 high, 28 medium, 24 low.

All critical/high issues resolved:
- Thread-safe `_engine_cache` with double-checked locking
- Per-request `TokenCountingHandler`
- `reset()` None guard in vector_store
- `is_loaded()` public method (removed private attr access)
- Cached cross-ref retriever (performance)
- Polish diacritics regex (re.UNICODE)

See `cross-reference-audit.md` for the full cross-reference capability audit.

---

## 2026-03-13 — Query engine uplifts (no re-ingest required)

### What was done

Six improvements implemented without re-ingestion or eval dependency:

| # | Change | Location |
|---|--------|----------|
| 1 | **CRR abbreviation expansion** — `_ABBREV_MAP` (17 terms: CET1, AT1, T2, LCR, NSFR, MREL, RWA, IRB, CVA, CCR, EAD, LGD, ECAI, SFT, CCP, QCCP, EBA) expanded inline in `_normalise_query()` before embedding | `query_engine.py` |
| 2 | **Per-stage latency logging** — Split `engine.query()` into `engine.retrieve()` + `engine.synthesize()` using `QueryBundle`; logs `t_retrieval`, `t_expansion`, `t_synthesis` per query | `query_engine.py` |
| 3 | **Upper-bound version pins** — Added `<major+1` upper bounds to all 17 unpinned packages in `requirements.txt` | `requirements.txt` |
| 4 | **Multi-hop cross-reference expansion (Gap 4)** — `_expand_cross_references()` now accepts `depth` param; recursive second-hop expansion follows refs-of-refs | `query_engine.py` |
| 5 | **Task-type routing** — `_detect_direct_article_lookup()` detects "Explain Article 92" / "What does Article 92 say?" queries and routes to `_direct_article_retrieve()` with exact metadata filter (`article=N`), bypassing vector ranking | `query_engine.py` |
| 6 | **Gap 6 (hierarchical expansion) descoped** — Confirmed no chapter/section-level documents exist in corpus; expanding to nearby articles would add noise | WORKLOG only |

### Test results
- 164/164 unit tests pass (43 new tests added for abbreviation expansion and direct lookup detection)
- Checklist score: 8/13 → 9/13 (latency/cost per stage now met)

---

## 2026-03-13 — Structured answer template

### What was done
Enforced structured answer format in LLM prompt (`f9aad79`):

```
**Direct Answer** — 1–3 sentence concise answer
**Key Provisions** — Bullet points referencing specific Articles
**Conditions, Exceptions & Definitions** — Qualifications and carve-outs
**Article References** — Comma-separated list of all cited Articles
```

This was item #9 from the RAG checklist gap analysis (checklist #29: domain-specific answer structure).
