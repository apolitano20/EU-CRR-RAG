# COMPLETED — Project History

This file is the chronological record of all completed work on the EU CRR RAG system.
For open tasks and backlog, see `WORKLOG.md`.

---

## 2026-03-17 — 4 medium/low-priority code-only fixes; 217 unit tests green

### Summary
Closed 4 Codex V2 findings that required no re-ingestion. Added 25 new unit tests.

### Fix 1 — Direct article lookup: external ref misclassification (`query_engine.py`)
- Added `_EXTERNAL_DIRECTIVE_RE` regex to strip `Article N of Directive/Regulation ...`
  citations before counting CRR article references.
- Added `_ARTICLE_COORD_RE` regex to detect coordinated bare-number runs
  (`Article 92 and 93`, `Article 92, 93 and 94`) and return `None` (multi-article intent).
- `_detect_direct_article_lookup()` now correctly returns `None` for external refs
  and coordinated phrasing, and returns the single CRR article when only one remains
  after stripping externals.

### Fix 2 — Cross-ref expansion non-deterministic cap (`query_engine.py`)
- Added `_ref_sort_key()` helper: sorts article numbers numerically with alpha suffix
  tie-break (`"92" < "92a" < "92aa"`, non-numeric falls to front).
- `_expand_cross_references()` now sorts candidates with `_ref_sort_key` before slicing,
  making expansion order stable across Python runs.
- A failed/missing article no longer consumes a cap slot: the loop now checks
  `len(expanded) >= limit` at the top of each iteration and continues to the next
  candidate on exception, so `limit` successful expansions are always attempted.

### Fix 3 — Stale PromptHelper cache (`query_engine.py`)
- `QueryEngine._configure_settings()` now resets `Settings._prompt_helper = None`
  immediately after setting the LLM, forcing LlamaIndex to rebuild the PromptHelper
  from GPT-4o's 128k context window on next use.

### Fix 4 — `_configure_settings()` global Settings mutation (`index_builder.py`)
- Added `_settings_scope()` context manager (snapshot + try/finally restore).
- `HierarchicalIndexer.build()` and `.load()` now wrap their bodies with
  `_settings_scope()`, so the indexer's Settings mutations (`llm=None`,
  `transformations=[]`, etc.) are unwound before returning.

### Test additions
- `test_query_normalise.py`: 8 new cases covering external directive refs and
  coordinated bare-number phrasing.
- `test_query_engine_unit.py`: `TestRefSortKey` (5 cases), `TestExpandCrossReferences`
  (4 cases), `test_configure_settings_resets_prompt_helper`.
- `test_index_builder.py`: `TestSettingsScope` (3 cases) covering normal exit,
  post-`_configure_settings` restoration, and exception path.

**Test count: 217 unit tests, all green.**

---

## 2026-03-17 — Qdrant duplicate accumulation fixed; clean 1490-item index

### Problem
After the `Settings.transformations = []` fix, Qdrant reported 2151 items instead of the
expected 1490. Diagnosed via `scripts/diagnose_qdrant.py`: 337 duplicate `node_id`s confirmed
root cause H1 — LlamaIndex generates a random UUID per `Document` on every run. Without
`--reset`, Qdrant accumulated new UUIDs on top of old ones (no overwrite, since point ID ≠ old ID).

### Fix
`src/ingestion/eurlex_ingest.py` — both `_process_article_div()` and `_process_annex_div()`
now pass `id_=_node_id_to_uuid(node.node_id)` to `Document(...)`. `_node_id_to_uuid()` uses
`uuid.uuid5` with a fixed namespace to produce a **stable, valid UUID** from the human-readable
node_id (e.g. `art_92_en` → `c15658fe-a835-5a73-99d8-af3390127e2f`). Same article always maps
to the same Qdrant point ID → upserts are now idempotent across re-runs without `--reset`.

Note: `id_=node.node_id` (raw string) was tried first but Qdrant rejected it with HTTP 400:
`value art_1_en is not a valid point ID, valid values are either an unsigned integer or a UUID`.

### Tools added
- `scripts/diagnose_qdrant.py` — audits Qdrant payloads: item counts per language,
  duplicate node_ids, annex sub-item breakdown, optional parser ground-truth comparison.
- Cell 8b added to `colab_ingest.ipynb` — runs the diagnose script after each ingest.

### Housekeeping
- Deleted diverged `master` branch (local + remote); all work now on `main`.
- Colab `git pull` was not picking up fixes because pushes went to `master` while Colab
  tracks `main` (GitHub default). Fixed by `git push origin master:main` then branch cleanup.

### Verified
- Colab T4 re-ingest: `--reset` EN (745) + IT (745) = **1490 items, PASS**
- Smoke test: Article 92 query returns correct 4.5%/6%/8%/3% ratios with Article 92 as top-ranked source.

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

---

## 2026-03-13 — Performance and retrieval fixes (launch day)

### Performance fixes
| Issue | Fix | File |
|-------|-----|------|
| API hangs on query (OOM) | Removed double BGE-M3 loading: `HuggingFaceEmbedding` was loading a second 570MB copy alongside singleton. Switched back to `BGEm3Embedding` wrapper in `_configure_settings()` | `query_engine.py`, `bge_m3_sparse.py` |
| Frontend dev mode slow | Changed `frontend/Dockerfile` from `npm run dev` to multi-stage production build (`next build` + `node server.js`) | `frontend/Dockerfile` |
| Port conflicts on relaunch | Created `launch.bat`: cleans stale `.next` cache, activates `.venv`, forces ports 8080 (API) + 3001 (frontend) | `launch.bat` |

### Retrieval improvements
| Issue | Fix | File |
|-------|-----|------|
| Direct article lookup too strict | Broadened `_detect_direct_article_lookup()` to trigger on ANY query mentioning exactly one article ("What are the requirements of Article 73?" now matches) instead of only strict patterns | `query_engine.py` |
| HYBRID mode incompatible with strict filters | Changed `_direct_article_retrieve()` from `VectorStoreQueryMode.HYBRID` to `DEFAULT` (dense-only); sparse pass was returning zero results with article="N" filter | `query_engine.py` |
| No fallback for empty direct lookup | Added fallback: if direct article lookup returns 0 nodes, retry with semantic retrieval | `query_engine.py` |

### Outstanding issue
Query "What are the requirements of Article 73?" still returns "insufficient context". Root cause under investigation — Article 73 exists in HTML but may have metadata mismatch or be filtered upstream. Next step: check API logs for "Direct lookup returned no nodes" warning.

---

## 2026-03-14 — UI/UX quick wins (session 2)

### Changes

| # | Change | Files |
|---|--------|-------|
| 1 | **"Sources" → "Cross-References"** — Renamed label in chip list and clipboard markdown | `SourceChipList.tsx`, `AnswerCard.tsx` |
| 2 | **Language filter for cross-references** — Backend returns `language` in `QueryResponse`; frontend threads it through `AnswerCard` → `SourceChipList` and filters sources by query language before deduplication | `api/main.py`, `types.ts`, `ChatPanel.tsx`, `AnswerCard.tsx`, `SourceChipList.tsx` |
| 3 | **Inline list item indentation** — `ProvisionText.tsx` now splits inline `(a)/(b)/(i)` items (preceded by `;` or `:`) onto separate lines and renders with nested indentation: lettered items (`ml-5`) under numbered paragraphs, roman numeral sub-items (`ml-10`) nested deeper | `ProvisionText.tsx` |
| 4 | **External regulation refs no longer clickable** — `ProvisionText.tsx` detects when "Article N" is followed by "of Regulation/Directive/Decision/..." and renders as plain text. Handles ranges like "Articles 10 to 14 of Regulation (EU) No 1093/2010" | `ProvisionText.tsx` |
| 5 | **External regulation refs removed from cross-ref metadata** — Fixed `_extract_cross_references()` in ingestion to exclude "Article N of Regulation/Directive/..." patterns (EN/IT/PL). Patched 942 existing Qdrant points via `scripts/fix_cross_refs.py` (payload update, no re-embedding). Cross-reference chips no longer show articles belonging to external regulations. | `eurlex_ingest.py`, `scripts/fix_cross_refs.py` |

### UI/UX quick win scorecard
- Chat history — ✅ (done prior session)
- Article text formatting — ✅ (done prior session, improved this session)
- "Sources" → "Cross-References" — ✅
- Language filter for cross-references — ✅
- Source relevance — ✅ (confirmed fine)
- Citation order — ✅ (confirmed fine)
- External regulation refs scoped — ✅ (frontend + backend metadata)

---

## 2026-03-15 — Codex review: QueryEngine correctness + embedding consolidation

### Findings addressed (from Codex targeted review)

| # | Finding | Severity | Fix |
|---|---------|----------|-----|
| 1 | **Expanded nodes excluded from LLM synthesis** — `_expand_cross_references()` results were appended to `sources` (UI only) but never passed to `engine.synthesize()`; the LLM had no visibility into referenced articles | High | Merged `deduped_expanded` into `all_nodes_for_synthesis`; dedup by `node_id` prevents double-context; synthesis now called with `source_nodes + deduped_expanded` |
| 2 | **`Settings.callback_manager` thread-unsafe global mutation** — `query()` overwrote the module-level `Settings` singleton per request; in practice the token counter was also disconnected from already-constructed engine/synthesizer objects (LlamaIndex captures callbacks at construction) | High | Removed `TokenCountingHandler` and `CallbackManager` mutation entirely; updated log format accordingly |
| 3 | **DEFAULT vs HYBRID inconsistency** — `_direct_article_retrieve()` and `get_article()` used hard-coded `DEFAULT`; `_expand_cross_references()` used hard-coded `HYBRID` — three methods, two contradictory rules | Medium | Introduced `_retrieve_with_filters()` helper: tries `HYBRID` first, falls back to `DEFAULT` on empty result or exception; all three methods now use this single path |
| 4 | **Embedding backend inconsistency** — `HierarchicalIndexer._configure_settings()` used `HuggingFaceEmbedding(bge-m3)` (sentence-transformers backend) while query path used `BGEm3Embedding` (FlagEmbedding backend); one BGE-M3 singleton, two wrappers | Low | `index_builder.py` now imports and uses `BGEm3Embedding` directly; `embed_model_name` constructor parameter removed; both indexing and querying use the same class and singleton |

### Files changed
- `src/query/query_engine.py` — removed `CallbackManager`/`TokenCountingHandler` imports and usage; synthesis now over `all_nodes_for_synthesis`; `_retrieve_with_filters()` helper added; `_direct_article_retrieve()`, `get_article()`, `_expand_cross_references()` updated
- `src/indexing/index_builder.py` — replaced `HuggingFaceEmbedding` with `BGEm3Embedding`; removed `embed_model_name` parameter
- `tests/unit/test_query_engine_unit.py` — **NEW** — 8 unit tests: `_retrieve_with_filters` HYBRID→DEFAULT fallback, exception fallback, short-circuit on HYBRID success; synthesis includes expanded nodes, deduplication, `expanded` flag in sources

### Test results
177/177 unit tests pass.

---

## 2026-03-15 — Codex review P1 fixes

### Findings addressed

| # | Finding | Fix |
|---|---------|-----|
| P1 | `get_article()` used `HYBRID` mode with exact metadata filter — silently returned zero results for articles that exist, breaking source-chip navigation | Switched to `DEFAULT` (dense-only), matching the fix already applied to `_direct_article_retrieve()` |
| P1 | `frontend/public/` directory missing from repo — `docker build` COPY step failed on clean checkout | Added `frontend/public/.gitkeep` |
| P2 | No committed example for `NEXT_PUBLIC_API_URL` — fresh checkout used wrong port (8000 vs 8080) for local dev | Added `frontend/.env.local.example` documenting `NEXT_PUBLIC_API_URL=http://localhost:8080` |

### Remaining P2s (open in WORKLOG)
- Document viewer only linkifies English article refs — Italian/Polish cross-references not clickable (covered by Italian parity audit item)
- `launch.bat` port comment clarified via `.env.local.example`

---

## 2026-03-15 — Reverse reference lookup + article title precision investigation

### Reverse reference lookup (`GET /api/article/{id}/citing`)

New query-time capability: "which articles cite Article 92?" — implemented with no re-ingest.

| File | Change |
|------|--------|
| `src/indexing/vector_store.py` | Added `scroll_payloads(language)` — Qdrant native scroll with optional language filter; returns all document payloads without vectors |
| `src/query/query_engine.py` | Added `get_citing_articles(article_num, language)` — calls `scroll_payloads()`, post-filters in Python to ensure exact CSV token match (prevents "92" matching "192"/"292"), returns sorted list of citing article dicts |
| `api/main.py` | Added `CitingArticleItem` + `CitingArticlesResponse` Pydantic models; new `GET /api/article/{article_id}/citing?language=en` endpoint |

**Design notes:**
- Used Qdrant scroll (not LlamaIndex filter abstraction) to avoid `CONTAINS` operator limitations for CSV substring matching
- Python post-filter splits `referenced_articles` CSV and checks exact set membership — no false positives
- Results sorted by article number (numeric); articles with letter suffixes (e.g. "92a") sort after their base number

### Article title precision (investigation complete)

**Finding**: `article_title` in metadata is parsed directly from the `stitle-article-norm` CSS class inside `eli-title` in EUR-Lex HTML — **not LLM-synthesized**. Any title variation reflects the EUR-Lex source.

**Fix**: Added fallback in `_process_article_div()` — if `stitle-article-norm` is absent, the ingester now tries the first `<p>` in `eli-title` that is not `title-article-norm` (the article-number heading). Takes effect on next re-ingest.

---

## 2026-03-15 — UI fixes: missing article detection + numbered point label

### Missing/deleted article detection
Backend already returned 404 for missing articles. Frontend handling added:
- `ArticleNotFoundError` class added to `api.ts` (thrown on 404)
- `AppLayout` owns `viewerError` state
- Both `ChatPanel.handleSourceClick` and `DocumentViewer.handleArticleRef` catch `ArticleNotFoundError` and call `onArticleNotFound(id)`, which sets `viewerError` and clears `selectedArticle`
- Viewer renders a "Article N was not found in the CRR" message panel instead of blank

### Numbered point label not bold (minor formatting fix)
Root cause: `NUMBERED_PARA_RE = /^\d+\.\s+/` required text on the same line — a bare `7.` label (backend returns the number on one line, text on the next) didn't match and rendered as a plain `<p>`. Fixed by extending regex to `/^\d+\.(\s|$)/` and handling the empty-body case in the render path. (`ProvisionText.tsx`)

---

## 2026-03-15 — Article 94 duplicate paragraphs: root cause fixed

### Bug
Article 94 in the document viewer showed paragraphs 4–6 twice and paragraph 7 as a bare "7." label with no body text.

### Root cause
LlamaIndex `VectorStoreIndex.from_documents()` split Article 94 into **2 overlapping Qdrant nodes** despite `transformations=[]` being set. The article (~1 200 tokens) exceeded the default `Settings.chunk_size = 1024`, which some internal LlamaIndex paths still consult independently of the transformations list. `get_article()` retrieved both chunks and concatenated them verbatim, producing the duplicate.

Diagnostic: `VectorStore.scroll_payloads()` revealed **117 articles** with >1 node in the current EN collection (948 total nodes vs expected 745), confirming this is a stale-data issue from a previous ingest before the `chunk_size` guard was in place. Notable duplicates: Article 4 (23 nodes), Annex IV (31 nodes), Article 473a (7 nodes).

The BeautifulSoup parser itself is clean — running `_extract_structured_text()` directly on `crr_raw_en.html` produces correct, non-duplicate output.

### Fixes applied

| File | Change |
|------|--------|
| `src/indexing/index_builder.py` | `_configure_settings()` now sets `Settings.chunk_size = 8192` and `Settings.chunk_overlap = 0` — explicit guard so no article can be chunked regardless of other LlamaIndex settings |
| `src/query/query_engine.py` | `get_article()` deduplicates retrieved nodes by LlamaIndex internal `node_id` before concatenating (safeguard against the same Qdrant record returned twice) |
| `tests/conftest.py` | Added `eurlex_html_with_amendment_blocks` fixture — reproduces Article 94's `<p class="modref">` amendment marker structure |
| `tests/unit/test_eurlex_ingest.py` | Added `TestAmendmentBlockParsing` (4 regression tests): no duplicate paragraphs, amendment markers stripped, para 4 not duplicated, all points present |
| `tests/unit/test_query_engine_unit.py` | Added `test_get_article_deduplicates_by_internal_node_id` |

### Test results
182/182 unit tests pass.

### Action required
Run `python -m src.pipelines.ingest_pipeline --reset` (or the Colab notebook) to rebuild the Qdrant collection cleanly. All 117 affected articles will be fixed after re-ingest.
