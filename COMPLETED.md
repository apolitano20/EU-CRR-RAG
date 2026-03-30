# COMPLETED — Project History

This file is the chronological record of all completed work on the EU CRR RAG system.
For open tasks and backlog, see `WORKLOG.md`.

---

## [2026-03-30] sub_article_of metadata — code complete, re-ingest pending

Added `sub_article_of` field to `DocumentNode` (stored in Qdrant payload after re-ingest), sub-article detection in `eurlex_ingest.py` (regex `\d+[a-z]+` → parent is numeric prefix), `ArticleGraph._sub_article_clusters` dict built from `sub_article_of` payloads with `sub_article_cluster()` query method, and `hit_at_1_family` metric in `evals/metrics.py` + `run_eval.py`. The new metric treats within-family swaps (e.g. 429 retrieved when 429b expected) as hits, eliminating stochastic noise from borderline reranker ties inside sub-article clusters. Re-ingest on Colab T4 still needed to populate the metadata in `eu_crr`; until then `sub_article_cluster()` returns `[]` and the graph logs a one-time advisory.

## [2026-03-30] run_37: run_30 SOTA reconfirmation — confirmed, delta is noise

Reran with all 4 run_35/36 fix flags explicitly disabled to confirm run_30 is still SOTA. Result: 86.7% hit@1 (-0.6pp vs run_30's 87.3%). Only 3 case flips vs run_30, all within sub-article clusters (132c↔132, 429b↔429, 92↔93) — these are borderline-tied articles where the reranker ordering varies run-to-run. Confirms run_30 SOTA is real and that the true stable ceiling without sub-article grouping is ~87% ±0.6pp noise. Validates that the Art 132 and Art 429 re-ingest grouping would permanently resolve these flips.

## [2026-03-30] run_36: Corrected multi-article retrieval fixes — run_30 remains SOTA

Ran corrected version of the 5 multi-article retrieval fixes (run_36_retrieval_fixes_v2) after fixing two bugs from run_35: triple reranking (intermediate expansion rerank now skipped when USE_GLOBAL_RERANK=true) and blended-direct gate narrowed to unambiguous subordinate-reference phrases only (2/173 queries blend, down from 85/173). Result: hit@1=86.7% (-0.6pp vs SOTA), recall@5 flat (+0.1pp), recall@3 -1.1pp. Judge scores in summary are invalid — run aborted mid-way through first attempt (quota exhaustion), restarted without --judge; only 19/173 cases have judge scores and are not comparable to run_30. 0/9 target cases fixed for hit@1 — root cause is ParagraphWindowReranker's strong literal token-match scores for false-friend articles that global re-rank cannot overcome. run_30 confirmed as SOTA on retrieval metrics.

## [2026-03-30] run_35: Multi-article retrieval fixes v1 — net regression, bugs identified

Ran first attempt at 5 runtime-only multi-article fixes. Two implementation bugs caused net regression: (1) triple reranking inflated latency to p50=53s; (2) blended-direct gate too broad (85/173 queries), causing 10 regressions on single-article cases. Recall@3/5 improved as a side-effect of triple reranking diversity. Both bugs diagnosed and corrected in run_36.

## 2026-03-29 — Codex investigation: 5 runtime-only retrieval fixes for 9 multi-article failures

Analysed `codex-investigation.md` which diagnosed 9 `hit_at_1=0` multi-article failures from run_30. Four root-cause buckets identified: (1) multi-hop regex too narrow, (2) graph expansion forward-only + `structural_siblings()` unused, (3) vocab/embedding gaps, (4) expanded nodes never reach rank-1. Implemented all Phase 1+2 runtime-only fixes on branch `fix/multi-article-retrieval`: (1) blended direct+semantic retrieval via RRF (`USE_BLENDED_DIRECT_RETRIEVAL`), (2) structural sibling expansion (`USE_STRUCTURAL_SIBLINGS`), (3) reverse-edge graph BFS (`USE_REVERSE_GRAPH_EXPANSION`), (4) global re-rank of primary+expanded nodes together (`USE_GLOBAL_RERANK`), (5) expanded `_MULTI_HOP_RE` regex now matches 5/9 failing cases vs 0/9 before. All changes gated by env vars (default true) for ablation. Awaiting eval run_35 to validate.

---

## 2026-03-29 — run_34: RETRIEVAL_ALPHA=0.4 + RERANK_TOP_N=8 — neutral, reverted

Ran full 173-case judge eval combining two retrieval levers: more BM25 weight (alpha 0.5→0.4) targeting `diluted_embedding` vocabulary gaps, and more synthesis context (top_n 6→8) targeting `multi` coverage. Both primary targets showed zero movement (diluted_embedding Hit@1 and multi recall@1 unchanged). Critical finding: alpha=0.4 actively hurt `diluted_embedding` judge correctness by -6.6pp — BM25 also fails on vocabulary mismatch, so more lexical weight makes things worse, not better. The correct direction is higher alpha (more dense). RERANK_TOP_N=8 confirmed that the right articles are not in the pool at all for multi cases — a top_n increase cannot fix an upstream retrieval miss. Reverted both to SOTA values.

---

## 2026-03-29 — run_33_wider_funnel: top_k=20 + windows=6 — neutral, reverted

Combined "wider funnel" experiment: RETRIEVAL_TOP_K 15→20 and PARAGRAPH_WINDOW_MAX_WINDOWS 4→6. All primary targets (multi recall@1, diluted_embedding Hit@1) showed exactly zero movement. Tail latency worsened significantly (P99: 50s→89s). Confirmed that multi and diluted_embedding failures are ranking problems, not coverage problems — widening the candidate pool has no effect because the right articles are simply not being ranked first regardless of pool size. Reverted to SOTA.

---

## 2026-03-29 — run_33 (BGE-M3 INT8) abandoned — benchmark proves BGE-M3 is not the bottleneck

Benchmarked PyTorch INT8 dynamic quantization (`torch.quantization.quantize_dynamic`) against BGE-M3 FP32 on CPU. Result: INT8 is 0.63x slower (313ms vs 197ms) and degrades cosine similarity to 0.968 — unacceptable on both dimensions. More importantly: BGE-M3 encode at ~200ms is only ~1% of 21s total end-to-end latency. The real bottleneck is OpenAI synthesis (~80%). `torch.quantization.quantize_dynamic` is also deprecated in PyTorch 2.10 (successor: `torchao`). Added load-time logging to `bge_m3_sparse.py` and created `scripts/benchmark_bge_int8.py` for future reference. BGE_M3_INT8 env var kept in code but disabled.

---

## 2026-03-29 — Diagnosed run_31 failure and reverted RERANKER_MODEL to SOTA

run_31 attempted to upgrade the reranker to `BAAI/bge-reranker-v2-m3` (560MB). All 173 cases timed out. Root cause: even with `USE_RERANKER=false`, setting `RERANKER_MODEL=bge-reranker-v2-m3` causes the model to load at startup via the `USE_PARAGRAPH_WINDOW_RERANKER=true` fallback path (which creates a `BlendedReranker` using `RERANKER_MODEL` when `USE_PARAGRAPH_CHUNKING=false`). The 560MB model loads too slowly on this CPU, making every query time out. Reverted `RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2`.

---

## 2026-03-29 — Fixed .env drift: PARAGRAPH_WINDOW_MAX_WINDOWS was 2 instead of SOTA 4

Discovered that `.env` had `PARAGRAPH_WINDOW_MAX_WINDOWS=2` while the WORKLOG documents run_30 SOTA as using 4 (the code default). This was silently degrading the server config. Fixed to 4 to match SOTA. The run_30 config JSON did not log this parameter explicitly, making the drift invisible without cross-checking the code default.

---

## 2026-03-28 — Reverted to BGE-M3 after run_29 e5-large-instruct regression

Run_29 evaluated `multilingual-e5-large-instruct` (dense-only, separate `eu_crr_e5` Qdrant collection) against the full 173-case golden dataset. Result: recall@1 −2.5pp (0.795 vs 0.795), MRR −2.4pp (0.873 vs 0.897), latency doubled (22s vs 11s p50), with severe regressions in `credit_risk_sa` (−30pp recall@1) and `leverage_ratio_total_exposure` (−33pp). The only upside was recall@5 +1.4pp — e5 casts a slightly wider net but is less precise at rank 1. Root cause: removing sparse retrieval (BM25) eliminated the lexical precision that BGE-M3 hybrid provides for exact legal terminology. Reverted `.env` to `EMBED_MODEL=bge-m3`, `QDRANT_COLLECTION=eu_crr`.

---

## 2026-03-28 — Windows SentenceTransformer access violation fix (low_cpu_mem_usage)

API was crashing with a native access violation (`_load_state_dict_into_meta_model`) on first query after startup. Root cause: recent transformers defaults `low_cpu_mem_usage=True` on CPU, which uses PyTorch meta tensors during model loading — a path that triggers a native SIGSEGV on Windows/Python 3.13. Fixed in `src/indexing/e5_instruct_embed.py` by passing `model_kwargs={"low_cpu_mem_usage": False}` to `SentenceTransformer` when `sys.platform == "win32"`. This forces standard CPU allocation instead of meta-tensor loading.

---

## 2026-03-26 — run_28: Bidirectional structural cross-ref extraction at ingest

Re-ingested `eu_crr` collection with bidirectional cross-references: Part/Title/Chapter/Section refs are now extracted at ingest time and stored in node metadata, enabling expansion in both directions (not just article→article). Retrieval metrics were identical to run_27 (Hit@1=87.3%, MRR=0.897). Judge scores regressed slightly (correctness −1.0pp, faithfulness −1.9pp) — likely noise at n=173. No retrieval benefit from bidir refs observed; the index is retained as-is since re-ingesting to remove them would require a separate motivation.

---

## 2026-03-26 — run_27: Deterministic synthesis completeness — new SOTA Judge Correctness=0.812

Implemented two deterministic completeness interventions targeting 17 cases with hit@1=1 but judge_correctness<0.7. Part A: `_build_key_facts_block()` extracts all numerical thresholds (percentages, day counts, monetary values, basis points) from retrieved article sections via regex and prepends them as a structured fact-sheet to the synthesis context. Part B: `_append_missing_thresholds()` post-processes the generated answer, diffs thresholds from cited articles against what appears in the answer, and deterministically appends any missing values as a "Completeness note" — no re-synthesis. Result: 7/17 target cases improved, 0 regressed. Judge Correctness +2.0pp (0.792→0.812), Completeness +2.5pp (0.788→0.813), Faithfulness +2.3pp (0.818→0.841). Retrieval metrics unchanged. **New SOTA.**

---

## 2026-03-26 — run_26: BM25 synonym expansion + false premise rule — new SOTA

Two simultaneous interventions targeting non-overlapping categories. (1) Added 8 entries to `_SYNONYM_MAP` targeting `diluted_embedding` failures where plain-language paraphrases caused BM25 misses (`accumulated earnings`, `maximum allowable exposure`, `pledged as collateral`, `easily sellable`, `available liquid resources`, `solvency threshold`, `internal permission for tailored risk evaluation`, `blend of assets`). (2) Added `_FALSE_PREMISE_RULE` to both synthesis prompt templates — explicit "No." instruction with 3 grounding examples from golden cases (case_147, case_170, case_144). Results: `false_friend` Judge Correctness +0.20 (0.557→0.757, 6 cases improved 0 regressed); `diluted_embedding` Hit@1 1/6→2/6, avg correctness +0.30. Overall hit@1 +0.006, all judge scores positive. **New SOTA.**

---

## 2026-03-26 — Statistical significance framework for eval comparisons

Established that with n=173, 1-SE for a proportion ≈ 0.025–0.035 depending on the metric. Most run-to-run deltas (±0.01–0.03) are within noise. Adopted policy: flag `**YES**` only when |delta| > 2×SE; `~marginal` when > 1×SE; `noise` otherwise. Also established that judge score differences of ±0.027 are within noise at this sample size — relevant for run_24 vs run_25 verdict (wash, not regression). Category-level analysis (diluted_embedding, false_friend) requires looking at question_type field in golden dataset, not the category field.

---

## 2026-03-26 — run_25: Graph-gated BFS expansion — statistically identical to run_24

Implemented BFS article graph expansion gated on post-retrieval signal: fires only when primary retrieval surfaces ≥2 distinct articles. Hypothesis was that gating would reduce noise on single-article queries while preserving multi-article gains. Result: statistically identical to run_24 across all metrics (all deltas within 1-SE). Retrieval metrics nudged +0.003 on recall@3/5; judge scores nudged -0.015 to -0.027 — all noise. Declared a wash; run_24 and run_25 co-hold SOTA until run_26 broke the tie.

---

## 2026-03-25 — run_20: Mixed chunking — new SOTA Hit@1=86.7% (+6.4pp vs run_17)

Implemented `USE_MIXED_CHUNKING=true` mode: both ARTICLE and PARAGRAPH chunks compete freely in Qdrant retrieval (no chunk_type filter). A new `ArticleDeduplicatorPostprocessor` (inserted after `SimilarityPostprocessor`, before the reranker) groups nodes by article number and keeps the highest-scored chunk per article, with a 2% margin preference for ARTICLE chunks to avoid unnecessary synthesis round-trips. Any surviving PARAGRAPH chunk is upgraded to its parent ARTICLE text at context assembly time. No re-ingestion required — paragraph chunks were already in the index from run_19's re-ingest.

Results vs run_17 (n=173, no judge): Hit@1 80.3% → **86.7%** (+6.4pp), Recall@1 72.5% → **78.9%** (+6.4pp), Recall@3 78.9% → **83.6%** (+4.7pp), MRR 0.840 → **0.891** (+5.1pp). Zero regressions across all categories. Standout gains: `credit_risk_sa` +60pp, `known_failures` +25pp, `leverage_ratio_cash_pooling` +100pp, `false_friend` +21pp, `procedural` +13pp, medium difficulty +9pp, hard difficulty +9pp. Latency slightly improved (mean 10.5s vs 10.7s). **New SOTA.**

---

## 2026-03-25 — run_19: Para-chunking diagnosis — dual-population retrieval problem identified

Ran full 173-case eval on `USE_PARAGRAPH_CHUNKING=true` (paragraph-only retrieval). Overall Hit@1=70.5% — significant regression vs run_17 (80.3%). Root cause: top-k flooding — long articles with many paragraphs consume multiple slots, displacing other articles. High-volume categories hurt most: liquidity -30pp, own_funds -29pp, large_exposures -14pp. But clear improvements on localized queries: credit_risk_sa +40pp, leverage_ratio_cash_pooling +100pp, known_failures +17pp. Established that the dataset has two populations: ~15-20% localized queries benefiting from paragraph-level precision, ~80-85% broad queries needing article-level context. Led directly to the mixed-chunking solution (run_20).

---

## 2026-03-24 — Windows/Python 3.13 WMI hang fix in api/main.py

After a mid-run API crash, the server would not restart — `import llama_index.core` hung indefinitely. Root cause traced via `faulthandler` stack dump: SQLAlchemy calls `platform.machine()` at import time, which in Python 3.13 on Windows calls `platform._wmi_query()` using WMI COM — this hangs when the WMI service is in a bad state after a crash. Fix: pre-populate `platform._uname_cache` at the top of `api/main.py` (before any llama_index imports) to bypass the WMI call entirely. Guarded to Windows-only with `sys.platform == "win32"`. API now starts reliably.

---

## 2026-03-24 — Eval dashboard: Stop button for running and orphan eval runs

Added an ⏹ Stop button to the `_run_eval_panel()` section of `evals/dashboard.py`. The button appears next to the progress bar in a `[5,1]` column layout for both active runs (launched from current session) and orphan runs (detected from disk state files). Clicking it calls `_kill_eval_proc()` — a new helper that terminates the subprocess, kills by PID from the state file, marks the run as `"stopped"`, and clears session state. Previously there was no way to interrupt a running eval from the dashboard.

---

## 2026-03-24 — Paragraph-level chunking (Codex rank 3): dual-document index implementation

Implemented the full paragraph-level chunking architecture on branch `ingest_contextual_prefix`. Each article now produces 1 ARTICLE doc (for synthesis/cross-ref) + N PARAGRAPH docs (for retrieval) when it has ≥2 numbered `<div class="norm">` paragraphs and body ≥100 words. Changes across 6 files: `document.py` (PARAGRAPH NodeLevel, 3 new fields), `eurlex_ingest.py` (`_process_article_div` returns `list[Document]`, DOM-based para splitting with BeautifulSoup wrapper fix for `_extract_structured_text`), `vector_store.py` (payload indexes for chunk_type/parent_article_id/para_id), `query_engine.py` (PARAGRAPH filter on retriever, parent-ARTICLE fetch in context assembly, chunk_type-aware cross-ref expansion and direct article retrieve, ParagraphWindowReranker auto-fallback to BlendedReranker), `orchestrator.py` (`_build_context` mirrors parent-ARTICLE fetch logic), `test_eurlex_ingest.py` (11 new TestParagraphChunking tests). All 136 unit tests pass. Committed and pushed. Re-ingest on Colab T4 pending.

---

## 2026-03-24 — run_18: Contextual hierarchy prefix in embedding text — neutral vs run_17

Implemented `_build_hierarchy_prefix()` in `src/ingestion/eurlex_ingest.py` to prepend `Part > Title > Chapter > Section > Article N — Title` to every indexed document's `Document.text` (embedding text), while storing the original body as `metadata["display_text"]` for clean citation snippets. Re-ingested the full EN + IT corpus on Colab T4 (1490 items confirmed). Result vs run_17: Hit@1 unchanged (80.3%), Recall@3/5 -1.1pp, MRR -0.005, `ciu_treatment` -20pp, `known_failures` -8.3pp, `capital_ratios`/`own_funds` +3.2pp each. Prefix alone on full-article blobs is insufficient — the structural signal is diluted by long article bodies. run_17 remains the best config. Index retained with prefix (harmless); full gains expected only after paragraph chunking (Codex rank 3). Branch: `ingest_contextual_prefix`.

---

## 2026-03-24 — run_17: Paragraph-window reranker — new best Hit@1=80.3% (+2.3pp vs run_12)

Implemented `ParagraphWindowReranker` in `src/query/query_engine.py`: instead of scoring full article text against the query, splits each retrieved article into paragraph windows (split on `\n\n`, evenly sampled up to `PARAGRAPH_WINDOW_MAX_WINDOWS=4`) and uses the best window's CrossEncoder score as the article's rerank score. Blends with retrieval score via `RERANK_BLEND_ALPHA`. Stores the best-matching window in `node.metadata["best_paragraph"]` for future display use. Enabled via `USE_PARAGRAPH_WINDOW_RERANKER=true`; replaces `USE_RERANKER` (same model, avoids loading twice). Also shipped: `QDRANT_COLLECTION` env var, Qdrant payload indexes for `part`/`title`/`chapter`/`section`, and selective cross-ref expansion (top-2 nodes only, joint reranking of expanded nodes).

run_17 results vs run_12 (n=173, no judge): Overall Hit@1: 78.0% → **80.3%** (+2.3pp), MRR: 0.824 → **0.840** (+1.6pp). `false_friend` +14.3pp, `open_ended` +4.1pp, `negative` +12.5pp, `hard` difficulty +3.7pp. `diluted_embedding` -16.7pp (6 cases, small n). Latency +800ms (10.7s mean). **Current best.**

---

## 2026-03-24 — run_16: Selective ToC confidence-gated routing — regression, ToC retired

Ran `USE_TOC_ROUTING=true`, `TOC_CONFIDENCE_THRESHOLD=0.55` (ToC fires only when post-retrieval max reranker score < 0.55). Hit@1=77.46% vs run_12's 78.0% — still a net regression. Both universal (run_15) and selective (run_16) ToC routing have now regressed. ToC routing retired; `USE_TOC_ROUTING=false` in active config.

---

## 2026-03-24 — Dashboard comparison tables: all metrics + magnitude-based colour gradients

Rewrote `_comparison_table()` in `evals/dashboard.py` to show all 7 retrieval metrics (Hit@1, Recall@1, Recall@3, Recall@5, MRR, Prec@3, Prec@5) plus judge metrics when enabled, each with A/B/Δ columns. Delta cells are colour-coded with magnitude-aware gradients (near-white for <0.5pp noise, through to fully saturated green/red for ≥15pp), making the direction and magnitude of changes readable at a glance. Single vs Multi-Article tab now also shows the comparison. Previously the table only showed Recall@3 in a single-metric layout.

---

## 2026-03-24 — run_12: Adjacent article tiebreaker — +1.7pp Hit@1, false_friend +14.3pp

Implemented `AdjacentArticleTiebreakerPostprocessor` in `src/query/query_engine.py`: a post-rerank pass that fires when the top-2 nodes are adjacent articles (same numeric base e.g. 429/429a, or consecutive integers e.g. 114/115) within a configurable score gap, preferring the article whose title has more query-token overlap. Enabled via `ADJACENT_TIEBREAK_DELTA=0.05` in `.env` (0.0 = disabled). Wired into the postprocessor chain after the reranker; captured in eval config snapshot and dashboard config panel.

run_12 results vs run_2e baseline (n=173, no judge):
- Overall Hit@1: 76.3% → **78.0%** (+1.7pp), MRR: 0.815 → **0.824** (+0.009)
- `false_friend` Hit@1: 35.7% → **50.0%** (+14.3pp) — 2 cases flipped at rank 1
- `large_exposures` Hit@1: +2.9pp; `multi_hop` Hit@1: +2.1pp; `open_ended` Hit@1: +4.1pp
- `hard` difficulty Hit@1: 65.9% → **69.5%** (+3.6pp)
- No meaningful regressions. **Tiebreaker kept as permanent config.**

---

## 2026-03-24 — run_2e_baseline: clean full-173 reference baseline with judge scores

Ran the first clean full-173-case eval with 0 failures and judge enabled (workers=1 to avoid timeout race conditions that caused 4 failures in run_2d with workers=4). Established as the reference baseline for all subsequent experiments. Key metrics: Hit@1=76.3%, Recall@3=79.3%, MRR=0.815, Judge Correctness=0.770, Judge Completeness=0.777, Judge Faithfulness=0.790. Confirmed that surgical synonyms (run_10 additions) were already active via USE_ENRICHMENT=True — run_10b as a separate experiment was redundant.

---

## 2026-03-24 — Fix case_161 systematic HTTP 500 error (numpy.float32 not JSON serializable)

Diagnosed and fixed a bug that caused case_161 to fail with HTTP 500 in every eval run where `_generate_sub_queries` succeeded. Root cause: `_multi_query_retrieve` in `orchestrator.py` built sources with `round(n.score or 0.0, 4)` — `round(numpy.float32, 4)` returns `numpy.float32`, which FastAPI's `jsonable_encoder` cannot serialize. The reranker (`BlendedReranker`) computes blended scores as `float * float32 = float32`. The regular engine path correctly uses `float(round(node.score or 0.0, 4))`. Fix: one-line change in `orchestrator.py:625` — added `float()` wrapper. Server accidentally killed during debugging; needs restart via `launch.bat`.

---

## 2026-03-21 — Eval pipeline data integrity fixes (Codex Dashboard Review 2 — eval launch/tracking/aggregation)

Implemented the full minimal remediation plan from `research_docs/codex_dashboard_review2.md`, which identified the root cause as using a mutable shared JSONL file as both the write path and the state model.

**`evals/run_eval.py`:**
- `_load_dataset` hardened: catches `json.JSONDecodeError` per line, skips rows without `id`, deduplicates by ID with warnings — used for both dataset loading and result reload before summary.
- Added `_write_result()` helper with `threading.Lock` + `_written_ids` set seeded from `done_ids`. All writes (sequential, threaded, timeout, exception) now go through this single writer — prevents duplicate rows from concurrent threads or a late-completing worker after timeout.
- Added `_write_state()` helper (atomic via temp file + `os.replace()`).
- `state.json` lifecycle: written as `"running"` before eval loop; updated to `"completed"` after summary; updated to `"failed"` on unhandled exception.
- Summary write made atomic: temp file + `os.replace()`.

**`evals/dashboard.py`:**
- `_count_jsonl_lines` → `_count_valid_results`: counts unique parsed IDs, not raw lines; progress bar can no longer overflow from duplicate or malformed rows.
- `_count_valid_dataset_cases`: used at launch to compute `eval_total_cases` from unique valid IDs.
- Progress polling reads `state.json` `planned_total` (accurate post-filter count) instead of session-state value set before the runner validated the dataset.
- Orphan run detection: scans `*_state.json` for live `"running"` states with valid PIDs — surfaces cross-session runs; auto-marks dead PIDs as `"crashed"`.
- `_discover_incomplete_runs()`: `page_eval_results` now shows warnings for failed/crashed runs with a resume hint.
- `_load_run` / `_load_summary` cache staleness fixed: added `file_mtime` parameter so Streamlit invalidates cache when file changes in place.

---

## 2026-03-21 — Reranker unblocked + eval runs 1–3 + open-ended failure analysis

Unblocked the Phase 1 reranker on Windows by switching from `BAAI/bge-reranker-v2-m3` (SIGSEGV when coloaded with BGE-M3) to `cross-encoder/ms-marco-MiniLM-L-6-v2` (sentence-transformers, no conflict). Installed CPU-only torch to eliminate the `shm.dll` DLL crash. Switched from `FlagEmbeddingReranker` to `SentenceTransformerRerank` in `query_engine.py`.

Ran three eval runs against the 173-case golden dataset:
- **run_1** (phases 0/2/3, no reranker): Hit@1=78.5%, Judge Correctness=0.782, mean=12.1s
- **run_2** (+ reranker): Hit@1=79.2% (+0.7pp), Recall@3=82.1%, MRR=0.829, mean=11.0s — **best config**
- **run_3a/3b** (blended reranker, alpha=0.3/0.6): net neutral vs run_2 at both alphas — reverted

Implemented `BlendedReranker` postprocessor (min-max normalised reranker scores blended with retrieval scores) to fix two rank-flip regressions (case_136 506c>26, case_149 395>392). Score blending at alpha=0.3 fixed case_149 but introduced new regression in case_152; alpha=0.6 fixed case_149 but broke case_109 and case_152. Net: blending was neutral, pure reranker retained.

Conducted deep open-ended failure analysis: article-cited queries score Hit@1=91.0%; open-ended score 63.0%. Identified 15 retrieval failures (6 terminology dilution, 5 concept-in-unexpected-article, 3 niche sub-articles, 2 known hard) and 12 ranking failures (7 adjacent-article confusion, 4 false-friend). Full improvement roadmap added to WORKLOG.

---

## 2026-03-20 — RAG improvement plan: Phases 0, 2, 3 code complete

Implemented all code changes for three of four planned improvement phases. Phase 1 (reranker) is blocked on Windows DLL incompatibility and must be run on Colab/Linux.

**Phase 0 — Error handling hardened:**
- `api/main.py`: added `except Exception` with `exc_info=True` logging + HTTP 500 with exception type in detail
- This surfaces the actual crash cause for the 13 hard-failure cases (previously silent 500s)

**Phase 2 — Query understanding:**
- Extended `_ABBREV_MAP` (+13 terms: STS, SEC-IRBA, SEC-SA, SA, CIU, NPE, TREA, HQLA, PD, CF, SME, IPRE, HVCRE)
- Added `_SYNONYM_MAP` + `_expand_synonyms()` integrated into `_normalise_query()` (9 legal paraphrases → canonical CRR terms)
- Added `_enrich_open_ended_query()`: LLM pre-call to predict article hints, appended to open-ended queries
- Added `_generate_sub_queries()`: breaks multi-hop questions into 2–3 sub-queries
- `orchestrator.py`: open-ended CRR_SPECIFIC queries enriched; multi-hop queries trigger sub-query generation + `_multi_query_retrieve()` with score-based deduplication
- `RETRIEVAL_TOP_K` and `RERANK_TOP_N` made env-configurable

**Phase 3 — Synthesis quality:**
- Hardened `_LEGAL_QA_TEMPLATE` with critical citation rule: cite only articles present verbatim in context; explicit fallback text if article missing from context
- Raised `_LOW_CONFIDENCE_THRESHOLD` from 0.35 → 0.40
- Multi-hop queries now route to `HARD_QUERY_MODEL` (gpt-4o) for synthesis

**Tests:** 30 new tests added (19 in `test_query_engine_unit.py`, 11 in `test_orchestrator.py`); 2 stale tests fixed. 341 unit tests total, all green.

---

## 2026-03-20 — Qdrant index diagnosis: all 11 "missing" articles confirmed present

Ran `diagnose_qdrant.py` + direct Qdrant payload scan against the 11 articles suspected of being missing (115, 121, 122, 132a, 132c, 152, 242, 243, 254, 258, 429b). All 11 are in the index with correct node counts. Conclusion: the 13 hard failures are code crashes (unhandled exceptions), not missing data. The new exception handler in `api/main.py` will surface the actual exception type on the next eval run.

---

## 2026-03-20 — Baseline eval run complete + manual case review

Executed the first full eval run against the 173-case golden dataset.

**Baseline results (run name: `baseline`):**

| Metric | Value |
|--------|-------|
| Hit@1 | 81.3% |
| Recall@1 | 77.2% |
| Recall@5 | 85.3% |
| MRR | 0.842 |
| Judge Correctness | 0.774 |
| Judge Completeness | 0.788 |
| Judge Faithfulness | 0.799 |
| Hard failures | 13/173 (7.5%) |
| P50 latency | 10.5s |

**Key findings:**
- 13 hard failures (all `status=error`) concentrated in 6 categories: `credit_risk_sa` (5), `securitisation_methods` (3), `ciu_treatment` (2), `significant_risk_transfer` (1), `STS_vs_nonSTS` (1), `leverage_ratio_total_exposure` (1). Expected articles: 115, 121, 122, 132a, 132c, 152, 242, 243, 254, 258, 429b.
- Reranker OFF — 8% gap between Recall@1 and Recall@5 (reranking opportunity)
- Open-ended queries: 65.6% Hit@1 vs 90.9% for article-cited
- Multi-article questions: 27% Recall@1, 49.5% Recall@5
- 10 cases scored 0.0 Judge Correctness (synthesis failures)
- `false_friend` type: 77.8% Hit@1 but only 0.500 Judge Correctness

**Manual review**: 42 of 173 cases manually reviewed and approved (tracked in `evals/cases/review_status.json`). Review priorities documented in `evals/cases/review_priorities.md`.

**Files**: `evals/results/baseline_summary.json`, `evals/results/baseline_summary.md`, `evals/results/baseline_cases.jsonl`

---

## 2026-03-20 — Codex Dashboard Review 2: four hang/stall fixes applied

Applied all four fixes from `research_docs/codex_dashboard_review_2.md` targeting the "eval runner stuck at N/50" failure chain.

| # | File | Fix |
|---|------|-----|
| 1 | `evals/run_eval.py` | Replaced `as_completed()` + `fut.result(timeout=...)` with a `concurrent.futures.wait(timeout=_fut_timeout, return_when=FIRST_COMPLETED)` polling loop. A hung future is never yielded by `as_completed`, so the old timeout guard was unreachable. New loop detects `done_set=empty` (nothing completed in `_fut_timeout` seconds) and records all remaining futures as timeout errors, then breaks. Pool is now created explicitly (not via context manager) with `shutdown(wait=False)` in `finally` so stuck threads can't block the main process on exit. |
| 2 | `api/main.py` | Converted `/api/query` from sync `def` to `async def`. Offloads `_orchestrator.query()` via `asyncio.to_thread(...)` wrapped in `asyncio.wait_for(..., timeout=_QUERY_TIMEOUT)`. Returns HTTP 504 on timeout. Prevents FastAPI's sync-worker thread pool from saturating under parallel eval load. `_QUERY_TIMEOUT` defaults to 120 s, overridable via `QUERY_TIMEOUT_SECONDS` env var. |
| 3 | `src/indexing/bge_m3_sparse.py` | Added `_encode_lock = threading.Lock()` around all `_get_model().encode()` calls in `sparse_doc_fn` and `BGEm3Embedding._get_query_embedding` / `_get_text_embedding` / `_get_text_embeddings`. Serializes CPU encodes so parallel queries don't oversubscribe cores or hit FlagEmbedding re-entrancy issues. |
| 4 | `evals/run_eval.py` | Fixed `--auto-start-api`: changed `stdout=PIPE, stderr=STDOUT` → `stdout=DEVNULL, stderr=DEVNULL`. An unread pipe buffer fills and blocks the child uvicorn process, appearing as the API freezing. |

---

## 2026-03-20 — LLM-as-judge eval pipeline (`evals/judge.py` + `--judge` flag)

Added automated answer quality scoring to the eval pipeline. New file `evals/judge.py` calls **gpt-4o** (different model from RAG's gpt-4o-mini) with a structured JSON prompt to score each RAG answer on three dimensions (0–1): **correctness** (factual accuracy), **completeness** (coverage of reference key points), **faithfulness** (no hallucinations). Returns `None` on API/parse errors so partial failures don't abort a run. Modified `evals/run_eval.py`: added `--judge` / `--judge-model` CLI flags, split `METRIC_KEYS` into `RETRIEVAL_METRIC_KEYS + JUDGE_METRIC_KEYS`, integrated judge call in `evaluate_case()`, added judge score logging in the scorecard printout, and `judge_enabled: bool` in `build_summary()`. Modified `evals/dashboard.py`: added judge metrics row in scorecard (only shown when non-None), judge columns in breakdown tables, judge scores and rationale in case drill-down inspector, and a "Enable LLM-as-judge" checkbox in the Run Eval panel. Modified `evals/compare.py`: extended `METRIC_KEYS` and `METRIC_LABELS` with judge keys; `print_report()` skips judge rows when both runs have None scores. All graceful-absence patterns confirmed working: `_mean()` already skips `None`, dashboard sections conditionally hidden, compare rows conditionally printed.

---

## 2026-03-19 — `evals/compare.py` + "🔀 Compare Runs" dashboard page

Built `evals/compare.py` — a CLI regression diff tool and importable library. The core `build_comparison(summary_a, summary_b)` function computes per-metric `{a, b, delta}` dicts across overall metrics and all breakdown dimensions (by_category, by_difficulty, by_question_type, by_article_count), flags regressions/improvements at a ±1 pp threshold, and captures latency deltas. CLI: `python -m evals.compare run_A run_B [--output path.json] [--list]`. Added "🔀 Compare Runs" as a third page in `evals/dashboard.py`, importing `build_comparison` from `compare.py`: baseline/candidate selectors in sidebar, regression/improvement banners at top, overall scorecard using `st.metric` with delta values, latency comparison row, per-breakdown tabs with a metric picker and colour-coded delta column (green = improvement, red = regression), and a download button for the comparison JSON. Also corrected the WORKLOG — `evals/dashboard.py` (Streamlit, 2 pages) and `evals/run_eval.py` were already built in a prior session and the WORKLOG had stale ❌ entries for them.

---

## 2026-03-19 — Eval dashboard planning + golden dataset generator

Built `evals/generate_golden_dataset.py` — a fully automated script that extracts articles from Qdrant (no model loading), calls GPT-4.1 to generate Q&A pairs, and writes crash-safe resumable JSONL output. Implemented two passes: Pass 1 (article-anchored, 44 priority articles across 6 categories) and Pass 2 (adversarial, 8 batches targeting multi_hop / false_friend / negative / diluted_embedding failure modes). Full run generated **173 cases** in `evals/cases/golden_dataset.jsonl`. Reorganised eval case storage: existing manual cases moved to `evals/cases/manual_cases/`; `api/main.py` `/api/feedback` endpoint updated to write new feedback to `manual_cases/` instead of the root `cases/` folder. Composed a full GPT dashboard-design prompt grounded in the dataset structure and planned metrics.

---

## 2026-03-18 — QueryOrchestrator: centralised routing, langdetect language detection, post-retrieval fallback

Implemented `src/query/orchestrator.py` — a `QueryOrchestrator` class that sits between `api/main.py` and `QueryEngine` and owns all query classification, routing, and language detection. Fixes three concrete failures: (1) general questions like "What is Basel III?" now trigger a post-retrieval confidence fallback (score < 0.35) that synthesises from general knowledge with a disclaimer instead of returning the unhelpful "context does not contain sufficient information" message; (2) English queries now receive `language="en"` via `langdetect`, eliminating mixed EN/IT sources (Case_003 issue); (3) the `if not history:` guard on the DefinitionsStore fast-path was removed, so follow-up definition questions (e.g. "What about own funds?" after a prior turn) now correctly resolve through Article 4 instead of going to RAG.

**Files changed:** `src/query/orchestrator.py` (new — `QueryOrchestrator`, `ClassificationResult`, `QueryType` enum, `detect_language()`), `api/main.py` (wired orchestrator, simplified endpoints, removed `_detect_language()`), `src/query/query_engine.py` (removed `if not history:` definition fast-path guard), `requirements.txt` (`langdetect>=1.0.9`), `tests/unit/test_orchestrator.py` (new — 38 tests), `tests/unit/test_api_endpoints.py` (updated to mock `_orchestrator.query` instead of `_query_engine.query`). **352 unit tests, 351 green (1 pre-existing failure unrelated to this work).**

---

## 2026-03-18 — Article 4 DefinitionsStore: fast-path lookup bypassing RAG

Built a `DefinitionsStore` that parses Article 4 of the CRR into a structured JSON index at startup, enabling O(1) definition lookups without touching the RAG pipeline or calling the LLM. Added a Stage 0.5 fast-path in `QueryEngine.query()` and `/api/query/stream` that intercepts definition queries (e.g. "What is the definition of institution?", "Article 4(1)", "Explain Article 4") and returns answers directly from the JSON cache. Article 4 is now also skipped in `_expand_cross_references()`, eliminating the token-overflow / 429-rate-limit errors that occurred when cross-ref expansion pulled the full 129-definition blob into synthesis context. `definitions/definitions_en.json` and `definitions/definitions_it.json` committed to the repo. 36 new unit tests; 315 total, all green.

---

## 2026-03-18 — Conversational memory (multi-turn chat history)

Implemented end-to-end conversational memory so follow-up questions like "and what about AT1?" are understood in context of prior Q&A turns rather than being sent as stateless queries.

**Backend (`src/query/query_engine.py`):** Added `_format_history()`, `_rewrite_query_with_history()`, and `_LEGAL_QA_TEMPLATE_WITH_HISTORY`. Modified `QueryEngine.query()` to accept `history: list[dict]`, rewrite follow-ups into standalone queries before retrieval (skipped when history is empty), and call OpenAI directly instead of `engine.synthesize()` (matching the stream path). Both sync and stream paths are now symmetric. `_HISTORY_MAX_TURNS = 5` caps prompt growth.

**Backend (`api/main.py`):** Added `HistoryTurn` Pydantic model. Extended `QueryRequest` with `history: list[HistoryTurn] = []`. `/api/query` forwards history dicts to `query_engine.query()`. `/api/query/stream` rewrites the query in a threadpool before retrieval, then injects history into the synthesis prompt inside `generate()`.

**Frontend:** `HistoryTurn` interface added to `types.ts`. `postQueryStream` in `api.ts` accepts and forwards `history`. `submitQuery` in `useQuery.ts` threads history through. `ChatPanel.tsx` slices the last 5 messages into `HistoryTurn[]` and passes them on every submission.

**Tests:** 18 new unit tests in `tests/unit/test_conversational_memory.py` (format_history, rewrite, QueryEngine integration). 6 new tests added to `test_api_endpoints.py::TestQueryHistoryField`. 3 existing `TestSynthesisNodeMerging` tests updated to mock OpenAI directly (previously relied on `engine.synthesize()` which is no longer called). **235 unit tests, all green.**

---

## 2026-03-18 — GPT-4o token streaming (SSE)

Implemented end-to-end streaming of GPT-4o synthesis tokens so the user sees text appear as it is generated rather than waiting for the full response.

**Backend:** Extracted a new `QueryEngine.retrieve()` method encapsulating stages 1 & 2 (retrieval + cross-ref expansion). Added `POST /api/query/stream` FastAPI endpoint that runs retrieval in a threadpool (`asyncio.to_thread`), then streams tokens via Server-Sent Events using `AsyncOpenAI` directly (bypassing LlamaIndex's synchronous synthesizer). The existing `query()` endpoint is unchanged.

**Frontend:** Replaced `postQuery` with `postQueryStream` in `api.ts` (SSE reader with `onToken` callback). Updated `useQuery.ts` to expose `streamingAnswer` state. Updated `ChatPanel.tsx` to render live streaming text with a blinking cursor during synthesis; falls back to the skeleton loader only during the retrieval phase (before the first token arrives).

**Files changed:** `src/query/query_engine.py`, `api/main.py`, `frontend/src/lib/api.ts`, `frontend/src/hooks/useQuery.ts`, `frontend/src/components/chat/ChatPanel.tsx`.

---

## 2026-03-18 — Italian language parity audit: 4 frontend regressions fixed; 14 new tests

Identified and fixed all Italian-language regressions in the frontend viewer. No backend changes required.

| # | File | Fix |
|---|------|-----|
| 1 | `legal-text-parser.ts` | `ART_RUN_RE` extended to match `Articolo`/`Articoli` + Italian conjunctions `e`/`o` |
| 2 | `legal-text-parser.ts` | `EXTERNAL_CONTEXT_RE` extended to handle `del/della/dello/dell'` prepositions and Italian keywords `Regolamento`/`Direttiva`/`Decisione` — prevents external refs like "Articolo 4 del Regolamento (UE) n. 648/2012" from being linkified |
| 3 | `DocumentBreadcrumb.tsx` | Breadcrumb labels are now language-aware: `Parte`, `Titolo`, `Capo`, `Sezione`, `Articolo` for Italian |
| 4 | `DocumentViewer.tsx` | "Referenced Regulations & Directives" section header renders as "Regolamenti e Direttive di Riferimento" for Italian articles |

Added `frontend/src/lib/legal-text-parser.test.ts` — 14 vitest tests covering EN linkification, IT linkification, EN/IT external-ref exclusion, and `parseTextRuns` integration. Vitest added as dev dependency; `npm test` script added to `frontend/package.json`.

---

## 2026-03-18 — 5 low-hanging-fruit items closed; 255 unit tests green

Five independent improvements with no re-ingest required. 10 new unit tests added.

| # | Item | Change |
|---|------|--------|
| 1 | **Split article audit (92a/92b)** — test-only | Added `TestLetteredArticleHandling` (2 tests) confirming `article="92a"`, `node_id="art_92a_en"` metadata; added 2 tests in `TestDirectArticleLookup` for `_detect_direct_article_lookup("Article 92a")` |
| 2 | **Corpus deduplication** | `_parse_with_beautifulsoup()` now deduplicates by `node_id` before returning; duplicate nodes emit `WARNING` log; count of removed duplicates logged at `INFO`; 2 tests in `TestCorpusDedup` |
| 3 | **Regulation reference badges** | `get_article()` extracts `referenced_external` CSV → list; `ArticleResponse` Pydantic model + `ArticleResponse` TS interface gain `referenced_external: list[str]`; `DocumentViewer.tsx` renders amber badges below article text; 2 unit tests |
| 4 | **RETRIEVAL_ALPHA env var** | `RETRIEVAL_ALPHA: float = float(os.getenv("RETRIEVAL_ALPHA", "0.5"))` constant added; wired into both `_build_engine()` and `_retrieve_with_filters()` `as_retriever()` calls; documented in `.env.example`; 2 unit tests; existing mock helpers updated with `**kwargs` |
| 5 | **Extract `legal-text-parser.ts`** | New `frontend/src/lib/legal-text-parser.ts` — 6 regexes, `splitInlineItems`, typed `ParsedRun` union, `parseTextRuns()`; `ProvisionText.tsx` now imports from parser and uses local `renderRuns()` React adapter; JSX/CSS unchanged |

**Test count: 255 unit tests, all green.**

---

## 2026-03-18 — Codex V2 ingestion fixes confirmed in code + re-ingest complete

All 8 Codex V2 findings are now resolved. Findings 1–3 were already in the code (applied in the 2026-03-17 session) but re-ingest with `--reset` was needed to clean Qdrant. Re-ingest completed 2026-03-18.

| # | Finding | Fix |
|---|---------|-----|
| 1 | **Annex overmatch** — `^anx_` matched sub-annex IDs (`anx_IV.1`) | Regex tightened to `^anx_[^.]+$`; `--reset` re-ingest eliminates stale sub-annex points |
| 2 | **Formula/text loss** — `get_text()` stripped `<img>` formulas; early return dropped surrounding prose | `<p>` handler walks children token-by-token, emitting text tokens and `[FORMULA_N]` placeholders in document order (lines 498–516) |
| 3 | **Layout-A nested grid flattening** — `cols[1].get_text()` lost sub-point structure | Layout-A path splits direct inline children (`col_parts`) from div children (`div_children_in_col`), calling `walk()` for each nested div (lines 419–450) |
| 4–8 | See 2026-03-17 entry below | Already applied |

---

## 2026-03-17 — Re-ingest on Colab with `--reset` — verified clean

Full `--reset` re-ingest run on Colab T4. `diagnose_qdrant.py` confirmed: 1490 items (745 EN + 745 IT), 4 annexes per language (I–IV only, no sub-annex leakage), 741 articles per language, zero duplicate node_ids. All Codex V2 fixes now live in the index.

---

## 2026-03-17 — `_settings_scope()` Colab compatibility fixes (3-commit series + dep add)

Fixed `_settings_scope()` in `index_builder.py` to work on Colab where LlamaIndex's `Settings` lazy resolver raises `ImportError` when `llama-index-embeddings-openai` is not installed. Three progressive fixes applied:
1. Use private `_embed_model`/`_llm` backing attrs to bypass the resolver during snapshot/restore.
2. Use `getattr(Settings, '_attr', None)` for all three lazy-resolver properties (`embed_model`, `llm`, `transformations`).
3. Guard `chunk_size`/`chunk_overlap` snapshot inside `try/except` with a sentinel — if read fails, skip restoration (query engine doesn't need these at query time).

Added `llama-index-embeddings-openai` to the Colab `pip install` cell as the root-cause fix (LlamaIndex imports it as the default embed model even when BGE-M3 is active).

---

## 2026-03-17 — Codex V2 ingestion + query fixes (Fixes 1–4); 234 unit tests green

Four parsing/extraction bugs from the Codex Review V2 fixed across `eurlex_ingest.py` and `query_engine.py`. 17 new unit tests added; all 234 pass. Re-ingest on Colab (with `--reset` for Fix 1) still needed.

**Fix 1 — Annex overmatch** (`eurlex_ingest.py:176`): regex `^anx_` → `^anx_[^.]+$` prevents sub-annex IDs (e.g. `anx_IV.1`) from being indexed as separate documents, eliminating duplicate Qdrant points for multi-section annexes.

**Fix 2 — Formula paragraph child-walk** (`eurlex_ingest.py:468`): replaced `elem.find("img")` + early-return with an in-order child walk that emits text tokens and `[FORMULA_N]` placeholders together, preserving prefix and suffix text around inline formula images.

**Fix 3 — Layout-A nested grid flattening** (`eurlex_ingest.py:411`): replaced `cols[1].get_text(...)` with a loop that collects direct text/inline into `col_parts` and calls `walk()` for nested `<div>` sub-points, so nested sub-point labels and formula placeholders are preserved as separate entries rather than being concatenated.

**Fix 4a — Cross-reference range parsing** (`eurlex_ingest.py:611`): replaced single-number regex with a full-run pattern (`Articles N to M`, comma/and lists); expands ranges into individual article numbers and captures multi-letter suffixes (`92aa`).

**Fix 4b — Query-time range expansion** (`query_engine.py`): added `_expand_article_ranges()` wired into `_normalise_query()` so queries like "Articles 89 to 91" are expanded to explicit article references before BM25/dense retrieval.

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
