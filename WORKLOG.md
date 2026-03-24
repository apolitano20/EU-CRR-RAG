# WORKLOG — Open Tasks

For completed work history, see `COMPLETED.md`.

---

## Current State (as of 2026-03-25) ✅ MIXED CHUNKING — run_20 is new SOTA | branch: ingest_contextual_prefix → merging to main

Best eval run: **run_20_mixed_chunking** — Hit@1=**86.7%**, Recall@3=83.6%, MRR=0.891, mean latency ~10.5s (no judge). **New SOTA.**
Best run with judge: **run_2e_baseline** — Hit@1=76.3%, MRR=0.815, Judge Correctness=0.770, Judge Faithfulness=0.790. (judge run on run_20 config pending)
Config: `USE_MIXED_CHUNKING=true`, `USE_PARAGRAPH_WINDOW_RERANKER=true`, `PARAGRAPH_WINDOW_MAX_WINDOWS=4`, `cross-encoder/ms-marco-MiniLM-L-6-v2`, top_n=6, `RETRIEVAL_TOP_K=15`, `RETRIEVAL_ALPHA=0.5`, `TITLE_BOOST_WEIGHT=0`, `ADJACENT_TIEBREAK_DELTA=0.05`, `USE_TOC_ROUTING=false`, `gpt-4o-mini` (standard) + `gpt-4o` (multi-hop via orchestrator).

**Next experiment candidates:**
- run_20 judge — enable `--judge` for comparable judge scores vs run_2e baseline (Judge Correctness=0.770).
- Diluted embedding fix — remaining weak spot (hit@1=0.17 unchanged across all runs). Needs HyDE or alternative approach.

---

## Experiment Log

All runs on 173-case golden dataset, judge enabled (gpt-4o).

| Run | Date | Hit@1 | Recall@3 | MRR | Judge Correct. | Key change vs previous |
|-----|------|-------|----------|-----|----------------|------------------------|
| baseline | 2026-03-20 | 81.3% | — | — | 0.774 | Initial baseline (phases 0/2/3: orchestrator, query normalisation, cross-ref expansion) |
| run_1_20032025 | 2026-03-20 | 78.5% | — | — | 0.782 | Sub-query fusion, HyDE pre-retrieval enrichment — net regression on retrieval, slight judge gain |
| phase1_reranker | 2026-03-20 | — | — | — | — | Intermediate reranker test (partial run) |
| run_2_20032025 | 2026-03-20 | **79.2%** | 82.1% | 0.829 | 0.769 | Pure reranker (`ms-marco-MiniLM-L-6-v2`, top_n=6) — **best retrieval config found** |
| run_3_blend_ranker | 2026-03-20 | ~79% | — | — | — | Blended reranker alpha=0.3 — net neutral vs run_2, reverted |
| run_3_blend_ranker2 | 2026-03-20 | ~79% | — | — | — | Blended reranker alpha=0.6 — net neutral vs run_2, reverted |
| run_4_HyDE | 2026-03-20 | 79.2% | 81.6% | 0.828 | 0.768 | HyDE query enrichment — neutral overall, kept in codebase |
| run_4_HyDe_numHint | 2026-03-20 | ~79% | — | — | — | HyDE + numeric hint variant — no clear gain |
| run_5_title_boost | 2026-03-21 | 78.6% | 81.6% | 0.824 | 0.757 | Article title boost (`TITLE_BOOST_WEIGHT=0.15`) — hurt across the board, reverted to 0 |
| run_2b_20032025 | 2026-03-21 | **79.2%** | 81.8% | 0.829 | 0.760 | Re-run of run_2 config on updated dataset (2 cases fixed); `TOP_K=15`, orchestrator+gpt-4o for multi-hop. New baseline. |
| run_6_gpt4o | 2026-03-21 | 79.2% | 80.9% | 0.826 | 0.772 | `OPENAI_MODEL=gpt-4o` for all queries — judge correctness +1.2pp, faithfulness +2.6pp, but latency 25s (2.3x). Not worth it. Reverted. Confirms bottleneck is retrieval, not synthesis. |
| run_9_params_tuning | 2026-03-23 | 78.6% | 81.8% | 0.824 | 0.779 | ALPHA=0.7, TOP_K=20, RERANK_TOP_N=8 — neutral retrieval, +1.9pp judge correctness. Hypothesis not confirmed; baseline params retained. |
| run_2d_baseline | 2026-03-23 | 76.9% | 79.8% | 0.818 | — | Post code-review baseline (no judge); 4 failures from workers=4 timeout race. |
| run_10_synonyms | 2026-03-23 | 77.5% | 79.1% | 0.822 | 0.766 | Broad synonyms regressed (rolled back); surgical-only kept. Confounded by old 120s timeout. |
| **run_2e_baseline** | 2026-03-24 | **76.3%** | 79.3% | 0.815 | **0.770** | Clean reference baseline: workers=1 (0 failures), judge enabled, EVAL_MODE=true. Confirmed surgical synonyms already active. |
| **run_12_adj_tiebreaker** | 2026-03-24 | **78.0%** | 79.2% | **0.824** | — | Adjacent article tiebreaker (delta=0.05): +1.7pp Hit@1, false_friend +14.3pp. |
| run_15_toc2 | 2026-03-24 | 76.3% | 80.2% | 0.824 | — | Universal ToC routing (parallel, timeout-fixed): regression overall (-1.7pp Hit@1, +30% latency). Gains for multi-article (+9.4pp), false_friend (+7.1pp), diluted_embedding (+37.5pp Recall@3). Losses in liquidity (-7.4pp), own_funds (-6.5pp), threshold (-7.1pp). **Conclusion: selective routing needed.** |
| run_16_toc_confidence | 2026-03-24 | 77.5% | 79.2% | 0.821 | — | Selective ToC routing (fires when max reranker score < 0.55): still a net regression vs run_12 (-0.5pp). ToC routing retired. |
| **run_17_para_window** | 2026-03-24 | **80.3%** | 78.9% | **0.840** | — | Paragraph-window reranker: +2.3pp Hit@1 vs run_12. false_friend +14.3pp, open_ended +4.1pp, negative +12.5pp. diluted_embedding -16.7pp (6 cases only). +800ms latency. **Current best.** |
| run_18_contextual_prefix | 2026-03-24 | 80.3% | 77.8% | 0.835 | — | Contextual hierarchy prefix in embedding text (Codex rank 2). Neutral overall vs run_17. ciu_treatment -20pp, known_failures -8.3pp. capital_ratios +3.2pp, own_funds +3.2pp. +1s latency. Prefix on full-article blobs insufficient — gains gated on paragraph chunking. |
| run_19_para_chunking | 2026-03-24 | 70.5% | 69.6% | 0.733 | — | Para-chunked dual-doc index, PARAGRAPH-only retrieval. Regression overall — top-k flooding. Gains on localized queries (credit_risk_sa +40pp, cash_pooling +100pp). Led to mixed-chunking insight. |
| **run_20_mixed_chunking** | 2026-03-25 | **86.7%** | **83.6%** | **0.891** | — | Mixed ARTICLE+PARAGRAPH retrieval + ArticleDeduplicatorPostprocessor. No re-ingest. **New SOTA — beats run_17 by +6.4pp Hit@1, +4.7pp Recall@3, +5.1pp MRR. Zero regressions.** |

Codex Dashboard Review 2 hang fixes applied 2026-03-20: eval runner `as_completed` hang fixed (replaced with `wait()` polling loop + `shutdown(wait=False)`); `/api/query` converted to async with `asyncio.to_thread` + `asyncio.wait_for` (504 on timeout); BGE-M3 `_encode_lock` added to serialize CPU encodes; `--auto-start-api` stdout pipe bug fixed (DEVNULL).

Qdrant collection rebuilt clean on 2026-03-18 via Colab T4 (re-ingest with `--reset` after Codex V2 fixes).
Smoke test passed — Article 92 query returns correct answer with proper citations.
Git consolidated to single `main` branch (deleted local + remote `master`).
`definitions/definitions_en.json` and `definitions/definitions_it.json` committed.

---

## Minimum Checklist Scorecard

| # | Requirement | Status |
|---|-------------|--------|
| 1 | Parser preserves hierarchy, headings, tables | ✅ |
| 2 | Legal-structure-aware chunking | ✅ |
| 3 | Metadata used for filtering | ✅ |
| 4 | Hybrid retrieval | ✅ |
| 5 | Reranker | ✅ (opt-in via `USE_RERANKER`) |
| 6 | Embedding model benchmarked | ❌ Reasonable choice, no benchmark |
| 7 | Retrieval + answer quality measured separately | ✅ Baseline eval run completed 2026-03-20: 173 cases, Hit@1=81.3%, Judge Correctness=0.774 |
| 8 | Smallest useful grounded context | ✅ (top_k=6, cutoff=0.3) |
| 9 | Citations + explicit uncertainty | ✅ |
| 10 | Structured answer format | ✅ |
| 11 | Index updates versioned/auditable | ❌ Manual WORKLOG only |
| 12 | Access control + language filtering | ✅ (language); N/A (single tenant MVP) |
| 13 | Latency/cost per stage | ✅ Per-stage timing: `t_retrieval`, `t_expansion`, `t_synthesis` |

**Score: 9/13 met, 1 N/A, 3 gaps.**

---

## Limitations / Trade-offs TBC

Known design constraints to revisit as the project matures.

| # | Area | Limitation | Notes |
|---|------|-----------|-------|
| 1 | **Cross-ref expansion prompt size** | ~~Expanded nodes appended without re-ranking.~~ ✅ **Partially fixed (2026-03-24):** expansion now limited to top-2 primary nodes and expanded nodes are jointly reranked before context assembly. Remaining gap: paragraph-level citation anchors (requires re-ingest). | |
| 2 | **Language detection heuristic** | ~~`_detect_language()` uses character-set diacritics only. Italian diacritics overlap with French, Romanian, Portuguese — English queries always return `None`.~~ ✅ **Replaced with `langdetect` in `orchestrator.py`** — `detect_language()` now returns `"en"` for English queries, enabling correct language filtering. | Diacritics heuristic retained as fallback if `langdetect` fails. |
| 3 | **BGEm3Embedding on Windows (indexer)** | Prior SIGSEGV traced to simultaneous reranker + BGE-M3 loading; reranker not loaded during ingestion so risk is low. Ingestion is currently Colab-only. | Test on Windows before enabling local ingestion. |
| 4 | **`--reset` drops all languages** | `ingest_pipeline --reset` wipes the entire Qdrant collection including all languages. No per-language reset. | Tracked in backlog as `--language-only-reset`. |
| 5 | **Single-article direct lookup only** | `_detect_direct_article_lookup()` routes to metadata-filtered retrieval only when exactly one article is mentioned. Multi-article queries may miss one article if cosine scores are unbalanced. | Partially addressed by Codex V2 finding #5. |
| 6 | **No token budget management** | Total prompt tokens for synthesis are unbounded. No truncation before the OpenAI call. | Low risk at current corpus size. Add token budget cap if context window errors appear. |

---

## High Priority

### RAG Improvement Plan: 81% Hit@1 → 90%+ — IN PROGRESS

Baseline eval complete (2026-03-20). Full improvement plan in `research_docs/crr_rag_improvement_report.md`. Executing in 4 phases:

**Phases 0, 1, 2, 3: COMPLETE**
- Eval run_1 (phases 0/2/3): Hit@1=78.5%, Judge Correctness=0.782
- Eval run_2 (+ reranker): Hit@1=79.2% (+0.7pp), Recall@3=82.1%, mean latency −1170ms — **best known config**
- Blended reranker tried (alpha=0.3 and 0.6): net neutral vs run_2, reverted to pure reranker
- Reranker unblocked: switched from `bge-reranker-v2-m3` (SIGSEGV with BGE-M3) to `cross-encoder/ms-marco-MiniLM-L-6-v2` (sentence-transformers, no conflict)

**Target metrics:** Hit@1 ≥ 90%, Judge Correctness ≥ 0.85 — **current gap: Hit@1=86.7% (no judge) / 76.3% (with judge, stale), Judge Correctness=0.770 (stale — judge run on run_20 pending)**

**Next improvement candidates (see open-ended failure analysis below):**

### Open-ended query failure analysis (run_2_20032025)

Article-cited queries: **Hit@1=91.0%** — essentially solved.
Open-ended queries (no article number in question): **Hit@1=63.0%** — main gap.

Of 73 open-ended cases:
- **15 not in top-3 at all** (retrieval failures — right article not surfaced)
- **12 in top-3 but not top-1** (ranking failures — right article retrieved, wrong rank)

Root causes of the 15 retrieval failures:

| Pattern | Count | Example |
|---------|-------|---------|
| **Terminology dilution** | 6 | "TREA" → Article 92. "pledged government bonds" → Articles 411/412. Query uses plain language; CRR uses precise legal terms. |
| **Concept in unexpected article** | 5 | "profit and loss transfer agreement" → Art.28; "national structural measures procedure" → Art.395 |
| **Niche sub-articles** | 3 | "cash pooling" → Art.429b; dense 429/429a/429e cluster dilutes scores |
| **Known hard failures** | 2 | Art.153 IRB specialised lending — consistently missed across all runs |

Root causes of the 12 ranking failures:
- 7 adjacent article confusion (correct at pos 2–3, similar neighbour ranks first)
- 4 false-friend (known, accepted for now)
- 1 multi-article (partial retrieval)

### Improvement Roadmap (prioritised, no re-ingest unless noted)

| # | Fix | Targets | Effort | Re-ingest | Status |
|---|-----|---------|--------|-----------|--------|
| 1 | **HyDE** — generate a hypothetical CRR-style answer pre-retrieval; embed and retrieve against that instead of raw query. | 6 terminology failures | Medium | No | ✅ Tried (run_8): neutral retrieval, +latency. Not adopted. |
| 2 | **Article title boost** — boost matching nodes pre-rerank based on article title / query token overlap. | 4 concept-in-unexpected failures | Low | No | ✅ Tried (run_5): hurt across the board. `TITLE_BOOST_WEIGHT=0`. |
| 3 | **Adjacent article tiebreaker** — when two adjacent articles score within 0.05 of each other, prefer the one whose title more closely matches query keywords. | 7 ranking failures | Low | No | ✅ **DONE (run_12): +1.7pp Hit@1, false_friend +14.3pp. Kept.** |
| 4 | **429x sub-article metadata** — add `sub_article_of` field to group 429/429a/429b/429c/429e. | 3 niche sub-article failures | Medium | Yes | ⬜ Not started |
| 5 | **run_18: judge run on run_17 config** — run full eval with judge enabled on current best config (para-window reranker) to get comparable judge scores vs run_2e baseline. | — | Low | No | ⬜ Next up |

### Codex Architectural Review Uplift — ✅ COMPLETE (no-re-ingest phases, branch: `codex_uplift`)

Findings from `codex_review_findings.md` (2026-03-24). All no-re-ingest phases shipped and validated via run_17.

| # | Phase | Status |
|---|-------|--------|
| S | `QDRANT_COLLECTION` env var for safe A/B collection switching | ✅ Done |
| 1 | Paragraph-window reranker (`USE_PARAGRAPH_WINDOW_RERANKER=true`) | ✅ Done — **+2.3pp Hit@1** |
| 3 | Qdrant payload indexes for `part`, `title`, `chapter`, `section` | ✅ Done |
| 4 | Selective cross-ref expansion (top-2 nodes only, joint reranking) | ✅ Done |

**Remaining (requires re-ingestion):**

| # | Phase | What |
|---|-------|------|
| 2 | **Contextual embedding text** | ✅ **Done (run_18, 2026-03-24)** — implemented + re-ingested. Neutral overall; gains gated on paragraph chunking. Index retained (prefix is harmless). |
| 3 | **Paragraph/point chunking** | ✅ **Done (2026-03-24)** — implemented on `ingest_contextual_prefix`. Awaiting Colab re-ingest + run_19 eval. |

---

### Cross-reference expansion improvements [High effort] — CRUCIAL

The current cross-reference system only handles **article-to-article** references via the `referenced_articles` metadata field. Two open gaps (both require re-ingest):

| # | Gap | What's missing | Requires re-ingest? | Priority |
|---|-----|---------------|---------------------|----------|
| 1 | **Structural ref extraction** | Part/Title/Chapter/Section refs not parsed at ingest time. Refs like "Part Six", "Title VII", "Chapter 1" are ignored. | **Yes** — ingest must extract and store structural refs in metadata (e.g. `referenced_parts`, `referenced_titles`, `referenced_chapters`) | High |
| 2 | **Range ref parsing for structural refs** | `Parts Two to Five` / `Titles I to III` not handled by regex. Article ranges are covered; structural ranges are not. | **Yes** (ingest regex) | High |

**Operational note**: Both gaps require re-ingestion (metadata changes at parse time, no re-embedding needed — Qdrant supports payload updates).

---

## UI/UX — Phase 2

### Clickable hierarchy breadcrumbs [Medium effort, blocked by tree navigator API]
`DocumentBreadcrumb.tsx` renders breadcrumbs as plain text. Clicking "Chapter 6" should show all articles in that chapter. Requires `GET /api/navigator/{level}/{id}` endpoint.

### Relevant paragraph highlighting + auto-scroll [High effort]
When opening a cited article, the viewer shows the full article with no indication of which paragraph is cited. Backend would need to return the specific node/paragraph ID used in retrieval; frontend would highlight and scroll to it.

### Hover previews for article references [High effort]
Hovering over an inline "Article N" reference should show a tooltip with the article title, first paragraph, and an "Open article" button. Requires a lightweight article-preview API call on hover.

---

## Research / Architecture Enhancements

### ToC-guided LLM routing [Medium effort] — NEXT UP
Parallel retrieval path: GPT-4o-mini reasons over a cached ToC (~745 articles) to identify candidate articles, merged with vector results via RRF. Targets diluted-embedding failures (cases 006/007). Full implementation plan: **`research_docs/toc_plan.md`**.

### Extended thinking / model reasoning [Medium effort]
Leverage Claude's extended thinking API feature to improve synthesis quality for complex queries:
1. **Synthesis-time reasoning** — model reasons over retrieved articles before producing the final answer; most useful for multi-hop questions where articles cross-reference each other.
2. **Pre-retrieval query planning** — model reasons about which structural areas of the CRR to search before hitting Qdrant. Pairs naturally with ToC routing above.

Detect query complexity and conditionally invoke extended thinking only for complex cases to manage latency and cost.

---

## Efficiency Improvements

Bottleneck analysis (2026-03-18): main latency sources in order of impact.

| # | Bottleneck | Current state | Improvement options | Effort |
|---|------------|--------------|---------------------|--------|
| 1 | **BGE-M3 CPU inference** | 570MB XLM-RoBERTa running on CPU; ~5–15s per query encoding | (a) **ONNX Runtime + INT8 quantization** — 2–4x speedup, same model, no GPU needed. (b) **Switch to `bge-small-en-v1.5`** (33MB) — much faster, some quality trade-off. (c) **GPU** — 10–50x if hardware available (code already has CUDA detection). | Medium |
| 2 | **OpenAI GPT-4o synthesis** | ~~Blocking call, 5–30s; user sees nothing until complete~~ ✅ Streaming done 2026-03-18 | (a) **Switch to GPT-4o-mini** — 3–5x faster, cheaper, minimal quality loss for structured Q&A with retrieved context. | Low |
| 3 | **Qdrant cloud round-trips** | Multiple sequential calls per query (retrieval + cross-ref expansion); each ~1–3s network RTT | (a) **Local Qdrant** — eliminates network latency. (b) **Parallelise cross-ref expansion** — currently sequential; easy win with `ThreadPoolExecutor` or `asyncio.gather`. | Low |
| 4 | **Cold-start (first query)** | BGE-M3 loads from disk on first query (~25s) | Solved by GPU / ONNX above. Pre-warming at startup is an option but risks OOM if a prior process still holds the model. | — |

**Note:** C++/Rust rewrite is not the right lever here — PyTorch and the Qdrant client are already native C++/CUDA. Python overhead is negligible vs. model inference and network I/O.

**Recommended sequence:** (1) GPT-4o-mini → faster + cheaper; (2) ONNX INT8 for BGE-M3 → biggest raw speedup on CPU; (3) parallelise cross-ref expansion.

---

## Medium Priority

### Hybrid alpha tuning [Low effort] — covered by Phase 1 above
`RETRIEVAL_ALPHA` env var is now wired in (default `0.5`). Sweep values (0.3–0.7) as part of Phase 1 reranker tuning.

### Experiment tracking [Medium effort]
Add `evals/config.json` per run capturing: embed model, top_k, cutoff, alpha, prompt hash, corpus version (CELEX + date). Store alongside results for regression detection.

### Index upsert mode [Medium effort]
Add `--upsert` mode to ingest pipeline that checks existing node_ids and only re-embeds changed articles. Add `--language-only-reset` to wipe one language without affecting others.

---

## Low Priority

### Embedding model benchmark [Medium effort]
Benchmark BGE-M3 vs. alternatives (e.g. `multilingual-e5-large-instruct`, `snowflake-arctic-embed-m`) on Recall@k / MRR using golden dataset.

### Dynamic top_k [Low effort]
Tune top_k based on query complexity. Partially covered by Phase 1 tuning.

### Continuous eval [Medium effort]
Nightly eval run against golden dataset, alert on regression > 5%.

---

## Deferred / Out of Scope

### PGVector migration
Replace Qdrant Cloud with PGVector for production (self-hosted, lower latency). Not needed for MVP.

### Rate limits + retry logic for LlamaParse
Exponential backoff, max retries, partial-result handling. Only relevant when LlamaParse is used for formula enrichment at scale.

### Polish ingestion
Obtain `crr_raw_pl.html`, run `--language pl` ingest. Language config already exists; blocked on sourcing the HTML file.

### OCR / extraction error handling
N/A — source is born-digital HTML from EUR-Lex, not OCR'd PDFs.

---

## Evaluation Pipeline Design — ✅ COMPLETE

All eval components built and operational. Baseline run completed 2026-03-20. 42/173 cases manually reviewed.

```
evals/
  cases/golden_dataset.jsonl   # 173 cases (Pass 1 + Pass 2 adversarial)
  cases/manual_cases/          # 9 hand-crafted cases
  cases/review_status.json     # 42 cases reviewed
  run_eval.py                  # eval runner (parallel, --judge flag)
  judge.py                     # LLM-as-judge (gpt-4o)
  metrics.py                   # retrieval metrics
  compare.py                   # regression diff
  dashboard.py                 # Streamlit dashboard (3 pages)
  results/                     # baseline run outputs
```

### Open questions
- Remaining 131 cases not yet manually reviewed
- How to handle CRR article updates? (golden dataset must track CELEX version)
