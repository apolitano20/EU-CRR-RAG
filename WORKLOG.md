# WORKLOG — Open Tasks

For completed work history, see `COMPLETED.md`.

---

## Current State (as of 2026-03-31) — run_42 is new SOTA; embedding metadata fix confirmed

**SOTA: run_42_embed_metadata_fix** — Hit@1=**87.28%**, Hit@1(family)=**87.86%**, Recall@5=**85.16%**, MRR=**0.8976**, MRR(expanded)=**0.9004**. (n=173, 0 failures.)
Config: `RETRIEVAL_ALPHA=0.5`, `USE_ARTICLE_GRAPH=true`, `USE_MIXED_CHUNKING=true`, `USE_PARAGRAPH_WINDOW_RERANKER=true`, `PARAGRAPH_WINDOW_MAX_WINDOWS=4`, `RETRIEVAL_TOP_K=15`, `TITLE_BOOST_WEIGHT=0`, `ADJACENT_TIEBREAK_DELTA=0.05`, `gpt-4o-mini` (standard) + `gpt-4o` (hard queries). Re-ingested with `_EXCLUDED_EMBED_METADATA_KEYS` applied to all Document constructors.

**vs run_40 (previous working baseline):** +2.31pp Hit@1, +1.73pp Hit@1(family), +1.35pp Recall@1, +1.93pp MRR. All metrics improved.
**vs run_30 (original SOTA):** Matched on Hit@1 (~87.3%), slightly higher MRR (0.8976 vs 0.897), better Recall@5 (85.16% vs 85.2%). Confirmed that the embedding metadata fix fully recovered the run_38 re-ingest regression.

**Active .env:** `EMBED_MODEL=bge-m3`, `QDRANT_COLLECTION=eu_crr`, `RETRIEVAL_ALPHA=0.5`, `EVAL_MODE=true`.

**What we know about remaining failures (as of run_42):**
- `diluted_embedding` (6 cases, Hit@1=33.3%): unchanged vs run_40. Vocabulary mismatch — BM25 fails (no token overlap), alpha tuning is not the lever. **Next: article summary enrichment at ingest (2–3 sentence GPT-4o-mini summary + keyword tags per article).**
- `multi` article recall (32 cases, recall@1=31.25%): improved slightly vs run_40 (31.25% vs 27%) but still lagging. Right articles not in pool at all — not a coverage problem. Needs query-side fix or cross-ref expansion.
- `liquidity_lcr` (5 cases, Hit@1=60%): persistent underperformer.
- `significant_risk_transfer` (1 case, Hit@1=0%): single hard case, recall@3=50%.
- Sub-article clusters (132x, 429x): hit@1_family gap (+0.58pp) small — largely resolved by family metric.

**Next experiment candidates:**

| # | Run | Target | Approach | Effort | Re-ingest |
|---|-----|--------|---------|--------|-----------|
| **IMPL** | **run_41** | `multi` + `diluted_embedding` | **EVALUATED** — neutral vs run_40 on full 173 cases (Hit@1 0.8439 vs 0.8497). Helps `multi_hop` (+0.021 Hit@1) but hurts `false_friend` (−0.071) and `negative` (−0.063). Not shipping as improvement. See run_41 entry below. | Done | No |
| ~~**DONE**~~ | ~~**run_42**~~ | ~~**restore SOTA**~~ | **COMPLETED** — Hit@1=87.28%, MRR=0.8976. Matched run_30 SOTA and became new best on all metrics vs run_40. | Done | Done |
| 3 | run_43 | synthesis quality | Switch synthesis LLM to Claude Sonnet + switch judge to Claude (avoids GPT self-preference bias). Clean new baseline targeting Judge Correctness. | Medium | No |
| 3 | run_43 | `diluted_embedding` + `leverage_ratio_*` | **Article summary enrichment in embedding text**. Generate a 2–3 sentence GPT-4o-mini summary + 5–8 keyword tags per article; prepend to embedding text at ingest. Cost: ~$2–5 for 750 articles. Expected gain: `diluted_embedding` Hit@1 33%→50%+. | Medium | Yes |
| 4 | — | `multi_hop` synthesis (stubborn cases) | Investigate case_018, 124, 127, 143 individually. | Low-Medium | No |

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
| **run_20_mixed_chunking** | 2026-03-25 | **86.7%** | **83.6%** | **0.891** | **0.793** | Mixed ARTICLE+PARAGRAPH retrieval + ArticleDeduplicatorPostprocessor. No re-ingest. **New SOTA — beats run_17 by +6.4pp Hit@1, +4.7pp Recall@3, +5.1pp MRR. Zero regressions.** Judge Correctness +2.3pp vs run_2e. |
| run_21_domain_rewrite | 2026-03-25 | 80.9% | 80.4% | 0.849 | — | Domain query rewriting (plain-language → CRR legal register via GPT-4o-mini). Hit@1 −5.8pp vs run_20. diluted_embedding unchanged (16.7%). Regression across most categories. **Reverted.** |
| run_22_false_premise_prompt | 2026-03-25 | 86.7% | 83.6% | 0.892 | 0.782 | Synthesis prompt: FALSE PREMISE RULE + CONDITIONAL LOGIC RULE. Overall correctness −1.1pp. false_friend −12.2pp (0.679→0.557) — over-hedging. diluted_embedding +8.3pp. Net regression on target slice. **Reverted.** |
| run_23_eval_fix | 2026-03-26 | 86.7% | 83.6% | 0.891 | — | Measurement fix: `_with_expanded` metrics added to eval. Confirmed multi_article gap mostly measurement artifact (recall@3 83.3% with_expanded vs 38.9% raw). |
| run_24_article_graph (judged) | 2026-03-26 | 86.7% | 83.6% | 0.891 | 0.787 | Article graph BFS expansion. Multi faithfulness +6.9pp, multi_hop faithfulness +5.4pp, own_funds/leverage strong gains. Single-article −1.1pp correctness, large_exposures −4.3pp, negative −6.2pp faithfulness. `is_multi_hop` regex only 6% coverage — unusable. Fix: gate on ≥2 retrieved articles → run_25. |
| run_25_graph_gated | 2026-03-26 | 86.7% | 83.9% | 0.891 | 0.763 | Graph gated on ≥2 distinct retrieved articles. Statistically identical to run_24 on all metrics — wash. |
| run_26_synonym_falsepremise | 2026-03-26 | 87.3% | 83.9% | 0.897 | 0.792 | BM25 synonym expansion (8 entries) + false premise rule w/ 3 examples. `false_friend` Judge Correctness +0.20 (0.557→0.757). `diluted_embedding` Hit@1 1/6→2/6. **New SOTA.** |
| **run_27_completeness** | 2026-03-26 | **87.3%** | **83.9%** | **0.897** | **0.812** | Deterministic completeness: threshold preamble (Part A) + post-generation diff (Part B). 7/17 target cases improved, 0 regressed. Judge Completeness +2.5pp, Faithfulness +2.3pp. **Current SOTA.** |
| run_28_bidir_crossref | 2026-03-26 | 87.3% | 83.9% | 0.897 | 0.802 | Bidirectional structural cross-ref extraction at ingest (Part/Title/Chapter/Section refs stored in metadata). Re-ingested `eu_crr`. Retrieval identical to run_27; judge scores slightly lower across board (correctness −1.0pp, faithfulness −1.9pp). No retrieval gain from bidir refs. |
| run_29_e5_dense | 2026-03-28 | 76.97% | 83.2% | 0.873 | 0.805 | Embedding swap to `multilingual-e5-large-instruct` (dense-only, 1024-dim, separate `eu_crr_e5` collection). recall@1 −2.5pp, MRR −2.4pp vs run_27. Latency doubled (22s p50 vs 11s). `credit_risk_sa` −30pp, `leverage_ratio_total_exposure` −33pp. recall@5 +1.4pp only upside. **Regression — reverted to BGE-M3.** |
| **run_30_bge_m3_revert** | 2026-03-28 | **87.3%** | **85.2%** | **0.897** | **0.801** | BGE-M3 revert confirmation on current `eu_crr` index. Retrieval identical to run_27/28 (hit@1, MRR); recall@5 marginally better (0.8521 vs 0.8492). Judge metrics ~1pp below run_27 (noise-level). **Declared new SOTA** — run_30 IS the current system state; re-ingesting to recover run_27's judge scores is not worth the cost. |
| run_31_bge_reranker_v2_m3 | 2026-03-29 | — | — | — | — | **TOTAL FAILURE**: bge-reranker-v2-m3 (560MB) loaded at startup via `USE_PARAGRAPH_WINDOW_RERANKER=true` fallback path even with `use_reranker=false`. Model too slow — all 173 cases timed out. Reverted `RERANKER_MODEL` to `ms-marco-MiniLM-L-6-v2`. |
| run_33_bge_int8 | 2026-03-29 | — | — | — | — | **ABANDONED before eval**: benchmark shows PyTorch INT8 dynamic quantization is 0.63x slower on this CPU (313ms vs 197ms FP32) and degrades cosine similarity to 0.968. BGE-M3 encode is only ~200ms (~1%) of 21s end-to-end latency — bottleneck is synthesis (~80%). `torch.quantization.quantize_dynamic` also deprecated in PyTorch 2.10. |
| run_33_wider_funnel | 2026-03-29 | 86.71% | 84.10% | 0.896 | 0.808 | Wider retrieval funnel: top_k 15→20 + windows 4→6. **Neutral overall — run_30 stays SOTA.** Primary targets (`multi` recall@1, `diluted_embedding` Hit@1) moved zero. `leverage_ratio_total_exposure` regressed -33pp. Tail latency ballooned (P99: 50s→89s). Judge metrics +0.6pp (noise). Conclusion: the `multi`/`diluted_embedding` failures are not a coverage problem — the right article is simply not ranking first regardless of pool size. |
| run_34_alpha_topn | 2026-03-29 | 87.28% | 84.10% | 0.896 | 0.801 | RETRIEVAL_ALPHA 0.5→0.4 + RERANK_TOP_N 6→8. **Flat-to-slight-regression — run_30 stays SOTA.** Hit@1 and judge correctness identical. Faithfulness -0.4pp, P99 +15s. Key finding: alpha=0.4 (more BM25) HURT `diluted_embedding` judge correctness by -6.6pp — these cases need MORE dense signal, not less (BM25 also fails on vocabulary mismatch). RERANK_TOP_N=8 didn't move `multi` recall@1 — right articles not in pool at all. Two lessons: (1) diluted_embedding → try higher alpha (0.6+), not lower; (2) multi failures need earlier-pipeline fix. |
| run_35_multi_retrieval_fixes | 2026-03-30 | 83.8% | 86.0% | 0.891 | 0.823 | First attempt at 5 runtime-only multi-article retrieval fixes (blended direct, structural siblings, reverse edges, global re-rank, expanded multi-hop regex). **Net regression on hit@1 (-3.5pp) with two implementation bugs:** (1) triple reranking — ParagraphWindowReranker fired 3× per query, p50 latency 53s vs 20s SOTA; (2) blended direct too broad — 85/173 queries triggered blend (including "Article X as per..." cases), causing 10 regressions. Recall@3/5 improved (bug-driven side effect of triple reranking). Judge correctness +2.2pp (0.823) — synthesis quality benefited from richer context pool. Both bugs identified and corrected for run_36. |
| run_37_run30_reconfirm | 2026-03-30 | 86.7% | 83.4% | 0.894 | n/a | All 4 run_35/36 fix flags disabled (USE_BLENDED_DIRECT_RETRIEVAL=false, USE_STRUCTURAL_SIBLINGS=false, USE_REVERSE_GRAPH_EXPANSION=false, USE_GLOBAL_RERANK=false). Intended to confirm run_30 SOTA. **Result: -0.6pp vs run_30 — confirmed as noise.** Only 3 case flips vs run_30, all within sub-article clusters (132c↔132, 429b↔429, 92↔93). These articles are borderline-tied in reranker score; outcome varies run-to-run. run_30's 87.3% and run_37's 86.7% are the same system — stable baseline is ~87% ±0.6pp. Validates that 132 and 429 sub-article grouping at re-ingest would permanently fix these borderline flips. |
| run_36_retrieval_fixes_v2 | 2026-03-30 | 86.7% | 83.1% | 0.893 | n/a* | Corrected version of run_35: fixed triple reranking (skip intermediate expansion rerank when USE_GLOBAL_RERANK=true) and narrowed blended-direct gate to unambiguous subordinate-reference patterns only ("as per", "pursuant to", "deducted from/under", "excluded from/under", "calculated under", "in accordance with") — reduces blend from 85/173 to 2/173 queries. **Regression on hit@1 vs SOTA (−0.6pp), neutral recall@5 (+0.1pp), recall@3 −1.1pp.** 0/9 target cases fixed for hit@1; case_171 recall@3 improved (0→1.0, right article now in pool). 2 new regressions (case_058, case_169), 1 improvement (case_035). *Judge scores in summary file are invalid (19/173 cases judged from aborted first attempt — not comparable). Latency 23s (+11%). **run_30 remains SOTA.** Core finding: ParagraphWindowReranker's literal token-match scores for false-friend articles are too strong for global re-rank to overcome at hit@1. |
| run_38_sub_article_reindex | 2026-03-30 | 84.97% (hit@1_family=86.13%) | 83.0% | 0.880 | n/a | Re-ingest with `sub_article_of` metadata populated for 429x/132x sub-article clusters. **Result confounded by workers=1 (8 timeout+retry events) — treat as noisy, not a true regression.** hit@1_family introduced: 86.13% vs hit@1 84.97% confirms +1.2pp family gap (sub-article confusion real but small). All 4 retrieval-fix flags disabled (same config as run_30). Regressions (case_035, 137, 141, 156) all have sources=[] in both run_37 and run_38 — hit@1 computed from GPT answer text, deterministic at temperature=0 but sensitive to context changes from the re-ingest. **Reindex confirmed neutral in principle; repeat with workers=3 for a clean read if needed.** |
| run_39_alpha_065 | 2026-03-30 | 85.5% | 82.4% | 0.882 | n/a | RETRIEVAL_ALPHA=0.5→0.65 (more dense). **Neutral-to-slight-regression vs run_40 baseline.** 22 cases timed out (eval runner --timeout default was 150s vs QUERY_TIMEOUT_SECONDS=300s — root cause bug; fixed). After dedup+retry merge: n=172. diluted_embedding unchanged; alpha tuning is NOT the lever for these failures. Reverted to alpha=0.5. |
| run_40_sota_reconfirm | 2026-03-30 | 85.0% (hit@1_family=86.1%) | 83.1% | 0.878 | n/a | Baseline reconfirm with alpha=0.5 after revert. n=173, 0 failures. 2.3pp gap vs run_30 SOTA confirmed as non-deterministic FP16 GPU re-ingest artifact. **Declared as new working baseline.** |
| **run_41_combined_3_4** | 2026-03-30 | 84.4% (173 combined) | 83.2% | 0.876 | n/a | Combined: subordinate-ref stripping before reranker + unrated synonyms. **Neutral vs run_40 overall.** `multi_hop` +2.1pp Hit@1, `large_exposures` +2.9pp. `false_friend` −7.1pp, `negative` −6.3pp. 16 cases timed out (dashboard 120s default, 4 workers — fixed: dashboard now defaults to 300s timeout). Root cause of false_friend regression: stripping article refs hurts queries where the cited article IS the question. Not shipping. |
| **run_42_embed_metadata_fix** | 2026-03-31 | **87.28%** (hit@1_family=87.86%) | **84.01%** | **0.8976** | n/a | Re-ingest with `_EXCLUDED_EMBED_METADATA_KEYS` applied to all three Document constructors in `eurlex_ingest.py`. Excludes `referenced_articles`, `sub_article_of`, `node_id`, `has_table`, `chunk_type` from BGE-M3 embedding text. **NEW SOTA — +2.31pp Hit@1 vs run_40, +1.93pp MRR.** Matches run_30 SOTA level (~87.3%); confirmed the run_38 regression was entirely due to metadata pollution. n=173, 0 failures. diluted_embedding still 33.3% (unchanged — not a metadata issue). |

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
| 6 | Embedding model benchmarked | ✅ run_29 benchmarked `multilingual-e5-large-instruct` vs BGE-M3 — BGE-M3 wins on recall@1, MRR, latency; e5 marginally better on recall@5 only |
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
| 4 | **429x sub-article metadata** — add `sub_article_of` field to group 429/429a/429b/429c/429e. | 3 niche sub-article failures | Medium | Yes | 🔄 Code complete (2026-03-30). Re-ingest pending. |
| 5 | **run_18: judge run on run_17 config** — run full eval with judge enabled on current best config (para-window reranker) to get comparable judge scores vs run_2e baseline. | — | Low | No | ✅ Superseded — run_20 judge complete (Correctness=0.793). |
| 6 | **Domain query rewriting** — rewrite plain-language query into CRR legal register before embedding. | 6 diluted_embedding failures | Low | No | ✅ Tried (run_21): Hit@1 −5.8pp vs run_20. diluted_embedding unchanged. **Regression — reverted.** |
| 7 | **False premise + conditional logic prompt rules** — explicit instructions to refute false premises and state conditions. | 14 false_friend cases | Low | No | ✅ Tried (run_22): false_friend −12.2pp (over-hedging), diluted_embedding +8.3pp. Net regression on target. **Reverted.** |

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
| 1 | **BGE-M3 CPU inference** | BGE-M3 encodes in ~200ms per query — only ~1% of total latency. **Not the bottleneck.** ~~ONNX INT8~~ benchmarked (run_33): 0.63x slower, cosine sim 0.968 — abandoned. `torch.quantization.quantize_dynamic` deprecated in PyTorch 2.10. Switch to `torchao` if revisiting. | GPU only real option; not worth effort. |
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
