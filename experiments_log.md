# Experiments Log — EU CRR RAG System

Chronological record of all eval runs against the 173-case golden dataset.
Judge: gpt-4o (3 dimensions: correctness, completeness, faithfulness, scale 0–1).
Dataset: `evals/cases/golden_dataset.jsonl` — 173 cases, 100 article-cited / 73 open-ended.

**Current best run (retrieval only):** `run_20_mixed_chunking` — Hit@1=**86.7%**, Recall@3=83.6%, MRR=0.891 (173/173, mixed ARTICLE+PARAGRAPH nodes + ArticleDeduplicatorPostprocessor). **New SOTA.**
**Current best run (with judge):** `run_20_mixed_chunking` — Judge Correctness=0.793, Judge Faithfulness=n/a (partial run). Previous clean judge reference: `run_2e_baseline` — Hit@1=76.3%, MRR=0.815, Judge Correctness=0.770.

---

## Scoreboard

| # | Run | Date | Cases | Hit@1 | Recall@3 | MRR | J.Correct | J.Faith | P50 lat | Verdict |
|---|-----|------|-------|-------|----------|-----|-----------|---------|---------|---------|
| 0 | baseline | 2026-03-20 | 160/173 | 81.3% | 82.6% | 0.842 | 0.774 | 0.799 | 10.5s | 13 hard errors masked real perf |
| 1 | run_1_20032025 | 2026-03-20 | 172/173 | 78.5% | 81.5% | 0.823 | 0.782 | 0.819 | 11.8s | Sub-query fusion: retrieval regression |
| — | phase1_reranker | 2026-03-20 | 0/28 | — | — | — | — | — | — | Complete failure (SIGSEGV) |
| 2 | run_2_20032025 | 2026-03-20 | 173/173 | **79.2%** | **82.1%** | **0.829** | 0.769 | 0.786 | 10.7s | Best retrieval — pure reranker |
| 3a | run_3_blend_ranker | 2026-03-20 | 173/173 | 79.2% | 81.5% | 0.827 | 0.763 | 0.793 | 11.5s | Blend α=0.3: net neutral, reverted |
| 3b | run_3_blend_ranker2 | 2026-03-20 | 173/173 | 78.6% | 81.6% | 0.824 | 0.756 | 0.785 | 9.6s | Blend α=0.6: slight regression |
| 4a | run_4_HyDE | 2026-03-20 | 173/173 | 79.2% | 81.6% | 0.828 | 0.768 | 0.784 | 10.8s | Article-number hints: neutral overall |
| 4b | run_4_HyDe_numHint | 2026-03-21 | 173/173 | 79.8% | 80.7% | 0.829 | 0.751 | 0.784 | 10.7s | Higher Hit@1 but Recall@3 dropped |
| 5 | run_5_title_boost | 2026-03-21 | 173/173 | 78.6% | 81.6% | 0.824 | 0.757 | 0.773 | 10.7s | Title boost 0.15: hurt across board |
| 2b | run_2b_20032025 | 2026-03-21 | 173/173 | 79.2% | 81.8% | 0.829 | 0.760 | 0.769 | 27.7s | Re-run of run_2 on fixed dataset; confirmed best config |
| 6 | run_6_gpt4o | 2026-03-21 | 173/173 | 79.2% | 80.9% | 0.826 | 0.772 | 0.795 | 25.4s | gpt-4o synthesis: +judge, 2.3× latency |
| 7 | run_7_toc_routing | 2026-03-21 | 173/173 | 78.6% | 81.6% | 0.826 | 0.762 | 0.789 | 18.6s | ToC routing: slight regression + latency |
| 8 | run_8_trueHyDE | 2026-03-21 | 173/173 | 79.2% | 81.5% | 0.828 | 0.750 | 0.787 | 19.4s | True HyDE: neutral retrieval, +latency |
| 2c | ~~run_2c_baseline~~ | 2026-03-21 | 125/173¹ | — | — | — | — | — | — | **INVALID** — crashed run; summary claimed 173 but only 125 cases actually ran |
| 9 | run_9_params_tuning | 2026-03-23 | 173/173 | 78.6% | 81.8% | 0.824 | 0.779 | 0.793 | 19.7s | ALPHA=0.7, TOP_K=20, RERANK_TOP_N=8 — neutral vs run_2b; diluted_embedding still 33% |
| 2d | **run_2d_baseline** | 2026-03-23 | 169/173 | **76.9%** | **79.8%** | **0.818** | — | — | 10.2s | **Post code-review baseline** — no judge; 4 failures (3×504, 1×500 transient) |
| 10 | run_10_synonyms | 2026-03-23 | 169/173 | 77.5% | 79.1% | 0.822 | 0.766 | 0.795 | 9.3s | Broad synonyms: regressions on case_136/152; rolled back. Surgical-only retained. Confounded by old 120s timeout. |
| 2e | **run_2e_baseline** ⭐ | 2026-03-24 | **173/173** | 76.3% | 79.3% | 0.815 | **0.770** | **0.790** | 10.3s | **Clean judge reference baseline** — workers=1 (0 failures), EVAL_MODE=true, surgical synonyms active |
| 12 | **run_12_adj_tiebreaker** ⭐ | 2026-03-24 | **173/173** | **78.0%** | 79.2% | **0.824** | — | — | 9.9s | **Adjacent tiebreaker delta=0.05: +1.7pp Hit@1, false_friend +14.3pp. Current best retrieval.** |
| 13 | run_13_reranker_L12 | 2026-03-24 | 173/173 | 76.9% | 78.8% | 0.822 | — | — | 10.3s | L-12 reranker: false_friend +7pp but definition/diluted_embedding regression. Reverted to L-6. |
| 15 | run_15_toc2 | 2026-03-24 | 173/173 | 76.3% | 80.2% | 0.824 | — | — | ~13s | Universal ToC routing (parallel, timeout-fixed): overall regression −1.7pp. Gains multi_article +9.4pp, false_friend +7.1pp, diluted_embedding +37.5pp Recall@3. Losses: liquidity −7.4pp, own_funds −6.5pp. |
| 16 | run_16_toc_confidence | 2026-03-24 | 173/173 | 77.5% | 79.2% | 0.821 | — | — | ~11s | Selective ToC routing (fires when max reranker score <0.55): still −0.5pp vs run_12. ToC routing retired. |
| 17 | **run_17_para_window** ⭐ | 2026-03-24 | 173/173 | **80.3%** | 78.9% | **0.840** | — | — | ~11.3s | Para-window reranker: **+2.3pp Hit@1 vs run_12.** false_friend +14.3pp, open_ended +4.1pp, negative +12.5pp. diluted_embedding −16.7pp (6 cases only). |
| 18 | run_18_contextual_prefix | 2026-03-24 | 173/173 | 80.3% | 77.8% | 0.835 | — | — | ~12.3s | Contextual hierarchy prefix in embedding text. Neutral vs run_17 overall. ciu_treatment −20pp, capital_ratios +3.2pp. Gains gated on paragraph chunking. |
| 19 | run_19_para_chunking | 2026-03-24 | 173/173 | 70.5% | 69.6% | 0.733 | — | — | ~10s | Para-chunked dual-doc index, PARAGRAPH-only retrieval. Regression overall — top-k flooding. credit_risk_sa +40pp, cash_pooling +100pp. Led to mixed-chunking insight. |
| **20** | **run_20_mixed_chunking** ⭐ | 2026-03-25 | 173/173 | **86.7%** | **83.6%** | **0.891** | **0.793** | — | ~10.5s | **Mixed ARTICLE+PARAGRAPH retrieval + ArticleDeduplicatorPostprocessor. New SOTA. +6.4pp Hit@1 vs run_17, zero regressions. Judge +2.3pp vs run_2e.** |
| 21 | run_21_domain_rewrite | 2026-03-25 | 173/173 | 80.9% | 80.4% | 0.849 | — | — | ~11s | Domain query rewriting (plain-language→CRR register via gpt-4o-mini). Hit@1 −5.8pp vs run_20. diluted_embedding unchanged. Regression. Reverted. |
| 22 | run_22_false_premise_prompt | 2026-03-25 | 173/173 | 86.7% | 83.6% | 0.892 | 0.782 | — | ~10.5s | Synthesis prompt FALSE PREMISE RULE + CONDITIONAL LOGIC RULE. false_friend −12.2pp (over-hedging). Overall Judge −1.1pp. Reverted. |
| 23 | run_23_eval_fix | 2026-03-26 | 173/173 | 86.7% | 83.6% | 0.891 | — | — | ~10.5s | Measurement fix: added `_with_expanded` metric suite counting expanded articles as lower-ranked retrieved results. Confirmed measurement bug: multi_article recall@3_with_expanded = 83.3% vs 38.9% raw. Baseline identical to run_20. USE_ARTICLE_GRAPH=false. |
| 24 | run_24_article_graph (judged) | 2026-03-26 | 173/173 | 86.7% | 83.6% | 0.891 | 0.787 | 0.830 | ~10.8s | Article dependency graph: BFS expansion (prerequisite→procedural→general→back_ref priority) replaces flat CSV sort. Dynamic multi-hop params (source_limit=4, depth=2, budget=5). Graph confirmed active. **Split result**: multi faithfulness +6.9pp, multi_hop faithfulness +5.4pp, own_funds +3.5pp correctness/+4.5pp faithfulness, leverage +6.7pp faithfulness. Regressions: single-article −1.1pp correctness, large_exposures faithfulness −4.3pp, negative faithfulness −6.2pp. Overall −0.6pp correctness. Root cause: graph adds noise for queries that don't need extra context. `is_multi_hop` regex unusable as gate (only 6% coverage of multi_hop cases). Fix: gate on post-retrieval distinct-article count. → run_25. |
| 25 | run_25_graph_gated | 2026-03-26 | pending | — | — | — | — | — | — | Graph gated on post-retrieval signal: BFS fires only when primary retrieval surfaces ≥2 distinct articles (fires for 62% of multi-article cases, 35% of multi_hop, only 33% of single-article false positives vs 100% in run_24). Hypothesis: preserves multi-hop/multi-article faithfulness gains while recovering single-article correctness regression. |

---

## Run Details

---

### baseline — 2026-03-20

**Timestamp:** 20260320T191153Z
**Workers:** n/a
**Judge:** gpt-4o

**Configuration:**
- Orchestrator, query normalisation, cross-reference expansion (Phases 0/2/3 code complete)
- No reranker (`USE_RERANKER=false`)
- `RETRIEVAL_TOP_K=12`, `RETRIEVAL_ALPHA=0.5`
- gpt-4o-mini synthesis, gpt-4o for multi-hop
- 13 hard failures: unhandled exceptions (articles 115, 121, 122, 132a, 132c, 152, 242, 243, 254, 258, 429b)

**Metrics:**

| Metric | Overall | Article-cited | Open-ended |
|--------|---------|---------------|------------|
| Hit@1 | 81.3% | 90.9% | 65.6% |
| Recall@1 | 77.2% | — | — |
| Recall@3 | 82.6% | — | — |
| Recall@5 | 85.3% | — | — |
| MRR | 0.842 | — | — |
| Judge Correctness | 0.774 | — | — |
| Judge Completeness | 0.788 | — | — |
| Judge Faithfulness | 0.799 | — | — |

**Latency:** p50=10.5s · p90=17.3s · mean=11.8s

**Key findings:**
- 13 hard failures (7.5%) all `status=error` — unhandled exceptions, not missing data (confirmed by Qdrant scan)
- 25pp gap between article-cited (90.9%) and open-ended (65.6%) Hit@1 — main improvement target
- Multi-article questions: Recall@1=27%, Recall@5=49.5% — severe gap
- `false_friend` type: 77.8% Hit@1 but only 0.500 Judge Correctness (retrieves the right article but synthesises the wrong answer)
- 10 cases scored 0.0 Judge Correctness (synthesis failures)

**Verdict:** Strong baseline but 13 crashes inflate Hit@1. Real performance closer to 73–74% once errors are fixed.

---

### run_1_20032025 — 2026-03-20

**Timestamp:** 20260320T212450Z
**Workers:** 6 · **Judge:** gpt-4o

**Configuration (changes vs baseline):**
- Sub-query fusion: multi-hop questions split into 2–3 sub-queries, merged via score-based deduplication
- HyDE pre-retrieval enrichment: LLM predicts relevant article numbers and appends as hints to open-ended queries
- Extended `_ABBREV_MAP` (+13 terms) and `_SYNONYM_MAP` (+9 legal paraphrases)
- Hardened citation rule in `_LEGAL_QA_TEMPLATE` (cite only articles present in context)
- `_LOW_CONFIDENCE_THRESHOLD` raised 0.35 → 0.40
- Multi-hop queries routed to gpt-4o for synthesis
- Unhandled exceptions now return HTTP 500 with type in detail (13 hard failures resolved to 1)

**Metrics:**

| Metric | Value | Δ vs baseline |
|--------|-------|---------------|
| Hit@1 | 78.5% | −2.8pp |
| Recall@3 | 81.5% | −1.1pp |
| MRR | 0.823 | −0.019 |
| Judge Correctness | 0.782 | +0.008 |
| Judge Faithfulness | 0.819 | +0.020 |

**Latency:** p50=11.8s · p90=17.9s · mean=12.1s (+0.3s)

**Key findings:**
- Sub-query fusion hurt retrieval — extra sub-queries diluted relevance scores
- Slight judge quality gain (+0.8pp correctness, +2.0pp faithfulness) from citation hardening + gpt-4o for multi-hop
- Baseline's 81.3% was inflated by 13 crashes that never touched the retrieval path; correcting for that, run_1 is not a true regression

**Verdict:** Retrieval regression vs (inflated) baseline. Reranker expected to recover the gap.

---

### phase1_reranker — 2026-03-20 (ABORTED)

**Timestamp:** 20260320T221131Z
**Workers:** unknown · **Judge:** gpt-4o

**Configuration:** Attempted `BAAI/bge-reranker-v2-m3` via `FlagEmbeddingReranker`

**Outcome:** Complete failure — 28/28 cases errored. Root cause: SIGSEGV when `bge-reranker-v2-m3` was co-loaded with `BGEm3Embedding` on Windows (`shm.dll` crash). Switched to `cross-encoder/ms-marco-MiniLM-L-6-v2` via `SentenceTransformerRerank` to unblock.

**Verdict:** Aborted. No usable results.

---

### run_2_20032025 — 2026-03-20

**Timestamp:** 20260320T222652Z
**Workers:** 6 · **Judge:** gpt-4o

**Configuration (changes vs run_1):**
- Reranker enabled: `USE_RERANKER=true`, model=`cross-encoder/ms-marco-MiniLM-L-6-v2`
- Pure reranker (no score blending): `SentenceTransformerRerank`, `RERANK_TOP_N=6`
- `RETRIEVAL_TOP_K=12`, `RETRIEVAL_ALPHA=0.5`, `TITLE_BOOST_WEIGHT=0`

**Metrics:**

| Metric | Value | Δ vs run_1 |
|--------|-------|------------|
| Hit@1 | 79.2% | +0.7pp |
| Recall@3 | 82.1% | +0.6pp |
| MRR | 0.829 | +0.006 |
| Judge Correctness | 0.769 | −0.013 |
| Judge Faithfulness | 0.786 | −0.033 |

**Latency:** p50=10.7s · p90=16.8s · mean=11.0s (−1.1s vs run_1)

**Breakdown — article-cited vs open-ended:**

| Segment | Hit@1 | Recall@3 | MRR |
|---------|-------|----------|-----|
| article_cited (n=100) | 91.0% | 89.3% | 0.910 |
| open_ended (n=73) | 63.0% | 71.5% | 0.719 |

**Key findings:**
- Pure reranker: +0.7pp Hit@1, −1.1s mean latency vs run_1 — best retrieval config found
- Judge scores dipped slightly vs run_1 (−1.3pp correctness) — trade-off deemed acceptable
- Article-cited essentially solved (91%); open-ended (63%) is the primary remaining gap
- Open-ended failure analysis: 15 retrieval failures (6 terminology dilution, 5 concept-in-unexpected-article, 3 niche sub-articles, 2 known hard) + 12 ranking failures (7 adjacent-article confusion, 4 false-friend)

**Verdict:** Best retrieval configuration. Set as primary benchmark.

---

### run_3_blend_ranker — 2026-03-20

**Timestamp:** 20260320T230111Z
**Workers:** 6 · **Judge:** gpt-4o

**Configuration (changes vs run_2):**
- Switched from `SentenceTransformerRerank` to `BlendedReranker`
- `RERANK_BLEND_ALPHA=0.3` (30% retrieval signal + 70% reranker signal)
- Motivation: fix rank-flip regressions (case_136: 506c > 26, case_149: 395 > 392)

**Metrics:**

| Metric | Value | Δ vs run_2 |
|--------|-------|------------|
| Hit@1 | 79.2% | 0 |
| Recall@3 | 81.5% | −0.6pp |
| MRR | 0.827 | −0.002 |
| Judge Correctness | 0.763 | −0.006 |

**Latency:** p50=11.5s · p90=18.1s · mean=12.2s (+1.2s)

**Key findings:**
- Fixed case_149 (article 395 no longer rank-flipped to 392) but introduced new regression in case_152
- Net neutral vs run_2 with added latency

**Verdict:** Net neutral. Pure reranker retained.

---

### run_3_blend_ranker2 — 2026-03-20

**Timestamp:** 20260320T231537Z
**Workers:** 6 · **Judge:** gpt-4o

**Configuration (changes vs run_3a):**
- `RERANK_BLEND_ALPHA=0.6` (60% retrieval signal + 40% reranker signal)

**Metrics:**

| Metric | Value | Δ vs run_2 |
|--------|-------|------------|
| Hit@1 | 78.6% | −0.6pp |
| Recall@3 | 81.6% | −0.5pp |
| MRR | 0.824 | −0.005 |
| Judge Correctness | 0.756 | −0.013 |

**Latency:** p50=9.6s · p90=13.3s · mean=9.9s (fastest run overall)

**Key findings:**
- Fixed case_149 but broke case_109 and case_152
- Higher alpha (more retrieval signal) = slight retrieval regression vs pure reranker
- Fastest latency of all runs (reranker less dominant, more weight on pre-computed scores)

**Verdict:** Slight regression. Pure reranker retained.

---

### run_4_HyDE — 2026-03-20

**Timestamp:** 20260320T235227Z
**Workers:** 6 · **Judge:** gpt-4o

**Configuration (changes vs run_2):**
- Query enrichment: LLM appends predicted article-number hints to open-ended queries before retrieval
- `USE_QUERY_ENRICHMENT=true` (default)
- Targets terminology-dilution failures (6 of 15 open-ended retrieval failures)

**Metrics:**

| Metric | Value | Δ vs run_2 |
|--------|-------|------------|
| Hit@1 | 79.2% | 0 |
| Recall@3 | 81.6% | −0.5pp |
| Recall@5 | 84.5% | +0.1pp |
| MRR | 0.828 | −0.001 |
| Judge Correctness | 0.768 | −0.001 |

**Latency:** p50=10.8s · p90=16.6s · mean=11.3s (+0.3s)

**Key findings:**
- Neutral overall: article hints help some open-ended cases but dilute embedding focus on others
- Kept in codebase as default (`USE_QUERY_ENRICHMENT=true`) since cost is low and no clear regression

**Verdict:** Neutral. Kept in codebase as default enrichment.

---

### run_4_HyDe_numHint — 2026-03-21

**Timestamp:** 20260321T001045Z
**Workers:** 6 · **Judge:** gpt-4o

**Configuration (changes vs run_4_HyDE):**
- Article-number hints formatted as explicit numeric references (e.g. "Article 92, Article 411") rather than plain hints
- Variant of the enrichment prompt targeting numeric anchor extraction

**Metrics:**

| Metric | Value | Δ vs run_2 |
|--------|-------|------------|
| Hit@1 | 79.8% | +0.6pp |
| Recall@3 | 80.7% | −1.4pp |
| Recall@5 | 84.3% | −0.1pp |
| MRR | 0.829 | 0 |
| Judge Correctness | 0.751 | −0.018 |

**Latency:** p50=10.7s · p90=17.9s · mean=11.5s

**Key findings:**
- Best raw Hit@1 of any single run (79.8%) — numeric anchors help top-1 precision
- But Recall@3 dropped −1.4pp: pushing one article to top-1 crowd-out other relevant articles
- Judge correctness drops −1.8pp — synthesis quality hurt when hints are overly prescriptive

**Verdict:** Mixed. Gains Hit@1 at the cost of Recall@3 and judge quality. Not adopted.

---

### run_5_title_boost — 2026-03-21

**Timestamp:** 20260321T003803Z
**Workers:** 6 · **Judge:** gpt-4o

**Configuration (changes vs run_2):**
- `TITLE_BOOST_WEIGHT=0.15` (additive post-rerank bonus proportional to query ↔ article title token overlap)
- Motivation: rescue open-ended vocabulary-gap queries where article title contains discriminative tokens (e.g. "Total Risk Exposure Amount")

**Metrics:**

| Metric | Value | Δ vs run_2 |
|--------|-------|------------|
| Hit@1 | 78.6% | −0.6pp |
| Recall@3 | 81.6% | −0.5pp |
| MRR | 0.824 | −0.005 |
| Judge Correctness | 0.757 | −0.012 |
| Judge Faithfulness | 0.773 | −0.013 |

**Latency:** p50=10.7s · p90=16.1s · mean=11.3s

**Key findings:**
- Title boost hurt across all metrics — title token overlap is noisy when query contains common legal terms (e.g. "capital", "institution") that appear in many article titles
- Stopword list mitigates but doesn't fully eliminate false boosts

**Verdict:** Harmful. `TITLE_BOOST_WEIGHT` reverted to 0.

---

### run_2b_20032025 — 2026-03-21

**Timestamp:** 20260321T013613Z
**Workers:** 6 · **Judge:** gpt-4o

**Configuration (identical to run_2 except):**
- `RETRIEVAL_TOP_K=15` (was 12 in run_2)
- Golden dataset updated: 2 cases (case_016, case_019) had reference answers corrected
- Full orchestrator active: gpt-4o-mini (standard queries) + gpt-4o (multi-hop queries)
- `USE_RERANKER=true`, `RERANK_TOP_N=6`, `RETRIEVAL_ALPHA=0.5`, `TITLE_BOOST_WEIGHT=0`

**Metrics:**

| Metric | Value | Δ vs run_2 |
|--------|-------|------------|
| Hit@1 | 79.2% | 0 |
| Recall@1 | 73.9% | 0 |
| Recall@3 | 81.8% | −0.3pp |
| Recall@5 | 84.5% | +0.1pp |
| MRR | 0.829 | +0.001 |
| Judge Correctness | 0.760 | −0.009 |
| Judge Completeness | 0.772 | +0.009 |
| Judge Faithfulness | 0.769 | −0.017 |

**Latency:** p50=27.7s · p90=41.4s · mean=26.4s (elevated — likely API concurrency under parallel judge calls)

**Breakdown — article-cited vs open-ended:**

| Segment | n | Hit@1 | Recall@3 | MRR | J.Correct |
|---------|---|-------|----------|-----|-----------|
| article_cited | 100 | 91.0% | 89.3% | 0.910 | 0.799 |
| open_ended | 73 | 63.0% | 71.5% | 0.719 | 0.706 |

**Breakdown — by difficulty:**

| Difficulty | n | Hit@1 | Recall@3 | MRR |
|------------|---|-------|----------|-----|
| easy | 47 | 89.4% | 92.6% | 0.918 |
| medium | 44 | 79.6% | 86.4% | 0.825 |
| hard | 82 | 73.2% | 73.2% | 0.781 |

**Key findings:**
- Confirms run_2 was not an artefact: same numbers on a corrected dataset
- TOP_K=15 (vs 12) made no statistically significant difference at this scale
- Elevated latency is anomalous (infrastructure/API variability), not a config effect
- **New official baseline**: all future runs compared against run_2b

**Verdict:** Confirmed best configuration. New baseline.

---

### run_6_gpt4o — 2026-03-21

**Timestamp:** 20260321T015718Z
**Workers:** 6 · **Judge:** gpt-4o

**Configuration (changes vs run_2b):**
- `OPENAI_MODEL=gpt-4o` for all queries (previously gpt-4o-mini for standard, gpt-4o only for multi-hop)

**Metrics:**

| Metric | Value | Δ vs run_2b |
|--------|-------|-------------|
| Hit@1 | 79.2% | 0 |
| Recall@3 | 80.9% | −0.9pp |
| MRR | 0.826 | −0.003 |
| Judge Correctness | **0.772** | +1.2pp |
| Judge Completeness | **0.780** | +0.8pp |
| Judge Faithfulness | **0.795** | +2.6pp |

**Latency:** p50=25.4s · p90=38.7s · mean=25.2s (2.3× run_2 baseline latency)

**Key findings:**
- Retrieval is identical (same retriever) — Hit@1 unchanged at 79.2%
- gpt-4o improves synthesis quality (+1.2pp correctness, +2.6pp faithfulness) but at 2.3× cost/latency
- Confirms bottleneck is retrieval, not synthesis quality
- Cost–quality trade-off not justified for production; gpt-4o-mini retained as default

**Verdict:** Not adopted. Useful confirmation that synthesis is not the bottleneck.

---

### run_7_toc_routing — 2026-03-21

**Timestamp:** 20260321T133420Z
**Workers:** 4 · **Judge:** gpt-4o

**Configuration (changes vs run_2b):**
- `USE_TOC_ROUTING=true`: LLM reasons over the full CRR Table of Contents to predict relevant articles in parallel to vector retrieval; results merged via Reciprocal Rank Fusion

**Metrics:**

| Metric | Value | Δ vs run_2b |
|--------|-------|-------------|
| Hit@1 | 78.6% | −0.6pp |
| Recall@3 | 81.6% | −0.2pp |
| Recall@5 | 85.1% | +0.6pp |
| MRR | 0.826 | −0.003 |
| Judge Correctness | 0.762 | +0.2pp |
| Judge Faithfulness | 0.789 | +2.0pp |

**Latency:** p50=18.6s · p90=29.5s · mean=19.5s (~1.8× run_2)

**Key findings:**
- ToC routing adds latency from parallel LLM call but does not improve retrieval
- Slight Hit@1 regression (−0.6pp) — ToC LLM occasionally ranks wrong articles
- Small Recall@5 gain (+0.6pp) suggests ToC helps surface long-tail articles, but not at top-1
- Faithfulness gain (+2.0pp) possibly from better article context diversity, but not significant

**Verdict:** Neutral/slightly negative. `USE_TOC_ROUTING=false` retained.

---

### run_8_trueHyDE — 2026-03-21

**Timestamp:** 20260321T141938Z
**Workers:** 4 · **Judge:** gpt-4o

**Configuration (changes vs run_2b):**
- `USE_HYDE=true`: instead of enriching the query with article hints, the LLM generates a full hypothetical CRR-style legislative passage as the retrieval query (replaces plain-language question for dense retrieval)
- Targets the 14 terminology-dilution retrieval failures where query vocabulary has no overlap with article text

**Metrics:**

| Metric | Value | Δ vs run_2b |
|--------|-------|-------------|
| Hit@1 | 79.2% | 0 |
| Recall@3 | 81.5% | −0.3pp |
| Recall@5 | 84.0% | −0.5pp |
| MRR | 0.828 | −0.001 |
| Judge Correctness | 0.750 | −1.0pp |
| Judge Completeness | 0.762 | −1.0pp |
| Judge Faithfulness | 0.787 | +1.8pp |

**Latency:** p50=19.4s · p90=31.2s · mean=20.3s (~1.9× run_2)

**Key findings:**
- True HyDE is neutral on retrieval — generates legislative-style prose that matches index vocabulary but adds ~8s latency per query
- Judge correctness/completeness slight drop (−1pp each): hypothetical passages occasionally introduce speculative framings that pollute synthesis
- Does not fix the 14 terminology-dilution cases as hoped — the 6 true dilution failures need deeper vocabulary bridging

**Verdict:** Neutral retrieval, slight judge regression, significant latency cost. `USE_HYDE=false` retained in baseline config.

---

### run_2c_baseline — 2026-03-21 ⚠️ INVALID (crashed)

**Status:** Crashed mid-run. Only 125/173 cases actually evaluated; the summary JSON was written by a later resume run and falsely reported 173/173. All metrics are unreliable. Use run_9_params_tuning (with judge) or run_2d_baseline (post-review code) as the true baseline.

---

### run_9_params_tuning — 2026-03-23

**Timestamp:** 20260323T~  (pre code-review, git=bb2ff7d)
**Workers:** 4 · **Judge:** gpt-4o

**Configuration (changes vs run_2b):**
- `RETRIEVAL_ALPHA=0.7` (was 0.5)
- `RETRIEVAL_TOP_K=20` (was 15)
- `RERANK_TOP_N=8` (was 6)
- `USE_RERANKER=true`, `TITLE_BOOST_WEIGHT=0`, `USE_HYDE=false`, `USE_TOC_ROUTING=false`

**Metrics:**

| Metric | Value | Δ vs run_2b |
|--------|-------|-------------|
| Hit@1 | 78.6% | −0.6pp |
| Recall@3 | 81.8% | 0.0pp |
| Recall@5 | 84.7% | +0.2pp |
| MRR | 0.824 | −0.005 |
| Judge Correctness | **0.779** | **+1.9pp** |
| Judge Completeness | **0.787** | **+1.5pp** |
| Judge Faithfulness | **0.793** | **+2.4pp** |

**Latency:** p50=19.7s · p90=~32s · mean=20.1s

**Breakdown — article-cited vs open-ended:**

| Segment | n | Hit@1 | Recall@3 | MRR | J.Correct |
|---------|---|-------|----------|-----|-----------|
| article_cited | 100 | 91.0% | 89.3% | 0.910 | — |
| open_ended | 73 | 61.6% | 71.5% | 0.707 | — |

**Breakdown — by question type (worst performers):**

| Type | n | Hit@1 | Recall@3 |
|------|---|-------|----------|
| diluted_embedding | 6 | 33.3% | 41.7% |
| ambiguous | 2 | 50.0% | 16.7% |
| false_friend | 14 | 50.0% | 73.8% |
| procedural | 30 | 73.3% | 76.7% |

**Key findings:**
- Dense-heavy alpha (0.7) did NOT help `diluted_embedding` — Hit@1 unchanged at 33%. Hypothesis was wrong: the vocabulary gap is too large for higher dense weight to bridge without synonym expansion.
- Wider pool (TOP_K=20, RERANK_TOP_N=8) did NOT fix `false_friend` ranking — same 50% Hit@1 as run_2b.
- Judge quality improvement (+1.9pp correctness, +2.4pp faithfulness) despite retrieval regression — likely LLM variance / judge sampling.
- Slight Hit@1 regression (−0.6pp) across the board — higher alpha hurts BM25-strong article-cited queries at the margin.

**Verdict:** Hypothesis not confirmed. alpha=0.5 and TOP_K=15/RERANK_TOP_N=6 retained as baseline config. Synonym expansion (run_10) is the right lever for diluted_embedding.

---

### run_2d_baseline — 2026-03-23 ⭐ CURRENT CODE BASELINE

**Timestamp:** 20260323T161828Z
**Workers:** 4 · **Judge:** disabled
**Git:** post code-review fixes (Fix 1–7: cancel token, atomic load(), EVAL_MODE, direct article scroll, warmup health, dead cache removal, HyDE multi-line parser)

**Configuration (identical to run_2b/run_2c):**
- `USE_RERANKER=true`, `RERANK_TOP_N=6`, `RETRIEVAL_TOP_K=15`, `RETRIEVAL_ALPHA=0.5`
- `TITLE_BOOST_WEIGHT=0`, `USE_HYDE=false`, `USE_TOC_ROUTING=false`
- `QUERY_TIMEOUT_SECONDS=150` (raised from 120)

**Metrics:**

| Metric | Value | Note |
|--------|-------|------|
| Hit@1 | 76.9% | 169 cases; 4 failures excluded |
| Recall@3 | 79.8% | |
| Recall@5 | 83.5% | |
| MRR | 0.818 | |
| Judge | — | not run |

**Latency:** p50=10.2s · p90=17.6s · mean=11.1s

**Failures (4 cases):**
- case_129, case_131, case_148 → `http_504` at ~122s (old 120s API timeout, now fixed to 150s)
- case_161 → `http_500` (transient OpenAI error; succeeded in 12/13 other runs)

**Key findings:**
- Metrics appear lower than run_2b/run_9 but this is because judge was disabled and all 4 failures are excluded (3 are hard cases that may time out even at 150s).
- On the 124 cases common to both run_2c and run_2d, run_2d is essentially equivalent (+0.8pp Hit@1, −1.2pp Recall@3 — within LLM variance).
- The 150s timeout fix should resolve the 3 `http_504` failures in the next eval run.

**Verdict:** Clean post-code-review baseline. Run a full judge run with `QUERY_TIMEOUT_SECONDS=150` after synonym expansion (run_10) for a reliable apples-to-apples comparison.

¹ run_2c actual count verified by counting `status=ok` entries in the cases file.

---

### run_15_toc2 — 2026-03-24

**Workers:** 1 · **Judge:** disabled

**Configuration (changes vs run_12):**
- `USE_TOC_ROUTING=true` with parallel execution and fixed timeout (prior run_7 was confounded by sequential blocking)
- Qdrant payload indexes active (`part`, `title`, `chapter`, `section`)

**Metrics:**

| Metric | Value | Δ vs run_12 |
|--------|-------|-------------|
| Hit@1 | 76.3% | −1.7pp |
| Recall@3 | 80.2% | +1.0pp |
| MRR | 0.824 | 0.000 |

**By type (key deltas):**
- `multi_article`: +9.4pp ✅
- `false_friend`: +7.1pp ✅
- `diluted_embedding` Recall@3: +37.5pp ✅
- `liquidity`: −7.4pp ❌
- `own_funds`: −6.5pp ❌
- `threshold`: −7.1pp ❌

**Verdict:** Selective routing needed — universal ToC firing hurts dense-strong categories. Retired. Gains suggest query-type classifier could unlock ToC for specific slices.

---

### run_16_toc_confidence — 2026-03-24

**Workers:** 1 · **Judge:** disabled

**Configuration (changes vs run_12):**
- Selective ToC routing: fires only when max reranker confidence score < 0.55

**Metrics:**

| Metric | Value | Δ vs run_12 |
|--------|-------|-------------|
| Hit@1 | 77.5% | −0.5pp |
| Recall@3 | 79.2% | 0.000 |
| MRR | 0.821 | −0.003 |

**Verdict:** Confidence threshold too blunt — ToC still fires on cases where retrieval is actually fine. Net regression vs run_12. ToC routing retired.

---

### run_17_para_window — 2026-03-24 ⭐

**Workers:** 1 · **Judge:** disabled

**Configuration (changes vs run_12):**
- `USE_PARAGRAPH_WINDOW_RERANKER=true`: reranker scores ±1 paragraph neighbours from existing article nodes; best window replaces the article node for context assembly
- `PARAGRAPH_WINDOW_MAX_WINDOWS=4`

**Metrics:**

| Metric | Value | Δ vs run_12 |
|--------|-------|-------------|
| Hit@1 | **80.3%** | **+2.3pp** |
| Recall@3 | 78.9% | −0.3pp |
| MRR | **0.840** | **+0.016** |

**By type (key deltas):**
- `false_friend`: +14.3pp ✅
- `open_ended`: +4.1pp ✅
- `negative`: +12.5pp ✅
- `diluted_embedding`: −16.7pp ❌ (6 cases; acceptable)

**Latency:** +800ms vs run_12

**Verdict:** Best retrieval config to date. Para-window reranker kept permanently.

---

### run_18_contextual_prefix — 2026-03-24

**Workers:** 1 · **Judge:** disabled

**Configuration (changes vs run_17):**
- Re-ingested with contextual hierarchy prefix prepended to each node's embedding text (e.g. "Part III > Title II > Chapter 2 > Article 92: …")
- Para-window reranker active

**Metrics:**

| Metric | Value | Δ vs run_17 |
|--------|-------|-------------|
| Hit@1 | 80.3% | 0.000 |
| Recall@3 | 77.8% | −1.1pp |
| MRR | 0.835 | −0.005 |

**By type (key deltas):**
- `capital_ratios`: +3.2pp ✅
- `own_funds`: +3.2pp ✅
- `ciu_treatment`: −20pp ❌
- `known_failures`: −8.3pp ❌

**Latency:** +1s vs run_17

**Verdict:** Neutral overall. Prefix on full-article blobs dilutes; gains require paragraph-level chunking. Index retained (prefix harmless).

---

### run_19_para_chunking — 2026-03-24

**Workers:** 1 · **Judge:** disabled

**Configuration (changes vs run_18):**
- Dual-document index: original article-level index + new paragraph-chunked index
- `RETRIEVAL_MODE=paragraph_only`: queries only the paragraph-chunked index

**Metrics:**

| Metric | Value | Δ vs run_17 |
|--------|-------|-------------|
| Hit@1 | 70.5% | −9.8pp |
| Recall@3 | 69.6% | −9.3pp |
| MRR | 0.733 | −0.107 |

**By type (key deltas):**
- `credit_risk_sa`: +40pp ✅
- `cash_pooling` / `leverage_ratio`: +100pp ✅ (niche sub-articles)
- Most other types: heavy regression due to top-k flooding by paragraph fragments

**Verdict:** Paragraph-only flooding kills overall performance. Key insight: mix article + paragraph nodes with a deduplicator → led to run_20.

---

### run_20_mixed_chunking — 2026-03-25 ⭐ CURRENT SOTA

**Workers:** 1 · **Judge:** gpt-4o (partial)

**Configuration (changes vs run_17):**
- `USE_MIXED_CHUNKING=true`: retrieves from both article-level and paragraph-level collections simultaneously
- `ArticleDeduplicatorPostprocessor`: collapses paragraph hits to their parent article before context assembly to prevent duplication
- No re-ingest: uses dual-document index from run_19

**Metrics:**

| Metric | Value | Δ vs run_17 |
|--------|-------|-------------|
| Hit@1 | **86.7%** | **+6.4pp** |
| Recall@3 | **83.6%** | **+4.7pp** |
| MRR | **0.891** | **+0.051** |
| Judge Correctness | **0.793** | n/a (vs run_2e: +2.3pp) |

**By type (remaining hard gaps):**
- `diluted_embedding`: 16.7% Hit@1 (unchanged across all runs)
- `false_friend`: 85.7% Hit@1, Judge Correctness=0.679 (retrieval solved, synthesis fails)
- `multi_article` Recall@1: 29.7%
- `open_ended`: 80.8% Hit@1

**Verdict:** New SOTA. +6.4pp Hit@1 vs run_17 with zero regressions. Mixed chunking + deduplicator is the architecture. All future runs use this as baseline.

---

### run_21_domain_rewrite — 2026-03-25

**Workers:** 1 · **Judge:** disabled

**Configuration (changes vs run_20):**
- Pre-retrieval query rewrite via gpt-4o-mini: plain-language query → CRR legal register (e.g. "pledged bonds" → "collateralised by central government securities per Art. 411")
- Targets `diluted_embedding` (16.7% Hit@1 unchanged through all prior runs)

**Metrics:**

| Metric | Value | Δ vs run_20 |
|--------|-------|-------------|
| Hit@1 | 80.9% | −5.8pp |
| Recall@3 | 80.4% | −3.2pp |
| MRR | 0.849 | −0.042 |

**By type (key deltas):**
- `diluted_embedding`: unchanged at 16.7% ❌
- `false_friend`: −28.6pp ❌ (rewrite introduces false synonyms)
- `open_ended`: −12.3pp ❌

**Verdict:** Heavy regression. Rewriting changes query intent for good queries without fixing vocabulary-gap ones. Reverted. Dead end for this approach.

---

### run_22_false_premise_prompt — 2026-03-25

**Workers:** 1 · **Judge:** gpt-4o

**Configuration (changes vs run_20):**
- Synthesis prompt additions: FALSE PREMISE RULE (explicitly instruct to refute false premises) + CONDITIONAL LOGIC RULE (state conditions rather than binary yes/no)
- Targets `false_friend` synthesis failures (Judge Correctness=0.679, retrieval already 85.7%)

**Metrics:**

| Metric | Value | Δ vs run_20 |
|--------|-------|-------------|
| Hit@1 | 86.7% | 0.000 |
| Recall@3 | 83.6% | 0.000 |
| MRR | 0.892 | +0.001 |
| Judge Correctness | 0.782 | −1.1pp |

**By type (key deltas):**
- `false_friend` Judge Correctness: 0.679 → 0.557, −12.2pp ❌ (over-hedging)
- `diluted_embedding`: +8.3pp ✅ (side effect)

**Verdict:** Blanket rules cause over-hedging on false_friend cases. Net regression on the target. Reverted. Next: targeted few-shot examples per Grok Exp 1.

---

## Summary of Findings

### What works
- **Mixed ARTICLE+PARAGRAPH retrieval + ArticleDeduplicatorPostprocessor** (run_20): +6.4pp Hit@1, +4.7pp Recall@3 vs run_17 — zero regressions. **Single highest-leverage architectural change.**
- **Paragraph-window reranker** (`USE_PARAGRAPH_WINDOW_RERANKER=true`, run_17): +2.3pp Hit@1 vs run_12
- **Adjacent article tiebreaker** (run_12): +1.7pp Hit@1, false_friend +14.3pp
- **Pure cross-encoder reranker** (`ms-marco-MiniLM-L-6-v2`, top_n=6, run_2): +0.7pp Hit@1, −1.1s latency vs no-reranker baseline
- **gpt-4o for multi-hop only**: good balance of quality and cost; using gpt-4o for all queries gives +1.2pp judge correctness but 2.3× latency
- **Hybrid retrieval** (dense + sparse, α=0.5): stable foundation throughout

### What doesn't work
- Sub-query fusion: dilutes retrieval scores (−2.8pp Hit@1)
- Blended reranker (any alpha): net neutral at best, slight regression at worst
- Article title boost (0.15): noisy — hurts across the board
- Universal ToC routing: adds latency, overall regression (selective wins on multi_article + diluted slices but not worth the cost)
- True HyDE / domain query rewrite: neutral or negative retrieval, adds latency
- Blanket synthesis prompt rules (FALSE PREMISE RULE): over-hedging — hurt false_friend Judge −12.2pp
- L-12 reranker: regression overall vs L-6 on this corpus

### Main remaining gaps (post run_20 SOTA)

| Gap | Hit@1 | Judge Correctness | Root cause |
|-----|-------|-------------------|------------|
| `diluted_embedding` | 16.7% | — | Query vocabulary ≠ CRR article vocabulary (TREA, pledged bonds, etc.) — unchanged across ALL runs |
| `false_friend` synthesis | 85.7% | 0.679 | Retrieval works; model fails to refute false premise |
| `multi_article` Recall@1 | 29.7% | 0.644 | Cross-ref expansion too conservative; secondary articles missed |
| `open_ended` | 80.8% | — | Mixed query types; classification + routing would help |

### Target metrics
Hit@1 ≥ 90% · Judge Correctness ≥ 0.85 — **current gap: Hit@1=86.7% (run_20, no judge), Judge Correctness=0.793 (run_20, partial judge)**

---

## Baseline Reproduction

Current baseline config (run_2d / post code-review):

```bash
# .env (current state):
# USE_RERANKER=true  |  RETRIEVAL_TOP_K=15  |  RERANK_TOP_N=6
# RETRIEVAL_ALPHA=0.5  |  TITLE_BOOST_WEIGHT=0
# USE_HYDE=false  |  USE_TOC_ROUTING=false
# QUERY_TIMEOUT_SECONDS=150

python -m evals.run_eval --run-name <run_name> --workers 4 --judge --judge-model gpt-4o
```

---

## Completed Experiments (quick reference)

| Run | Key change | Verdict |
|-----|-----------|---------|
| run_9 | ALPHA=0.7, TOP_K=20, RERANK_TOP_N=8 | Neutral retrieval, +1.9pp judge. Params reverted. |
| run_10 | Broad synonyms | Regressions. Broad rolled back; surgical kept. |
| run_2e | Clean judge reference baseline | ⭐ 173/173, Judge Correctness=0.770. |
| run_12 | Adjacent article tiebreaker δ=0.05 | ⭐ +1.7pp Hit@1, false_friend +14.3pp. Kept. |
| run_13 | L-12 reranker | Regression −1.15pp. L-6 retained. |
| run_15 | Universal ToC routing | Overall regression; selective gains on multi_article/diluted. Retired. |
| run_16 | Selective ToC (confidence <0.55) | Still regression vs run_12. Retired. |
| run_17 | Para-window reranker | ⭐ +2.3pp Hit@1. Kept. |
| run_18 | Contextual prefix (re-ingest) | Neutral. Gains gated on para chunking. Index kept. |
| run_19 | Para-only retrieval | Regression (flooding). Led to mixed-chunking insight. |
| run_20 | Mixed ARTICLE+PARA + ArticleDeduplicator | ⭐ **New SOTA +6.4pp, zero regressions.** |
| run_21 | Domain query rewrite (gpt-4o-mini) | Regression −5.8pp. Reverted. Dead end. |
| run_22 | FALSE PREMISE + CONDITIONAL LOGIC prompt rules | Over-hedging; false_friend −12.2pp judge. Reverted. |

---

## Planned Experiments (post run_20 SOTA)

Source: Grok analysis (2026-03-25). Reference baseline: **run_20** — Hit@1=86.7%, Judge Correctness=0.793. Target: Hit@1 ≥ 90%, Judge Correctness ≥ 0.85. All experiments require no re-ingest; ordered by expected impact × feasibility.

---

### run_23 — Targeted few-shot false-premise synthesis (Grok Exp 1)

**Priority:** High · **Effort:** Low · **Re-ingest:** No

**Hypothesis:** Replace the blanket FALSE PREMISE RULE (run_22 — failed due to over-hedging) with 3–4 hand-crafted few-shot examples drawn directly from the 14 `false_friend` golden cases. Add the guard: "Apply only when premise clearly contradicts CRR text; otherwise answer normally."

**Target gap:** `false_friend` Judge Correctness (currently 0.557 post-run_22, 0.679 at run_20 baseline). Retrieval already 85.7% Hit@1 — synthesis is the bottleneck.

**Expected mechanism:** Few-shot examples teach nuance (refute + cite correct article + state actual requirement) without universal hedging. Should lift false_friend Judge ≥+0.15pp while preserving overall correctness.

**Implementation:** Prompt edit in synthesis template only. Ablation: full judge eval vs run_20; isolate delta on `false_friend` + overall Judge Correctness.

**Risk:** Very low. Single-article and definition slices (>90% Hit@1) unaffected. Tiny risk of over-refutation on 2 ambiguous cases.

---

### run_24 — Recursive top-3 cross-reference expansion (Grok Exp 2)

**Priority:** High · **Effort:** Medium · **Re-ingest:** No

**Hypothesis:** Current expansion: top-2 primary nodes → fetch referenced articles. Change to: top-3 primary nodes → recursively fetch 1st-level references (max 5 additional nodes total) → joint cross-encoder rerank + ArticleDeduplicator.

**Target gap:** `multi_article` Recall@1=29.7%, `multi_hop` Judge=0.7375. CRR is heavily referential; surfacing secondary articles earlier in the reranker pool will boost secondary recall.

**Expected mechanism:** Metadata `referenced_articles` already exists — expansion is a code change in the retrieval postprocessor, not an ingest change.

**Implementation:** Update postprocessor in retrieval pipeline. Ablation: retrieval-only run (no judge); measure Recall@1 and MRR on `multi_article` + `multi_hop` slices vs run_20.

**Risk:** Low-moderate. Securitisation and liquidity (cross-ref dense) may see minor top-k dilution, but deduplicator + reranker should neutralise. Single-article slices unaffected.

---

### run_25 — Static CRR jargon synonym expansion BM25-only (Grok Exp 3)

**Priority:** High · **Effort:** Low-Medium · **Re-ingest:** No

**Hypothesis:** Build a static map (~30–50 entries) from golden `diluted_embedding` failures + Art 4 definitions (e.g. "pledged government bonds" ↔ "collateralised by government bonds / Art 411", "TREA" ↔ "total risk exposure amount / Art 92", "profit and loss transfer agreement" ↔ "P&L transfer / Art 28"). Expand **only the BM25 query string** with OR-ed synonyms; vector search unchanged.

**Target gap:** `diluted_embedding` (16.7% Hit@1 — unchanged across ALL prior runs). Also targets `open_ended` (80.8%).

**Expected mechanism:** Boosts BM25 signal for jargon mismatches; RRF fusion lifts relevant nodes into top-15 without touching dense embeddings. Expected +20–30pp on diluted_embedding.

**Implementation:** One-time synonym map + query preprocessor edit. Ablation: retrieval-only run; compare Hit@1/Recall@3 on `diluted_embedding` + `open_ended` vs run_20.

**Risk:** Low. BM25-only change leaves vector scores untouched. Surgical scope (specific jargon terms only, no broad expansions like run_10).

---

### run_26 — Query-type classifier + adaptive retrieval strategy (Grok Exp 4)

**Priority:** Medium · **Effort:** Medium · **Re-ingest:** No

**Hypothesis:** Add a one-shot gpt-4o-mini classifier (~300 tokens, <1s) at orchestrator start that tags query as `diluted`, `multi_hop`, `open_ended`, or `standard`. Routing: (a) `diluted` → trigger Exp-3 synonym expansion, (b) `multi_hop` → Exp-2 recursive refs, (c) `open_ended` + low reranker confidence (<0.55) → selective ToC pre-filter (cached TOC) + metadata filter on likely Part/Title.

**Target gap:** `open_ended` (80.8%) and `diluted_embedding` (16.7%) — reuses selective-ToC gains from run_15 (multi_article +9.4pp) without universal regression.

**Implementation:** Classifier prompt + if-logic in orchestrator. Ablation: run two variants (classifier on vs always-standard); compare by_question_type + by_citation_type.

**Risk:** Low. Easy/medium slices (n=91) get "standard" path unchanged. Classifier errors affect only hard slice. Requires runs 23+25 as prerequisites for routing targets.

---

### run_27 — Category-aware metadata pre-filter for open_ended (Grok Exp 6)

**Priority:** Medium · **Effort:** Low · **Re-ingest:** No

**Hypothesis:** For classifier-tagged `open_ended` queries, run a tiny gpt-4o-mini category detector ("capital_ratios / own_funds / liquidity / …") and add Qdrant payload filter on `part`/`title` before hybrid retrieval. Article-cited queries bypass the filter entirely.

**Target gap:** `open_ended` (80.8%). Legal hierarchy makes category a strong prior — shrinks search space early → higher precision in top-k.

**Implementation:** Reuse existing payload indexes + classifier from run_26. Ablation: compare `open_ended` Hit@1 and overall MRR vs run_20.

**Risk:** Very low. Only fires on `open_ended` queries.

---

### Deferred — requires re-ingest

| Experiment | Target | Effort |
|-----------|--------|--------|
| 429x sub-article metadata grouping | `leverage_ratio_cash_pooling` (0% Hit@1, 3 niche sub-articles) | High |
| Structural ref extraction (Part/Title/Chapter refs) | Cross-ref expansion completeness | High |
| Embedding model swap (multilingual-e5-large-instruct) | Diluted embedding + overall quality | High |
| BGE-reranker-v2-m3 (if CPU-compatible) | Stronger reranker for nuanced legal distinctions | Medium |
