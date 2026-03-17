# WORKLOG — Open Tasks

For completed work history, see `COMPLETED.md`.

---

## Current State (as of 2026-03-18) ✅ CLEAN INDEX — 1490 items (745 EN + 745 IT) | 255 unit tests green

Qdrant collection rebuilt clean on 2026-03-18 via Colab T4 (re-ingest with `--reset` after Codex V2 fixes).
Smoke test passed — Article 92 query returns correct answer with proper citations.
Git consolidated to single `main` branch (deleted local + remote `master`).

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
| 7 | Retrieval + answer quality measured separately | ❌ No eval pipeline |
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
| 1 | **Cross-ref expansion prompt size** | Expanded nodes appended after source nodes without re-ranking. For cross-reference-heavy articles, the prompt grows to `top_k (6) + max_expansions (3)` articles with no quality ordering among expanded nodes. | Acceptable for MVP. Revisit if synthesis quality degrades on heavy-reference articles. |
| 2 | **Language detection heuristic** | `_detect_language()` uses character-set diacritics only. Italian diacritics overlap with French, Romanian, Portuguese — English queries always return `None`. | Replace with `langdetect`/`lingua-py` if additional Romance languages are added. |
| 3 | **BGEm3Embedding on Windows (indexer)** | Prior SIGSEGV traced to simultaneous reranker + BGE-M3 loading; reranker not loaded during ingestion so risk is low. Ingestion is currently Colab-only. | Test on Windows before enabling local ingestion. |
| 4 | **`--reset` drops all languages** | `ingest_pipeline --reset` wipes the entire Qdrant collection including all languages. No per-language reset. | Tracked in backlog as `--language-only-reset`. |
| 5 | **Single-article direct lookup only** | `_detect_direct_article_lookup()` routes to metadata-filtered retrieval only when exactly one article is mentioned. Multi-article queries may miss one article if cosine scores are unbalanced. | Partially addressed by Codex V2 finding #5. |
| 6 | **No token budget management** | Total prompt tokens for synthesis are unbounded. No truncation before the OpenAI call. | Low risk at current corpus size. Add token budget cap if context window errors appear. |

---

## High Priority

### Golden dataset [High effort]
Curate 50–100 question/answer pairs for CRR covering: own funds, risk weights, leverage ratio, liquidity, large exposures, reporting. Each entry: `question`, `expected_articles`, `reference_answer`. Store as `evals/golden_dataset.jsonl`. Start with 20 high-value questions.

### Retrieval eval pipeline [High effort]
Implement `evals/run.py` — Recall@k, MRR, hit rate against golden dataset. Compare retrieved articles against `expected_articles`. Separate retrieval evaluation from answer quality evaluation. Add `evals/compare.py` for regression detection between runs.

### Cross-reference expansion improvements [High effort] — CRUCIAL

The current cross-reference system only handles **article-to-article** references via the `referenced_articles` metadata field. Two open gaps (both require re-ingest):

| # | Gap | What's missing | Requires re-ingest? | Priority |
|---|-----|---------------|---------------------|----------|
| 1 | **Structural ref extraction** | Part/Title/Chapter/Section refs not parsed at ingest time. Refs like "Part Six", "Title VII", "Chapter 1" are ignored. | **Yes** — ingest must extract and store structural refs in metadata (e.g. `referenced_parts`, `referenced_titles`, `referenced_chapters`) | High |
| 2 | **Range ref parsing for structural refs** | `Parts Two to Five` / `Titles I to III` not handled by regex. Article ranges are covered; structural ranges are not. | **Yes** (ingest regex) | High |

**Operational note**: Both gaps require re-ingestion (metadata changes at parse time, no re-embedding needed — Qdrant supports payload updates).

---

## UI/UX — Phase 2

### Italian language parity audit [Medium effort]
Verify all English-side fixes (external ref exclusion, inline list splitting, cross-ref rendering, article viewer formatting) work correctly for Italian text. Italian uses different prepositions (`del/della/dello`), article keywords (`Articolo`), and legislative terminology (`Regolamento/Direttiva`).

### Clickable hierarchy breadcrumbs [Medium effort, blocked by tree navigator API]
`DocumentBreadcrumb.tsx` renders breadcrumbs as plain text. Clicking "Chapter 6" should show all articles in that chapter. Requires `GET /api/navigator/{level}/{id}` endpoint.

### Relevant paragraph highlighting + auto-scroll [High effort]
When opening a cited article, the viewer shows the full article with no indication of which paragraph is cited. Backend would need to return the specific node/paragraph ID used in retrieval; frontend would highlight and scroll to it.

### Hover previews for article references [High effort]
Hovering over an inline "Article N" reference should show a tooltip with the article title, first paragraph, and an "Open article" button. Requires a lightweight article-preview API call on hover.

---

## Research / Architecture Enhancements

### ToC-based query routing [Medium effort]
Build a secondary Qdrant index (or in-memory map) of CRR Table of Contents entries — one document per Part/Title/Chapter with heading and scope description. On each query:
1. Retrieve the most relevant ToC entries (e.g. "CET1 requirements" → Part Two, Title I, Chapter 1)
2. Inject structural metadata filters (`part`, `title`, `chapter`) into the main Qdrant retrieval

Pairs well with the extended thinking enhancement below.

### Extended thinking / model reasoning [Medium effort]
Leverage Claude's extended thinking API feature to improve synthesis quality for complex queries:
1. **Synthesis-time reasoning** — model reasons over retrieved articles before producing the final answer; most useful for multi-hop questions where articles cross-reference each other.
2. **Pre-retrieval query planning** — model reasons about which structural areas of the CRR to search before hitting Qdrant. Pairs naturally with ToC routing above.

Detect query complexity and conditionally invoke extended thinking only for complex cases to manage latency and cost.

---

## Medium Priority

### Hybrid alpha tuning [Low effort, blocked by golden dataset]
`RETRIEVAL_ALPHA` env var is now wired in (default `0.5`). Tune by sweeping values (0.3–0.7) and measuring Recall@6 on golden dataset once it exists.

### Experiment tracking [Medium effort]
Add `evals/config.json` per run capturing: embed model, top_k, cutoff, alpha, prompt hash, corpus version (CELEX + date). Store alongside results for regression detection.

### Index upsert mode [Medium effort]
Add `--upsert` mode to ingest pipeline that checks existing node_ids and only re-embeds changed articles. Add `--language-only-reset` to wipe one language without affecting others.

---

## Low Priority

### Embedding model benchmark [Medium effort, blocked by golden dataset]
Benchmark BGE-M3 vs. alternatives (e.g. `multilingual-e5-large-instruct`) on Recall@6 / MRR using golden dataset.

### Dynamic top_k [Low effort, blocked by golden dataset]
Tune top_k based on query complexity after golden dataset exists.

### Continuous eval [Medium effort, blocked by eval pipeline]
Nightly eval run against golden dataset, alert on regression > 5%. Requires eval pipeline to exist first.

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

## Evaluation Pipeline Design (Phase 3)

### Components
1. **Golden dataset** — `evals/golden_dataset.jsonl` (50–100 Q&A pairs)
2. **Retrieval metrics** (no LLM needed) — Recall@k, MRR, hit rate
3. **Answer quality metrics** (LLM-as-judge) — faithfulness, relevance, correctness, citation accuracy
4. **Eval runner** — `python -m evals.run --dataset evals/golden_dataset.jsonl --output evals/results/`
5. **Regression comparison** — `evals/compare.py baseline.json candidate.json`

### File layout
```
evals/
  golden_dataset.jsonl       # curated Q&A pairs
  run.py                     # eval runner CLI
  compare.py                 # regression diff tool
  results/                   # gitignored run outputs
```

### Open questions
- Who creates the golden dataset? (domain expert vs. LLM-generated + human review)
- Which languages to cover? (EN only for MVP; EN+IT for full coverage)
- How to handle CRR article updates? (golden dataset must track CELEX version)
