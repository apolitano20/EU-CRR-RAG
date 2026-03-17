# WORKLOG — Open Tasks

For completed work history, see `COMPLETED.md`.

---

## Current State (as of 2026-03-17) ✅ CLEAN INDEX — 1490 items (745 EN + 745 IT) | 245 unit tests green

Qdrant collection rebuilt clean on 2026-03-17 via Colab T4.
Smoke test passed — Article 92 query returns correct answer with proper citations.
Git consolidated to single `main` branch (deleted local + remote `master`).

### Codex V2 fixes applied (2026-03-17)
Four ingestion + query bugs fixed (all unit-tested, 234 tests pass):

1. **Fix 1 — Annex overmatch** (`eurlex_ingest.py:176`): regex `^anx_` → `^anx_[^.]+$` stops sub-annex IDs (e.g. `anx_IV.1`) from creating duplicate documents. Requires `--reset` + re-ingest to purge stale Qdrant points.
2. **Fix 2 — Formula paragraph child-walk** (`eurlex_ingest.py:468`): `<p>` handler now walks children in document order, preserving prefix/suffix text around inline `<img>` formula elements instead of early-returning after the first formula.
3. **Fix 3 — Layout-A nested grid flattening** (`eurlex_ingest.py:411`): replaced `cols[1].get_text(...)` with a child-walk that calls `walk()` for nested `<div>` sub-points, so sub-point labels and formula placeholders are emitted separately rather than concatenated.
4. **Fix 4a — Cross-reference range parsing** (`eurlex_ingest.py:611`): replaced single-number regex with a full-run pattern that expands `Articles N to M` into individual article numbers and handles comma/and lists; also fixes multi-letter suffixes (`92aa`).
5. **Fix 4b — Query-time range expansion** (`query_engine.py`): added `_expand_article_ranges()` wired into `_normalise_query()` so queries like "Articles 89 to 91" are expanded to explicit article references before retrieval.

**Re-ingest needed**: Fix 1 requires `--reset`. Fixes 2–4 can upsert in-place (deterministic UUID5 IDs).

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
| 5 | **Single-article direct lookup only** | `_detect_direct_article_lookup()` routes to metadata-filtered retrieval only when exactly one article is mentioned. Multi-article queries may miss one article if cosine scores are unbalanced. | Partially addressed by Codex V2 finding #5 (see Medium Priority). |
| 6 | **No token budget management** | Total prompt tokens for synthesis are unbounded. No truncation before the OpenAI call. | Low risk at current corpus size. Add token budget cap if context window errors appear. |

---

## Codex Review V2 — New Findings (2026-03-17)

Reviewed `src/ingestion/eurlex_ingest.py`, `src/query/query_engine.py`, `src/indexing/index_builder.py`. Full report in `codex_review_v2.md`.

| # | Finding | Severity | File | Re-ingest? | Status |
|---|---------|----------|------|------------|--------|
| 1 | Annex overmatch: `^anx_` matches sub-annex IDs (`anx_IV.1`), creating duplicate Qdrant points | High | `eurlex_ingest.py:176` | Yes (`--reset`) | ❌ Open |
| 2 | Formula/text loss: `get_text()` strips `<img>` formulas; early return after first `<img>` drops surrounding prose | High | `eurlex_ingest.py:341–344, 468–472` | Yes (same UUIDs overwrite) | ❌ Open |
| 3 | Layout-A grid: nested `<div>` sub-points flattened; inner structure and formula placeholders lost | Medium | `eurlex_ingest.py:408–419` | Yes (same UUIDs overwrite) | ❌ Open |
| 4 | Cross-ref range parsing: plural/range forms (`Articles 89 to 91`) collapse to first number | Medium | `eurlex_ingest.py:611–614` | Yes (same UUIDs overwrite) | ❌ Open (also tracked in High Priority cross-ref section) |
| 5 | Direct lookup misclassifies external directive refs (`Article 10 of Directive 2014/59/EU`) and `Article 92 and 93` phrasing | Medium | `query_engine.py:96, 105–117` | No | ✅ Fixed 2026-03-17 |
| 6 | Cross-ref expansion cap applied to hash-ordered set; missing articles consume cap slots, later valid refs dropped | Medium | `query_engine.py:309–339` | No | ✅ Fixed 2026-03-17 |
| 7 | Stale `Settings._prompt_helper` cache: if an earlier path created a smaller prompt helper, synthesis repacks to wrong budget | Medium | `query_engine.py:535–569` | No | ✅ Fixed 2026-03-17 |
| 8 | `_configure_settings()` mutates global `Settings` without restore; later components inherit values by call order | Low | `index_builder.py:77–88` | No | ✅ Fixed 2026-03-17 |

**Confirmed / no finding:** circular-reference guard in `_expand_cross_references()` is correct; deterministic UUID5 IDs produce real overwrites for re-emitted documents; no direct GPT-4o overflow path in normal case (caveat: finding #7 above).

---

## High Priority

### Annex overmatch fix [Low effort, requires re-ingest with `--reset`]
`^anx_` regex in `eurlex_ingest.py:176` matches sub-annex IDs (`anx_IV.1`), so nested annex sections are indexed as separate documents alongside their parent annex — duplicate content and extra Qdrant points. Fix: restrict to `^anx_[^.]+$`. Add regression test with `anx_IV` + `anx_IV.1` fixture. Requires `--reset` because stale sub-annex points won't be removed by upsert.

### Formula and text preservation fix [Medium effort, requires re-ingest]
`_extract_structured_text()` loses formula-bearing text in two places (`eurlex_ingest.py:341–344, 468–472`): `get_text()` silently strips `<img>` formula elements; an early return after the first `<img>` drops surrounding prose and subsequent formulas. Fix: walk paragraph children token-by-token, emit text runs and `[FORMULA]` placeholders in order, remove early return. Add fixtures for `prefix <img> suffix` and multiple formulas in one paragraph. Same UUID5 IDs will overwrite existing points so `--reset` is not required.

### Layout-A nested grid flattening fix [Medium effort, requires re-ingest]
`_extract_structured_text()` Layout-A path (`eurlex_ingest.py:408–419`) flattens nested `<div>` grids with `cols[1].get_text(...)` instead of recursing. Inner sub-points survive only as flat text; nested formula placeholders are lost. Fix: mirror the Layout-B approach — extract direct row text, then recurse into child `<div>` blocks. Same UUID5 IDs will overwrite existing points.

### Golden dataset [High effort]
Curate 50–100 question/answer pairs for CRR covering: own funds, risk weights, leverage ratio, liquidity, large exposures, reporting. Each entry: `question`, `expected_articles`, `reference_answer`. Store as `evals/golden_dataset.jsonl`. Start with 20 high-value questions.

### Retrieval eval pipeline [High effort]
Implement `evals/run.py` — Recall@k, MRR, hit rate against golden dataset. Compare retrieved articles against `expected_articles`. Separate retrieval evaluation from answer quality evaluation. Add `evals/compare.py` for regression detection between runs.

### Cross-reference expansion improvements [High effort] — CRUCIAL

The current cross-reference system only handles **article-to-article** references via the `referenced_articles` metadata field. Three open gaps (all require re-ingest):

| # | Gap | What's missing | Requires re-ingest? | Priority |
|---|-----|---------------|---------------------|----------|
| 1 | **Structural ref extraction** | Part/Title/Chapter/Section/Annex refs not parsed at ingest time. Refs like "Part Six", "Title VII", "Chapter 1" are ignored. | **Yes** — ingest must extract and store structural refs in metadata (e.g. `referenced_parts`, `referenced_annexes`) | High |
| 2 | **Range ref parsing** | `Articles 89, 90 and 91` / `Parts Two to Five` not handled by regex. Only single `Article N` matches captured. Multi-letter suffixes (`92aa`) also truncated. | **Yes** (ingest regex) + **No** (query-time normalization) | High |
| 3 | ~~**Annex cross-ref linking**~~ | ~~Annexes exist as documents but no cross-references point to/from them.~~ | ~~**Yes** — ingest must capture annex refs and store in metadata~~ | ✅ Fixed 2026-03-17 |

**Operational note**: Gaps 1–3 require re-ingestion (metadata changes at parse time, no re-embedding needed — Qdrant supports payload updates).

---

## UI/UX — Phase 2

### Regulation reference tracking [Medium effort, no re-ingest needed]
`referenced_external` is already stored as a CSV of matched strings in Qdrant payloads. Work needed:
- **Backend**: Add `GET /api/article/{id}/external-refs` endpoint (or include in existing `ArticleResponse`) returning structured `{type: "Regulation"|"Directive", citation: "..."}` objects.
- **Frontend**: Render external regulation references as non-clickable badges (distinct style from internal article chips) in the article viewer and source chips.

### ~~Article cross-reference linkification improvements~~ ✅ Fixed 2026-03-17
Both `ProvisionText.tsx` bugs fixed via run-based parsing (`ART_RUN_RE`): external-ref exclusion now applied to the whole run; bare continuation numbers linkified individually. Backend `_extract_cross_references()` already used run-based parsing from the Codex V2 fix — no payload patch needed.

### Extract legal text parsing logic [TBD]
`ProvisionText.tsx` mixes text parsing (6+ regexes) with React rendering. Extract a `lib/legal-text-parser.ts` utility returning typed `StructuredLine[]` objects. External ref regex is also duplicated between frontend (TS) and backend (Python).

### Italian language parity audit [Medium effort]
Verify all English-side fixes (external ref exclusion, inline list splitting, cross-ref rendering, article viewer formatting) work correctly for Italian text. Italian uses different prepositions (`del/della/dello`), article keywords (`Articolo`), and legislative terminology (`Regolamento/Direttiva`).

### Split article handling audit [Low effort, investigation]
Verify `92a`/`92b`-style lettered sub-articles work end-to-end:
- **Ingestion**: are `art_92a`, `art_92b` div IDs parsed and stored as separate documents with `article="92a"` metadata?
- **Direct lookup**: does `_detect_direct_article_lookup("Explain Article 92a")` correctly extract `"92a"`?
- **Cross-reference expansion**: does `_expand_cross_references()` correctly fetch `92a`/`92b` from `referenced_articles` CSV?
- **Frontend**: does the article viewer handle `92a` correctly in chips and breadcrumbs?

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

### ~~Direct article lookup: external ref misclassification~~ ✅ Fixed 2026-03-17
### ~~Cross-ref expansion: non-deterministic cap~~ ✅ Fixed 2026-03-17
### ~~Stale PromptHelper cache~~ ✅ Fixed 2026-03-17

### Hybrid alpha tuning [Low effort, blocked by golden dataset]
Qdrant dense/sparse fusion weight (`alpha`) is the default — not tuned on CRR queries. Sweep `alpha` values (0.3–0.7) and measure Recall@6 on golden dataset.

### Experiment tracking [Medium effort]
Add `evals/config.json` per run capturing: embed model, top_k, cutoff, alpha, prompt hash, corpus version (CELEX + date). Store alongside results for regression detection.

### Index upsert mode [Medium effort]
Add `--upsert` mode to ingest pipeline that checks existing node_ids and only re-embeds changed articles. Add `--language-only-reset` to wipe one language without affecting others.

---

## Low Priority

### ~~`_configure_settings()` global mutation~~ ✅ Fixed 2026-03-17

### Embedding model benchmark [Medium effort, blocked by golden dataset]
Benchmark BGE-M3 vs. alternatives (e.g. `multilingual-e5-large-instruct`) on Recall@6 / MRR using golden dataset.

### Corpus dedup [Low effort]
Add post-parse dedup: hash article text, skip exact duplicates. Add whitespace normalization pass.

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
