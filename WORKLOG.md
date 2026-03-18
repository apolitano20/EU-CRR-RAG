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

### Conversational context / chat memory [High priority, Medium effort]

Currently every query is stateless — follow-up questions like "and what about y?" are sent as standalone queries with no knowledge of the prior exchange, producing poor results.

**Goal:** GPT-style continuous chat where follow-up questions are understood in context.

**Recommended approach — query rewriting before retrieval:**
1. **Frontend**: maintain conversation history (list of `{question, answer}` turns) and pass it alongside the new query in the request body.
2. **API**: add optional `history` field to `QueryRequest`.
3. **Backend (pre-retrieval)**: when history is present, call the LLM once to rewrite the follow-up into a fully self-contained query (e.g. "and what about y?" + prior context → "What are the CRR requirements for y, in the context of x?"). Then run normal retrieval on the rewritten query.
4. **Backend (synthesis)**: inject the last N turns into the synthesis prompt so the LLM can reference prior answers when composing its response.

Query rewriting is preferred over simply injecting history into retrieval because embedding quality degrades with conversational fragments — retrieval needs a well-formed standalone query.

**Scope:**
- Keep last N=5 turns (configurable) to avoid unbounded prompt growth.
- Rewriting step only fires when history is non-empty and the query looks like a follow-up (short, starts with pronoun/conjunction, no article reference). Otherwise skip it to save latency.

---

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

## UI/UX — Eval Feedback (shipped 2026-03-18)

### In-UI eval feedback capture ✅
Added per-message feedback mechanism to support eval case collection without leaving the browser.

**How it works:**
- Each Q&A pair in `ChatPanel` shows an "Add eval feedback" toggle below the answer
- Clicking it opens an amber-styled panel with a textarea and "Submit feedback" button
- The panel shows which article is currently open in the DocumentViewer (auto-included on submit)
- On submit: `POST /api/feedback` → backend writes `evals/cases/case_NNN.md`

**What gets saved per case:**
- Query asked
- LLM-generated answer
- Source articles with scores
- Full text of the article currently open in the DocumentViewer (if any)
- User's free-text feedback

**Files changed:**
- `api/main.py` — new `POST /api/feedback` endpoint + `FeedbackRequest`/`FeedbackResponse` models
- `frontend/src/lib/api.ts` — `submitFeedback()` function
- `frontend/src/components/chat/FeedbackBox.tsx` — new component
- `frontend/src/components/chat/ChatPanel.tsx` — renders `FeedbackBox` per message, receives `selectedArticle` prop
- `frontend/src/components/layout/AppLayout.tsx` — passes `selectedArticle` to `ChatPanel`

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

## Efficiency Improvements

Bottleneck analysis (2026-03-18): main latency sources in order of impact.

| # | Bottleneck | Current state | Improvement options | Effort |
|---|------------|--------------|---------------------|--------|
| 1 | **BGE-M3 CPU inference** | 570MB XLM-RoBERTa running on CPU; ~5–15s per query encoding | (a) **ONNX Runtime + INT8 quantization** — 2–4x speedup, same model, no GPU needed. (b) **Switch to `bge-small-en-v1.5`** (33MB) — much faster, some quality trade-off. (c) **GPU** — 10–50x if hardware available (code already has CUDA detection). | Medium |
| 2 | **OpenAI GPT-4o synthesis** | Blocking call, 5–30s; user sees nothing until complete | (a) **Stream tokens to frontend** — same total latency, but user sees text appear immediately. (b) **Switch to GPT-4o-mini** — 3–5x faster, cheaper, minimal quality loss for structured Q&A with retrieved context. | Low–Medium |
| 3 | **Qdrant cloud round-trips** | Multiple sequential calls per query (retrieval + cross-ref expansion); each ~1–3s network RTT | (a) **Local Qdrant** — eliminates network latency. (b) **Parallelise cross-ref expansion** — currently sequential; easy win with `ThreadPoolExecutor` or `asyncio.gather`. | Low |
| 4 | **Cold-start (first query)** | BGE-M3 loads from disk on first query (~25s) | Solved by GPU / ONNX above. Pre-warming at startup is an option but risks OOM if a prior process still holds the model. | — |

**Note:** C++/Rust rewrite is not the right lever here — PyTorch and the Qdrant client are already native C++/CUDA. Python overhead is negligible vs. model inference and network I/O.

**Recommended sequence:** (1) stream GPT-4o → instant UX win, minimal work; (2) GPT-4o-mini → faster + cheaper; (3) ONNX INT8 for BGE-M3 → biggest raw speedup on CPU; (4) parallelise cross-ref expansion.

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
