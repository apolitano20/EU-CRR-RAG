# Work Log — Open Tasks

For completed history see `completed.md`.

---

## High Priority

### Golden dataset [High effort]
Curate 50–100 question/answer pairs covering: own funds, risk weights, leverage ratio, liquidity, large exposures, reporting. Each entry: `question`, `expected_articles`, `reference_answer`. Store as `evals/golden_dataset.jsonl`. Start with 20 high-value questions.

### Retrieval eval pipeline [High effort]
Implement `evals/run.py` — Recall@k, MRR, hit rate against golden dataset. Compare retrieved articles against `expected_articles`. Separate retrieval evaluation from answer quality evaluation. Add `evals/compare.py` for regression detection between runs.

### Cross-reference expansion — structural refs [High effort, requires re-ingest]
Part/Title/Chapter/Section/Annex refs not parsed at ingest time. Refs like "Part Six", "Title VII", "Chapter 1" are ignored. Ingest must extract and store structural refs in metadata (e.g. `referenced_parts`, `referenced_annexes`). Payload update only, no re-embedding needed.

### Cross-reference expansion — annex linking [Medium effort, requires re-ingest]
Annexes exist as documents but no cross-references point to/from them. An article referencing "Annex I" doesn't trigger expansion. Ingest must capture annex refs and store in metadata.

---

## UI/UX — Phase 2

### Regulation reference tracking [Medium effort, no re-ingest]
`referenced_external` is stored as CSV in Qdrant payloads. Add `GET /api/article/{id}/external-refs` (or include in existing `ArticleResponse`) returning structured `{type, citation}` objects. Frontend: render external refs as non-clickable badges.

### Article cross-reference linkification improvements [Medium effort]
Two bugs in `ProvisionText.tsx`:
1. **External-ref exclusion too narrow** — only first article in a run excluded when followed by "of Directive/Regulation/…"; articles joined by "and" or comma before the suffix remain clickable. Requires backend `_extract_cross_references()` regex + payload patch via `scripts/fix_cross_refs.py`.
2. **"and N" suffix not linkified for internal refs** — trailing bare number after "and"/","  within an internal article-ref run is not clickable. Frontend-only fix.

### Italian language parity audit [Medium effort]
Verify all English-side fixes work for Italian text (external ref exclusion, inline list splitting, cross-ref rendering, article viewer formatting).

### Split article handling audit [Low effort, investigation]
Verify `92a`/`92b`-style lettered sub-articles work end-to-end: ingestion, direct lookup, cross-ref expansion, frontend chips/breadcrumbs.

### Clickable hierarchy breadcrumbs [Medium effort]
`DocumentBreadcrumb.tsx` renders breadcrumbs as plain text. Clicking "Chapter 6" should list articles. Requires `GET /api/navigator/{level}/{id}` endpoint.

### Relevant paragraph highlighting + auto-scroll [High effort]
When opening a cited article, highlight and scroll to the specific cited paragraph. Requires backend to return the node/paragraph ID used in retrieval.

### Hover previews for article references [High effort]
Tooltip on hover of inline "Article N" reference showing article title, first paragraph, "Open article" button. Requires lightweight preview API call.

---

## Research / Architecture Enhancements

### ToC-based query routing [Medium effort]
Secondary Qdrant index of CRR Table of Contents entries. On each query, retrieve relevant ToC entries and inject structural metadata filters into main retrieval.

### Extended thinking / model reasoning [Medium effort]
Leverage Claude extended thinking API for complex multi-hop queries. Detect query complexity; conditionally invoke for expensive cases only.

---

## Medium Priority

### Hybrid alpha tuning [Low effort, blocked by golden dataset]
Qdrant dense/sparse fusion weight (`alpha`) is the default — not tuned on CRR queries. Sweep `alpha` 0.3–0.7 and measure Recall@6 on golden dataset.

### Experiment tracking [Medium effort]
`evals/config.json` per run: embed model, top_k, cutoff, alpha, prompt hash, corpus version. Store alongside results for regression detection.

### Index upsert mode [Medium effort]
`--upsert` mode to re-embed only changed articles. `--language-only-reset` to wipe one language without affecting others.

---

## Low Priority

### Embedding model benchmark [Medium effort, blocked by golden dataset]
Benchmark BGE-M3 vs. alternatives (e.g. `multilingual-e5-large-instruct`) on Recall@6 / MRR using golden dataset.

### Corpus dedup [Low effort]
Hash article text, skip exact duplicates. Add whitespace normalization pass.

### Dynamic top_k [Low effort, blocked by golden dataset]
Tune top_k based on query complexity.

### Continuous eval [Medium effort, blocked by eval pipeline]
Nightly eval run against golden dataset; alert on regression > 5%.

---

## Deferred / Out of Scope

- **PGVector migration** — replace Qdrant Cloud with self-hosted PGVector for production
- **Rate limits + retry logic for LlamaParse** — only relevant if formula enrichment is used at scale
- **Polish ingestion** — blocked on sourcing `crr_raw_pl.html`
- **OCR / extraction error handling** — N/A (born-digital EUR-Lex HTML)
