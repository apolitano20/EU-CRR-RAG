# Lessons Learned

---

## [2026-03-17] BeautifulSoup `<p>` handler: early-return after first `<img>` silently drops surrounding text
**Context:** Fixing formula/text preservation in `_extract_structured_text()` (Codex V2 Fix 2).
**What happened / insight:** The original pattern `img = elem.find("img"); if img: parts.append(placeholder); return` discards any text nodes before *and* after the formula in the same `<p>` tag. `get_text()` on the no-formula branch also silently drops `<img>` tags with non-data-URI srcs without warning.
**Take-away:** Always walk `elem.children` in document order when a `<p>` may contain a mix of text nodes and inline elements. Never early-return from a paragraph handler after extracting one element — the sibling content is just as important.

---

## [2026-03-17] Layout-A grid `get_text()` collapses nested sub-point structure
**Context:** Fixing Layout-A nested grid flattening in `_extract_structured_text()` (Codex V2 Fix 3).
**What happened / insight:** `cols[1].get_text(" ", strip=True)` flattens all descendant text into one string, losing sub-point label/text separation and formula placeholders inside nested `<div>` elements.
**Take-away:** When the content column of a grid row may contain nested `<div>` elements, classify children explicitly: collect direct text/inline into a prefix list, and call the recursive `walk()` for `<div>` children so they emit as separate entries. Never use `get_text()` where structured output is expected.

---

## [2026-03-17] Annex regex `^anx_` matches dot-separated sub-annex IDs
**Context:** Fixing annex overmatch in `_parse_with_beautifulsoup()` (Codex V2 Fix 1).
**What happened / insight:** EUR-Lex uses IDs like `anx_IV`, `anx_IV.1`, `anx_IV.1.a` — the dot indicates a sub-annex section nested inside the parent. The original `^anx_` regex matched all of them, creating duplicate Qdrant points for every annex that has sub-sections.
**Take-away:** When selecting top-level structural elements by ID, always anchor both ends of the pattern and exclude separator characters: `^anx_[^.]+$`. The same principle applies to any hierarchical ID scheme (e.g. `art_`, `prt_`). Fixing this requires `--reset` + full re-ingest because upsert cannot remove stale points whose IDs were generated from sub-annex div IDs.

---

## [2026-03-17] Cross-reference regex needs a "run" approach to handle ranges and lists
**Context:** Fixing cross-reference range parsing in `_extract_cross_references()` (Codex V2 Fix 4a).
**What happened / insight:** The original `(\d+[a-z]?)` pattern (a) limits letter suffixes to one character (`92a` captured but `92aa` truncated) and (b) cannot see range/list forms like "Articles 89 to 91" — only the first number is captured. The external-suffix exclusion check also only applies per-number, not per-run.
**Take-away:** Match the full article-reference *run* first with a pattern that includes `,|and|or|to` connectors, then extract all numbers from the run with a separate `\d+[a-z]*` pattern. Apply the external-suffix exclusion check once per run (at `run_m.end()`), not once per number. Expand "N to M" ranges into individual integers after confirming the run is internal.

---

## [2026-03-17] Query-time range expansion improves multi-article retrieval
**Context:** Adding `_expand_article_ranges()` to `_normalise_query()` (Codex V2 Fix 4b).
**What happened / insight:** A query like "Articles 89 to 91" hits the vector store as a literal phrase and BM25/dense retrieval may only surface article 89. Expanding the range to explicit "Article 89 Article 90 Article 91" tokens before embedding gives retrieval a chance to fetch each article individually.
**Take-away:** Apply a sanity cap (≤ 20 articles) to prevent runaway expansion for full-regulation queries like "Articles 1 to 575". Wire the expansion step *after* abbreviation expansion and article-shorthand normalization so the input to range expansion is already canonical.
