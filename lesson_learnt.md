# Lessons Learnt

## 2026-03-12

### AutoMergingRetriever requires ALL nodes in the docstore

`AutoMergingRetriever` climbs the node hierarchy from leaf → parent using the `storage_context.docstore`.
If only leaf nodes are in the vector index (or no docstore is attached at all), merging silently produces no parent context.

**Fix:** Populate `SimpleDocumentStore` with ALL nodes (from `HierarchicalNodeParser`), attach it to `StorageContext`, persist with `storage_context.persist(persist_dir=...)`, and reload it via `StorageContext.from_defaults(persist_dir=...)` on subsequent runs.

---

### `Settings.llm` ordering matters — indexer resets it to `None`

`HierarchicalIndexer._configure_settings()` sets `Settings.llm = None` (intentionally, to prevent accidental LLM calls during indexing).
If `QueryEngine` calls `self._configure_settings()` before `self.indexer.load()`, the indexer then overwrites the OpenAI LLM with `None`, and synthesis silently fails (returns empty strings).

**Fix:** Always call `_configure_settings()` **after** `indexer.load()` in `QueryEngine.load()`.

---

### EUR-Lex HTML uses CSS classes, not semantic heading tags

EUR-Lex consolidated HTML does not use standard `<h1>`/`<h2>` tag hierarchy.
Legal structure is encoded in element CSS classes:

| Class | Level |
|-------|-------|
| `sti-doc`, `ti-section-1` | PART |
| `ti-section-2` | TITLE |
| `ti-section-3` | CHAPTER |
| `ti-section-4` | SECTION |
| `sti-art`, `ti-art` | ARTICLE heading |
| `normal` | Paragraph content |

**Fix:** Use a class-aware stateful DOM walker (`soup.find_all(True)` + check `element.get("class")`) instead of tag-based traversal.

---

### lxml vs html.parser

`BeautifulSoup(html, "lxml")` raises `FeatureNotFound` if `lxml` is not installed.
Always catch `FeatureNotFound` and fall back to `"html.parser"` so parsing works without the optional C extension.
