# Lessons Learnt

## 2026-03-17

### LlamaIndex `Settings` lazy resolver raises `ImportError` in restricted envs (e.g. Colab)

`Settings.embed_model`, `.llm`, `.chunk_size`, and `.chunk_overlap` are lazy properties that attempt to resolve the OpenAI default model. In environments without `llama-index-embeddings-openai` (e.g. a fresh Colab), accessing these public properties raises `ImportError` before `_configure_settings()` can set the BGE-M3 model. This was triggered inside `_settings_scope()` which snapshotted Settings attrs before the index build.

**Fix:** Access LlamaIndex `Settings` private backing attrs (`_embed_model`, `_llm`, `_transformations`) inside context-manager snapshots to bypass lazy resolution. Guard `chunk_size`/`chunk_overlap` with `try/except`. The permanent fix is adding `llama-index-embeddings-openai` to the install list — LlamaIndex imports it as a default even when a non-OpenAI model is used.

---

### BeautifulSoup `<p>` handler must walk children for formula-in-context preservation

Using `elem.find("img")` + early return inside the `<p>` handler silently discards any text before and after the formula. The fix is to iterate `elem.children` in document order, collecting text nodes and `<img>` placeholders into a token list. This pattern is consistent with how the rest of the `walk()` function handles div children.

### Layout-A grid-list `get_text()` collapses nested sub-point structure

`cols[1].get_text(" ", strip=True)` concatenates all descendants into one flat string, losing sub-point label/text separation and formula placeholders embedded in nested divs. The fix: classify `cols[1]` children — collect direct text/inline into a `col_parts` list (prefix), then call `walk()` on any `<div>` children so nested sub-points are emitted as separate `parts` entries.

### Annex regex overmatch: `^anx_` matches sub-annex IDs containing dots

`re.compile(r"^anx_")` matches `anx_IV.1`, `anx_IV.1.a`, etc. in addition to top-level annexes, creating duplicate Qdrant points. The safe pattern is `^anx_[^.]+$` (no dot in the ID suffix). Note: fixing this requires `--reset` + re-ingest because stale sub-annex points are not overwritten by upsert.

### Cross-reference regex needs a "run" approach, not single-number capture

The original `(\d+[a-z]?)` pattern (a) limits letter suffixes to one char (`92a` but not `92aa`) and (b) cannot see range/list forms like "Articles 89 to 91". The fix: match the full article-reference run first with a pattern that includes `,|and|or|to` connectors, then extract all numbers from the run with a separate `\d+[a-z]*` pattern and expand "to" ranges.

### Query-time range expansion improves multi-article retrieval

Queries like "Articles 89 to 91" hit the vector store as a literal phrase and miss articles 90–91. Expanding ranges to explicit "Article N" tokens in `_normalise_query()` before embedding gives BM25 and dense retrieval a chance to fetch each article individually. Cap expansion at 20 articles to avoid runaway query expansion.

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

---

### EUR-Lex class names differ from documentation (discovered 2026-03-12)

The consolidated CRR HTML (CELEX:02013R0575-20260101) uses **different class names** than older EUR-Lex documents. The actual classes are:

| Class | Content | Count |
|-------|---------|-------|
| `title-division-1` | Numbered heading: `PART ONE`, `TITLE I`, `CHAPTER 1`, `SECTION 1`, `SUB-SECTION 1` | 189 |
| `title-article-norm` | Article number: `Article 92`, `Article 114` | 741 |
| `stitle-article-norm` | Article subtitle (not needed for parsing) | 741 |
| `norm` | Body paragraph text | 11,486 non-empty |

**Not** `sti-art`, `ti-art`, `ti-section-*`, `normal` as in older documents.

The `title-division-1` class covers all hierarchy levels (PART/TITLE/CHAPTER/SECTION/SUB-SECTION) — the level is determined by the first word of the text content, not by a separate class per level.

Result: 742 article documents with full metadata (part, title, chapter, section, article) after the fix.

---

## 2026-03-12 — Multilingual EUR-Lex headings

### Italian heading keywords (verified against `crr_raw_ita.html`)

| Level | Italian keyword | Notes |
|-------|----------------|-------|
| PART | `PARTE` | All caps |
| TITLE | `TITOLO` | All caps |
| CHAPTER | `CAPO` | All caps |
| SECTION | `SEZIONE` | Title case (`Sezione 1`) — use `re.I` |
| SUB-SECTION | `SOTTOSEZIONE` | Title case — must be checked before SEZIONE |
| ARTICLE | `Articolo` | Title case |

Pattern order matters: check `SOTTOSEZIONE` before `SEZIONE` (prefix conflict).

### BGE-M3 sparse encoding
`FlagEmbedding.BGEM3FlagModel.encode()` returns `lexical_weights` as a dict of `{token_id: weight}`.
To pass to Qdrant `SparseVector`, convert keys/values to lists of `int`/`float` explicitly — Qdrant rejects numpy types.

### `--reset` destroys ALL languages in Qdrant
Unlike Chroma (local per-persist-dir), a single Qdrant collection holds all languages.
`--reset` drops the entire collection. Only use for a completely fresh multi-language ingest starting from EN.

---

## 2026-03-17 — LlamaIndex Document IDs must be deterministic UUIDs for Qdrant

### Non-deterministic Document IDs cause silent duplicate accumulation in Qdrant

`Document(text=text, metadata=meta)` without `id_=` lets LlamaIndex generate a **random UUID**
on every instantiation. When ingestion runs without `--reset`, Qdrant receives new point IDs
each time and accumulates duplicates (old IDs remain, new IDs are added alongside).

**Symptom**: Item count grows with each re-run. 745 + 745 = 1490 expected; got 2151 after
multiple runs without reset. `scripts/diagnose_qdrant.py` showed 337 duplicate `node_id`s.

**Fix**: Set `id_=_node_id_to_uuid(node.node_id)` so the same article always gets the same
Qdrant point ID → upserts overwrite instead of accumulate.

### Qdrant rejects non-UUID, non-integer point IDs

Passing `id_="art_92_en"` (a human-readable string) causes Qdrant HTTP 400:
`value art_92_en is not a valid point ID, valid values are either an unsigned integer or a UUID`.

**Fix**: Convert human-readable node_ids to deterministic UUIDs via `uuid.uuid5`:
```python
_NODE_ID_NAMESPACE = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")

def _node_id_to_uuid(node_id: str) -> str:
    return str(uuid.uuid5(_NODE_ID_NAMESPACE, node_id))
```
`uuid5` is SHA-1 based: same input → same UUID every time. The namespace UUID is arbitrary
but must be fixed — changing it would invalidate all existing point IDs.

### Colab tracks `main`; always push to `main` not `master`

GitHub's default branch (what `git clone` checks out) is `main`. If local work happens on
`master` and pushes go to `origin/master`, Colab's `git pull` will not pick up the changes.
Always verify with `git remote show origin | grep HEAD` and push to the correct default branch.

---

## 2026-03-15 — LlamaIndex chunking despite `transformations=[]`

### `transformations=[]` is falsy — LlamaIndex falls through to `Settings.transformations`

`VectorStoreIndex.from_documents()` contains:
```python
transformations = transformations or Settings.transformations
```
An empty list `[]` is **falsy** in Python, so `[] or Settings.transformations` evaluates to `Settings.transformations`, which defaults to `[SentenceSplitter()]`. Passing `transformations=[]` does **nothing** — the default splitter runs anyway.

**Symptom**: After a `--reset` + fresh ingest, Qdrant contains 2 000+ items instead of the expected ~1 490 (745 EN + 745 IT). Articles longer than the default 1 024-token chunk_size are split into multiple Qdrant points.

**Root fix**: Set `Settings.transformations = []` in `_configure_settings()`. Since `from_documents()` falls back to `Settings.transformations`, this ensures the fallback is also empty. Embedding still happens in `_add_nodes_to_index → _get_node_with_embedding` (not in the transformations pipeline), so setting this to `[]` does not break embedding.

**Secondary guard**: Keep `Settings.chunk_size = 8192; Settings.chunk_overlap = 0` as belt-and-suspenders.

**How to repair stale data**: Run `python -m src.pipelines.ingest_pipeline --reset` (EN first, then IT without `--reset`). Validate item count = ~1490.

**Diagnostic query** (check for any article with > 1 node):
```python
from src.indexing.vector_store import VectorStore
vs = VectorStore(); vs.connect()
payloads = vs.scroll_payloads(language="en")
from collections import Counter
counts = Counter(p.get("node_id") for p in payloads)
dupes = {k: v for k, v in counts.items() if v > 1}
print(dupes)  # should be empty after a clean ingest
```

---

## 2026-03-17 — LlamaIndex Settings gotchas

### `Settings.embed_model` and `Settings.llm` have validating property setters

Setting `Settings.embed_model = some_object` calls `resolve_embed_model(some_object)` which
does `assert isinstance(embed_model, BaseEmbedding)`. Passing a raw `object()` or a plain
`MagicMock()` raises `AssertionError` at assignment time.

`Settings.llm` has the same behaviour via `resolve_llm()`.

**Impact on tests:** You cannot use arbitrary sentinel objects as `Settings.embed_model` values
in unit tests. Options:
1. Use `None` for `llm` (LlamaIndex substitutes `MockLLM`; see next lesson).
2. Patch the entire `Settings` object (`patch("module.Settings")`) so property setters are
   bypassed entirely — useful when you only need to assert that a specific attribute was set.
3. Don't mock `BGEm3Embedding` with a plain `MagicMock()` — `BGEm3Embedding()` returns a real
   `BaseEmbedding` subclass instance even when `FlagEmbedding` is stubbed (the class still exists).

---

### `Settings.llm = None` stores `MockLLM`, not `None`

LlamaIndex intercepts the `None` assignment in `resolve_llm` and silently substitutes a
`MockLLM(...)` instance, printing "LLM is explicitly disabled. Using MockLLM."

**Impact on tests:** `assert Settings.llm is None` after `Settings.llm = None` always fails.
Use `assert Settings.llm is not orig_llm` to verify the LLM was changed, and
`assert Settings.llm is orig_llm` to verify restoration by `_settings_scope()`.

---

### `_expand_cross_references` `_seen` set excludes source article from refs_to_fetch

`_seen` is initialised from the articles of the source nodes passed in. References in
`referenced_articles` CSV that match the source article number are therefore excluded from
`refs_to_fetch`. When writing tests for cross-ref expansion, the refs CSV must not include the
source node's own article number — otherwise what looks like a 3-ref scenario is actually a
2-ref scenario at runtime, making cap-boundary assertions wrong.

---

## 2026-03-12 — Test suite

### `unittest.mock.patch()` with dotted submodule paths fails in Python 3.13

`patch("src.indexing.vector_store.VectorStore.connect")` raises `AttributeError: module 'src.indexing' has no attribute 'vector_store'` in Python 3.13. The mock resolution traverses the path via `pkgutil.resolve_name`, which requires each segment to already be an attribute of its parent — subpackages with empty `__init__.py` files don't satisfy this.

**Fix:** Use `patch.object(instance, "method_name")` (targeting the already-imported object) instead of string-path patches for subpackage targets. For class-level patching of subpackage modules, import the module explicitly first, then use `patch.object(module.ClassName, "method")`.

---

### Test sentences must actually contain the diacritics being tested

`_detect_language("Quali sono i requisiti di capitale?")` returns `None`, not `"it"`, because the sentence contains no Italian diacritics (àèéìíîòóùú). The detection heuristic checks character sets, not vocabulary.

**Fix:** Always use a sentence that contains at least one character from the target set — e.g. `"Qual è il requisito?"` for Italian, `"Jakie są wymogi?"` for Polish.

---

### Stub unavailable packages in `sys.modules` before importing app code in unit tests

When unit-test venv doesn't have heavy production packages (`qdrant-client`, `FlagEmbedding`, `llama-index-vector-stores-qdrant`), importing `api.main` (which transitively imports them) raises `ModuleNotFoundError` and blocks all API endpoint tests.

**Fix:** Add a `tests/unit/conftest.py` that injects `MagicMock()` stubs into `sys.modules` for these packages before any test module is imported. The stubs only need to provide the named constants used at import time (e.g. `Distance.COSINE`, `VectorParams`). This is safe because unit tests never call the real Qdrant/BGE-M3 code paths.

---

---

### Qdrant payload indexes required for metadata filtering

`MetadataFilters` on a field (e.g. `language`, `article`) fail with HTTP 400 unless a payload index
exists on that field in the collection:
```
Bad request: Index required but not found for "language" of one of the following types: [keyword].
```

**Fix:** Call `client.create_payload_index(collection_name=..., field_name=field, field_schema=PayloadSchemaType.KEYWORD)` for every field you filter on. This is idempotent — safe to call on existing collections. Add it to `VectorStore._ensure_payload_indexes()` and call it from `connect()` so it runs for both new and existing collections.

Fields to index for this project: `language`, `article`, `level`.

---

## 2026-03-12 — Major Redesign: DOM-based parser

### EUR-Lex HTML encodes full legal hierarchy in div IDs — use them directly

The consolidated CRR HTML has `<div id="art_92">` for every article, and the article's ancestor div has `id="prt_III.tis_I.cpt_1.sct_1"`. Parsing the ID string directly (split on `.`, strip prefix like `prt_`, `tis_`, `cpt_`, `sct_`) gives exact hierarchy with zero regex, zero text-based classification.

**Fix:** Replace the stateful CSS-class DOM walker with `soup.find_all('div', id=re.compile(r'^art_[^.]+$'))` + `_extract_hierarchy(parent.get('id'))`. This is multilingual by default (Italian/Polish HTML uses identical div IDs) and does not need `language_config.py` for the BeautifulSoup path.

---

### Removing AutoMergingRetriever requires removing the docstore dependency

`AutoMergingRetriever` requires a docstore populated with all nodes (not just leaf nodes) so it can "climb" the tree. When we switch to article-level documents (each article is a self-contained unit), the entire HierarchicalNodeParser + SimpleDocumentStore + docstore JSON persistence chain becomes dead code.

**Fix:** Remove `HierarchicalNodeParser`, `get_leaf_nodes`, `SimpleDocumentStore`, and all docstore persistence. Use `VectorStoreIndex.from_documents()` directly with article-level Documents. This also simplifies `load()` — no `persist_dir` needed.

---

### Cross-reference expansion requires the retriever to be re-built per-expansion

The `_expand_cross_references` method needs to build a `MetadataFilters`-constrained retriever for each referenced article. Use `vector_index.as_retriever(similarity_top_k=1, filters=filters)` directly — this is lightweight (no model reload) and can be called in a loop up to `max_cross_ref_expansions` times.

---

### LlamaIndex Qdrant adapter: `sparse_query_fn` calling convention

`QdrantVectorStore` calls `sparse_query_fn([query.query_str])` — passing a **list** with one element, not a bare string — and then indexes the result as `sparse_indices[0]` / `sparse_embedding[0]`.

Two bugs exist if you define the function incorrectly:

1. **Input double-wrapping** — if signature is `(query: str)` and you call `sparse_doc_fn([query])`, the input becomes `[["query"]]` (list of lists), which the BGE-M3 tokenizer rejects: `TypeError: TextEncodeInput must be Union[TextInputSequence, ...]`.

2. **Output single-unpack** — if you return `indices[0], values[0]` (a flat `list[int], list[float]`), LlamaIndex then indexes `[0]` into that and gets a bare `int`, which Pydantic's `SparseVector` rejects: `Input should be a valid list`.

**Fix:** Accept `list[str]` and return the full **batch** format — identical to `sparse_doc_fn`:
```python
def sparse_query_fn(query: list[str]) -> tuple[list[list[int]], list[list[float]]]:
    return sparse_doc_fn(query)
```

---

### Integration tests: use a dedicated test collection, never the production one

Integration tests that call `VectorStore.reset()` will drop **all data** in the target collection. Using the production `eu_crr` collection in tests would silently wipe the indexed corpus.

**Fix:** Integration test fixtures always pass `collection_name="eu_crr_test"` and tear down the collection in a `yield`-fixture finalizer. Production collection name is never referenced in tests.
