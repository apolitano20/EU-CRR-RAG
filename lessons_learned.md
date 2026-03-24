# Lessons Learned

---

## [2026-03-25] Mixed retrieval beats pure paragraph retrieval — deduplication is the key

**Context:** run_19 used PARAGRAPH-only retrieval and regressed badly overall (-9.8pp Hit@1 vs run_17) despite clear improvements on localized queries. The hypothesis was that finer chunks would improve precision.
**What happened / insight:** The failure mode was top-k flooding: articles with many paragraphs (e.g. liquidity, own_funds, large_exposures) consumed multiple top-k slots with closely-scored paragraph variants, crowding out other articles entirely. Switching to mixed mode (no chunk_type filter) and adding `ArticleDeduplicatorPostprocessor` — keeping only the best-scored chunk per article — eliminated flooding while preserving the paragraph-level precision signal for localized queries. The deduplicator uses a 2% margin to prefer ARTICLE chunks when scores are near-tied (avoiding unnecessary synthesis fetch round-trips). Result: run_20 beat run_17 by +6.4pp Hit@1 with zero regressions.
**Take-away:** When indexing multiple granularities of the same document, never retrieve across granularities without per-document deduplication. The right pattern is: retrieve mixed → deduplicate to one chunk per document → upgrade to full document text at synthesis. The paragraph chunks act as precision signals in the ranking race, not as the final synthesis unit.

---

## [2026-03-25] Two-population insight: ~15% localized queries need paragraphs, ~85% need full articles

**Context:** Analysing the category-level wins and losses between run_17 (article-only), run_19 (paragraph-only), and run_20 (mixed).
**What happened / insight:** The dataset splits into two populations. Localized queries (specific threshold, formula, or named sub-paragraph) benefit from paragraph-level retrieval because the relevant paragraph's dense vector outcompetes the full article blob. Broad queries (regulatory concepts, multi-hop, open-ended) need article-level context — paragraph chunks produce the correct ranking signal but the synthesis must still see the full article. The mixed mode with deduplication serves both: paragraph chunks can win the retrieval race for localized queries (signalling the right article with higher confidence), while article chunks remain competitive for broad queries.
**Take-away:** When experiments show some categories improve dramatically while others regress by similar magnitude, suspect a two-population problem rather than a single bad hyperparameter. Design a solution that routes or adapts per query rather than choosing one granularity globally.

---

## [2026-03-24] Windows/Python 3.13: `platform._wmi_query()` hangs after a crash — pre-populate the uname cache

**Context:** After the API server crashed mid eval-run, `python -m uvicorn api.main:app` would start the reloader but the worker never came up. `import llama_index.core` hung indefinitely in every diagnostic test.
**What happened / insight:** SQLAlchemy (`sqlalchemy/util/compat.py`) calls `platform.machine()` at import time. In Python 3.13 on Windows, `platform.machine()` → `platform.uname()` → `platform._wmi_query()`, which queries WMI (Windows Management Instrumentation) via COM. When WMI is in a bad state after a hard crash, this call blocks forever — no timeout. This silently kills every Python process that imports SQLAlchemy (which includes all of llama_index). Traced using `faulthandler.dump_traceback()` fired from a watchdog thread. Fix: at the very top of `api/main.py` (before any llama_index imports), pre-populate `platform._uname_cache` via `platform.uname_result.__new__(...)`. Guarded to `sys.platform == "win32"` only.
**Take-away:** If the API or any Python process hangs silently on startup after a crash on Windows/Python 3.13, suspect the WMI hang before anything else. The `platform._uname_cache` workaround is already in `api/main.py`. If the issue reappears in other entry points (e.g. ingest scripts), apply the same guard at the top of those files.

---

## [2026-03-24] Sequential timeout block in eval results = server crash, not query difficulty

**Context:** run_19 reported 85/173 failures (49%). The failures looked distributed across categories.
**What happened / insight:** The failed cases were cases 089–173 — a perfectly sequential block. All cases up to 088 succeeded; every case from 089 onwards timed out at 150s. This pattern unambiguously indicates the API server crashed or became unresponsive mid-run, not that those queries were harder or slower. The per-case timeout error message ("Future timed out after 150s") and the absence of any response in the logs confirmed it. The actual retrieval metrics on the 88 completed cases (MRR 0.742, Hit@1 72.7%) were perfectly healthy.
**Take-away:** When analysing eval failures, check case IDs first: a contiguous sequential block of failures starting from case N = server crash at that point. Scattered failures = genuine per-query errors (timeout, API error, parsing failure). Never assume a high failure rate means the config is broken without checking the failure ID pattern.

---

## [2026-03-24] `_extract_structured_text` must be called on a parent container, not on a norm div directly

**Context:** Implementing paragraph-level chunking — calling `_extract_structured_text(para_div)` for each `<div class="norm">` paragraph to extract its text for the PARAGRAPH embedding document.
**What happened / insight:** `_extract_structured_text` walks `container.children` and emits text when it encounters a `div.norm` element as a child. When called on the norm div itself, it walks inside the div and never triggers the norm-div handler — the function returns empty string for every paragraph. This was discovered only at test time; the ingest code ran silently, producing PARAGRAPH docs with empty text that were filtered out by `if doc.text.strip()`. Fix: wrap the norm div in a throwaway parent before calling the function: `_BS(f"<div>{para_div}</div>", "html.parser").find("div")`.
**Take-away:** `_extract_structured_text` is designed for article/annex containers, not for individual structural elements. Always wrap a single norm div in a parent div before passing it. When adding new call sites for this function, verify it produces non-empty output on a representative fixture rather than relying on it silently discarding the content.

---

## [2026-03-24] Contextual prefix on full-article blobs doesn't move retrieval metrics — gains are gated on chunking

**Context:** Implemented Codex rank-2 recommendation: prepend hierarchy breadcrumb (`Part > Title > Chapter > Article N — Title`) to every document's embedding text before re-ingesting.
**What happened / insight:** Overall Hit@1 was flat vs run_17, with regressions in `ciu_treatment` (-20pp) and `known_failures` (-8.3pp). The issue is that a 5-word prefix has negligible weight against a 500–1500 word article body in the embedding space. The paragraph-window reranker also shifts score distributions in ways that interact unpredictably with the prefix. The prefix hypothesis is sound — the information is now in the index — but the benefit only materialises when the indexed unit is a single paragraph (where the prefix is a significant fraction of the text), not a full article.
**Take-away:** Structural context prefixes are most effective on short chunks (paragraphs/points), not on long documents. When evaluating Codex rank-2 (prefix) and rank-3 (chunking) as separate experiments, expect rank-2 alone to be neutral and the real gain to come from rank-3. The two changes are synergistic — do them together in the next re-ingest cycle rather than sequentially.

---

## [2026-03-24] Qdrant _node_content field stores JSON, not raw text — parse before inspecting

**Context:** Writing validation code to confirm the contextual prefix landed in the Qdrant index after re-ingest.
**What happened / insight:** `payload["_node_content"]` contains a JSON string (`{"id_": ..., "metadata": {...}, "text": "..."}`) — it is LlamaIndex's full node serialization, not the raw document text. Checking `payload["_node_content"].startswith("Part")` always fails. The actual text is at `json.loads(payload["_node_content"])["text"]`.
**Take-away:** When spot-checking Qdrant payloads for text content, always do `json.loads(payload["_node_content"])["text"]`. Do not treat `_node_content` as a plain text field. The `display_text`, `article`, `language` etc. are stored as top-level payload keys and can be accessed directly.

---

## [2026-03-24] Paragraph-window reranker: scoring by best paragraph fixes false_friend but hurts diluted_embedding

**Context:** Implementing `ParagraphWindowReranker` based on the codex architectural review, which predicted that scoring whole articles was the main ranking bottleneck.
**What happened / insight:** The prediction was accurate for `false_friend` (+14.3pp) and `open_ended` (+4.1pp) — cases where the right paragraph was buried inside a long article. However, `diluted_embedding` regressed -16.7pp (6 cases). The likely reason: diluted_embedding failures happen when the right article has no paragraph that matches the query vocabulary — the vocabulary gap is at the article level. Scoring windows makes these cases harder, not easier, because no individual window surfaces the right signal. The codex review also flagged that diluted_embedding needs a different fix (HyDE or hierarchy-prefix embedding).
**Take-away:** Paragraph-window reranking is a reliable fix for ranking failures where the right article is retrieved but the wrong paragraph dominates the score. It is not a fix for vocabulary-gap retrieval failures. Track `diluted_embedding` separately in future evals and do not expect paragraph-level changes to help it.

---

## [2026-03-24] ToC routing at any threshold regresses overall — retire it

**Context:** Two experiments with ToC routing: run_15 (universal, all queries) and run_16 (selective, fires when max reranker score < 0.55). Both regressed vs run_12 (no ToC).
**What happened / insight:** Universal routing hurt high-confidence categories (liquidity, own_funds, threshold) more than it helped weak ones. Selective routing with a confidence threshold reduced the harm but couldn't eliminate it — even "low confidence" retrievals that triggered ToC often had correct top-1 results that got displaced by the LLM's routing suggestion. The latency overhead (+30% on run_15) is an additional cost.
**Take-away:** ToC-style LLM routing is high-variance. It helps on structurally-ambiguous queries but hurts when the retriever already has the right answer at low but real confidence. Do not reintroduce ToC routing unless it is strictly additive (i.e., only fires when retrieval truly failed, not just when confidence is below a threshold). Confidence thresholds are noisy proxies for retrieval failure.

---

## [2026-03-24] Auto-start API silently fails when old process holds the port — check netstat first

**Context:** Running `python -m evals.run_eval --auto-start-api` reported the API never became healthy within 120s.
**What happened / insight:** The old uvicorn process was still alive and holding port 8080. The auto-start spawns a new uvicorn with `stderr=DEVNULL`, so the "address already in use" error is completely silent. The new process exits immediately, and the runner waits 120s for a health check that never comes.
**Take-away:** Before running `--auto-start-api`, verify the port is free with `netstat -ano | grep :8080`. If a process is listed, kill it (`taskkill /PID <pid> /F`) before relying on auto-start. Alternatively, always start the API manually in a separate terminal where stdout/stderr are visible.

---

## [2026-03-24] Dashboard comparison tables that show only one metric hide gains in complementary metrics

**Context:** The `_comparison_table()` in `evals/dashboard.py` originally only showed Recall@3. After run_12, the adjacent tiebreaker produced +1.7pp Hit@1 and +14.3pp false_friend Hit@1, but the user saw "no improvement" in the dashboard because the tiebreaker only moves rank 1 vs rank 2 — Recall@3 is completely unaffected when the right article was already in top-3.
**What happened / insight:** A postprocessor that reorders rank-1 and rank-2 produces zero Recall@3 signal by definition. The entire gain is in Hit@1 and MRR. Showing only Recall@3 made a successful experiment look like a no-op.
**Take-away:** Comparison tables should always show at least Hit@1, Recall@3, and MRR together. When evaluating a rank-reordering change (tiebreaker, score blending), Hit@1 and MRR are the primary metrics; Recall@k only captures whether the right article is in the pool, not whether it's ranked first.

---

## [2026-03-24] run_10b was made redundant by run_2e — check active config before planning experiments

**Context:** run_10b was planned to test surgical synonyms after run_2e. When run_2e was actually run, its config showed `USE_ENRICHMENT=True` and the synonym code was already committed — meaning run_2e already captured the effect of the synonyms.
**What happened / insight:** Experiment plans written before code is merged can become stale. run_10b was designed when the synonyms were still a pending change; by the time run_2e ran, the synonyms were live in the codebase.
**Take-away:** Before launching a planned experiment, verify the active server config (dashboard config panel or `_config.json`) to confirm that the intended change is actually isolated. If the planned change is already baked into the baseline, the experiment is redundant — cross it off rather than running it.

---

## [2026-03-24] FastAPI response serialization failures bypass all custom exception handlers

**Context:** Debugging case_161 HTTP 500 — the error returned `content-type: text/plain` ("Internal Server Error") instead of JSON, even though both a `@app.exception_handler(Exception)` handler and an endpoint-level `except BaseException` block were in place.
**What happened / insight:** When FastAPI serializes the response model to JSON (via `jsonable_encoder`), this happens *after* the endpoint `return` statement, inside FastAPI's routing internals. Exceptions thrown there propagate directly to Starlette's `ServerErrorMiddleware` (the outermost layer), bypassing both the endpoint's `except` block and all registered `@app.exception_handler` handlers. The tell is `content-type: text/plain; charset=utf-8` with body "Internal Server Error" — that is Starlette's built-in fallback, not FastAPI's JSON error response. Debug by adding file writes inside the exception handlers; if nothing is written, the failure is in serialization, not in the handler.
**Take-away:** Never put non-JSON-serializable values in Pydantic response model fields. For `list[dict]` fields, ensure all numeric values are Python `float`/`int`, not numpy types. In particular, always wrap reranker scores with `float()` when building source dicts.

---

## [2026-03-24] numpy.float32 is not JSON serializable; float32 arithmetic propagates silently through score pipelines

**Context:** `_multi_query_retrieve` in `orchestrator.py` was the only path where reranked node scores were exposed directly in the HTTP response without `float()` conversion.
**What happened / insight:** `CrossEncoder.predict()` returns a numpy array of `float32`. `BlendedReranker` computes blended scores as `alpha * python_float + (1-alpha) * numpy.float32 = numpy.float32`. `round(numpy.float32, 4)` returns `numpy.float32` (not Python `float`). `json.dumps({"score": numpy.float32(...)})` raises `TypeError`. The regular `engine.retrieve` path already used `float(round(node.score or 0.0, 4))` — `_multi_query_retrieve` was the only place missing the `float()` wrapper. The bug only manifested for case_161 because it was the only golden-dataset query that is (a) classified `CRR_SPECIFIC` (no article number mentioned) AND (b) matches `_MULTI_HOP_RE` — the only path through `_multi_query_retrieve`.
**Take-away:** Whenever building a source/score dict that will be included in an HTTP response, always use `float(round(score or 0.0, 4))`, not just `round(...)`. Keep this consistent across all code paths (`_multi_query_retrieve`, ToC merge, engine.retrieve). Also: `numpy.float64` IS JSON serializable in Python's stdlib json; `numpy.float32` is NOT. If in doubt, always call `float()`.

---

## [2026-03-21] Shared mutable JSONL as write path + state model causes duplicate rows, overflow, and stale summaries

**Context:** Codex code review of `evals/run_eval.py` and `evals/dashboard.py` — eval pipeline producing inflated progress counts, duplicate result rows, and misleading summaries.
**What happened / insight:** All observed bugs traced to a single design choice: the `*_cases.jsonl` file was used simultaneously as the durable result store, the resume checkpoint, the progress counter, and the summary input. Once any of these roles produced corrupt data (duplicates from concurrent threads, malformed partial writes on crash), every downstream stage was poisoned. Concretely: (1) multiple threads opened the file in append mode with no lock — interleaved writes could corrupt rows; (2) the timeout branch wrote synthetic error rows for "stuck" futures, but `pool.shutdown(wait=False)` left those threads alive — a late-completing worker added a second row for the same ID; (3) progress bar used raw line count as the numerator, so duplicates pushed it above 100%; (4) summary aggregation consumed all rows including duplicates, distorting every mean.
**Take-away:** A JSONL file is an output format, not a state machine. If you need resumability, deduplication, and progress tracking, maintain them separately: (a) a single writer with a `threading.Lock` + a `_written_ids` set; (b) a per-run `state.json` written atomically (temp file + `os.replace()`); (c) progress counted as unique parsed IDs, not raw lines; (d) summary reload via a `_load_dataset` that deduplicates by ID. Generate the JSONL as a derived artefact — never read it back as the canonical truth without deduplication.

---

## [2026-03-21] `@st.cache_data` keyed only by path string returns stale data when files change in place

**Context:** `_load_run` and `_load_summary` in `evals/dashboard.py` decorated with `@st.cache_data(show_spinner=False)` and called with a string path argument.
**What happened / insight:** Streamlit's `@st.cache_data` caches the return value keyed by the function arguments. If the file at the same path is overwritten (e.g. a completed eval run replaces an in-progress one), the cache returns the old DataFrame/dict because the path string is unchanged. The user sees old summary numbers even after the run completes.
**Take-away:** For any cached loader of a mutable file, add a `file_mtime: float` parameter and pass `Path(p).stat().st_mtime` at the call site. Streamlit includes this in the cache key, so the cache invalidates whenever the file is modified. Do NOT use the `_`-prefix convention (Streamlit treats those as unhashed); use a normal float parameter name.

---

## [2026-03-21] Score blending for reranker: neutral at all alpha values when regressions are caused by strong reranker confidence

**Context:** Implementing `BlendedReranker` to fix two rank-flip regressions (case_136: 506c above 26; case_149: 395 above 392) caused by `ms-marco-MiniLM-L-6-v2` promoting adjacent/transitional articles.
**What happened / insight:** At alpha=0.3 (30% retrieval, 70% reranker), case_149 was fixed but case_152 broke (reranker demoted the correct article). At alpha=0.6 (60% retrieval, 40% reranker), case_149 was still fixed but both case_109 and case_152 regressed. Case_136 was never fixed at any alpha — the reranker's confidence in 506c over 26 was too strong for blending to overcome without harming other cases. Net Hit@1 was identical or worse vs pure reranker at every alpha value tried.
**Take-away:** Score blending is not the right fix when a general-purpose reranker is systematically wrong about a specific article pair (e.g. transitional provision vs primary article). The root cause is the reranker having no domain knowledge. Proper fixes: (a) domain-specific reranker fine-tuning with (query, correct_article, wrong_article) triplets, or (b) a heuristic rule targeting transitional articles specifically (e.g. deprioritise articles with "transitional" in their title when the query doesn't mention transitional provisions).

---

## [2026-03-21] `bge-reranker-v2-m3` SIGSEGVs when coloaded with BGE-M3 on Windows CPU; `ms-marco-MiniLM-L-6-v2` does not

**Context:** Unblocking Phase 1 reranker after installing CPU-only torch resolved the `shm.dll` DLL crash.
**What happened / insight:** `BAAI/bge-reranker-v2-m3` (via `FlagEmbeddingReranker`) caused SIGSEGV at inference time when BGE-M3 was already loaded in the same process on Windows CPU. The crash happened on the first query, not at startup. `cross-encoder/ms-marco-MiniLM-L-6-v2` (via `SentenceTransformerRerank` / `sentence-transformers.CrossEncoder`) coexists with BGE-M3 without any conflict — confirmed by loading both models sequentially and running inference.
**Take-away:** On Windows CPU, use `sentence-transformers` cross-encoders for reranking, not `FlagEmbedding`-based rerankers. The `ms-marco-MiniLM-L-6-v2` model is also faster (lighter model) and has no licensing issues. Set `RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2` in `.env`. The `BlendedReranker` class in `query_engine.py` also uses `CrossEncoder` directly, so it is safe on Windows.

---

## [2026-03-21] General-purpose cross-encoders promote semantically similar but legally wrong articles

**Context:** Analysing why reranker regressions occur on false-friend cases (case_136: 506c vs 26; case_149: 395 vs 392).
**What happened / insight:** `ms-marco-MiniLM-L-6-v2` was trained on MS MARCO web search queries. It scores relevance based on textual overlap and semantic similarity, not legal hierarchy. Article 506c (a transitional provision) contains language nearly identical to Article 26 (the primary rule it overrides). The reranker correctly identifies high textual overlap and ranks 506c first — but legally, 506c is secondary. Similarly, 395 (large exposure limits) contains much of the same vocabulary as 392 (definition of connected clients), causing a rank flip.
**Take-away:** General-purpose cross-encoders are blind to legal document hierarchy (primary article vs transitional provision, definition vs application rule). For legal RAG, accept these regressions as the cost of using a general reranker, or invest in fine-tuning with domain-specific (query, relevant, irrelevant) triplets. Document the known bad pairs: (case_136: 506c/26), (case_149: 395/392).

---

## [2026-03-20] PyTorch `shm.dll` crash on Windows blocks reranker — requires CPU-only torch or Linux

**Context:** Setting `USE_RERANKER=true` in `.env` and restarting the API server on Windows.
**What happened / insight:** `FlagReranker` import triggers `torch/__init__.py` → `_load_dll_libraries()` → `OSError: [WinError 1114] A dynamic link library (DLL) initialization routine failed` on `shm.dll`. The server crashes immediately on startup before serving any request. This is a PyTorch Windows shared-memory DLL incompatibility, not a code bug. Fix options: (a) reinstall CPU-only torch (`pip install torch --index-url https://download.pytorch.org/whl/cpu`), or (b) run on Colab/Linux where CUDA or CPU-only torch works without the DLL issue.
**Take-away:** Phase 1 reranker cannot be tested locally on Windows. Revert `USE_RERANKER=false` immediately if the server fails to start. The rest of the eval infrastructure (eval runner, dashboard) works locally; only the reranker step needs Colab/Linux.

---

## [2026-03-20] "Missing articles" is rarely the cause of hard failures — check the exception first

**Context:** 13 hard failures all with `status=error`, suspected articles 115, 121, 122, 132a, 132c, 152, 242, 243, 254, 258, 429b missing from the index.
**What happened / insight:** All 11 articles were in the index with correct node counts. The hard failures were code crashes (unhandled exceptions), not missing data. The original `api/main.py` caught `HTTPException` only; any other exception produced a silent HTTP 500 with no logged traceback. Adding a broad `except Exception` with `exc_info=True` is needed before any diagnosis is meaningful.
**Take-away:** Before concluding data is missing, always: (1) add `except Exception` with `exc_info=True` in the query handler, (2) reproduce a single failing case and read the server log. Only investigate the index if the log shows a retrieval/lookup error specifically. Missing-data hypotheses waste time when the real cause is an unhandled exception.

---

## [2026-03-20] `as_completed()` never yields a hung future — use `wait()` with timeout instead

**Context:** Eval runner stuck at N/50; Codex Review 2 diagnosed the root cause in `evals/run_eval.py`.
**What happened / insight:** `for fut in as_completed(futures)` only yields futures that have *already completed*. A future that is stuck (e.g. a socket that never times out) is never yielded, so `fut.result(timeout=...)` inside the loop body is never reached for it. The loop blocks inside `as_completed()` forever. Additionally, the `with ThreadPoolExecutor(...) as pool:` context manager calls `shutdown(wait=True)` on exit, which also blocks on any still-running thread. The fix: replace `as_completed` with `concurrent.futures.wait(pending, timeout=_fut_timeout, return_when=FIRST_COMPLETED)` in a `while pending` loop. When `done_set` is empty, all remaining futures are stuck — record timeout errors and break. Use explicit pool + `pool.shutdown(wait=False)` in `finally` to avoid the context-manager hang.
**Take-away:** Never rely on `fut.result(timeout=...)` inside an `as_completed` loop as a hang guard — `as_completed` won't yield the hung future in the first place. The only safe pattern for per-future timeouts is a `wait(..., timeout=T, return_when=FIRST_COMPLETED)` polling loop that detects `done_set=empty`.

---

## [2026-03-20] FastAPI sync endpoints saturate the thread pool under parallel load

**Context:** `/api/query` defined as `def query(...)` (synchronous); eval runner fires 4 parallel requests.
**What happened / insight:** FastAPI runs sync endpoints in its worker thread pool. If several calls block in BGE-M3 inference or Qdrant round-trips, those worker threads stay occupied. Once the pool is saturated, later requests queue *before* the handler body runs — so they look like "server stopped responding" rather than a 5xx. The logging middleware (which only fires after `call_next()` returns) makes queued/hung requests invisible in logs.
**Take-away:** Any endpoint that calls blocking I/O or CPU-heavy work must be `async def` with `asyncio.to_thread(...)`. Wrap with `asyncio.wait_for(..., timeout=N)` and return 504 on expiry so the client connection is never held open indefinitely. This is a mitigation (the underlying thread still runs), but it frees the event loop and unblocks later requests.

---

## [2026-03-20] Concurrent CPU BGE-M3 encodes cause latency amplification, not just slowness

**Context:** Eval runner with `--workers 4` firing parallel queries; each query calls `_get_model().encode()` for both sparse and dense vectors.
**What happened / insight:** FlagEmbedding is not documented as re-entrant. Under 4 concurrent CPU encode calls, the model oversubscribes available cores, causing all calls to slow down dramatically (10–60× normal latency). This is not a deadlock, but the wall-clock spike is large enough that HTTP timeouts fire, making it *look* like a deadlock. Adding `_encode_lock = threading.Lock()` around every `encode()` call serializes them: throughput drops but latency becomes predictable, and requests that previously timed out now succeed.
**Take-away:** On CPU, serialize BGE-M3 `encode()` calls with a module-level lock. The tradeoff (throughput ↓, latency predictable) is worth it for eval stability. On GPU this lock can be removed once concurrency is validated.

---

## [2026-03-20] `subprocess.PIPE` without a drain thread blocks child processes under load

**Context:** `evals/run_eval.py --auto-start-api` starts uvicorn with `stdout=PIPE, stderr=STDOUT`; parent never reads the pipe.
**What happened / insight:** OS pipe buffers are typically 64 KB. When uvicorn logs more than that (which happens quickly under load), it blocks on the write syscall, appearing as the API "freezing". The parent process never reads the pipe, so the buffer never drains. The symptom is indistinguishable from a real API hang.
**Take-away:** If you don't intend to read a subprocess's stdout/stderr, always redirect to `DEVNULL` (`stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL`). Never use `PIPE` for output you won't consume.

---

## [2026-03-20] LLM-judge metrics in compare.py: skip rows when both runs are None, not percentage-formatted

**Context:** Extending `evals/compare.py` `METRIC_KEYS` to include judge metrics (`judge_correctness` etc.) and updating `print_report()`.
**What happened / insight:** Two non-obvious details arise when judge metrics share the same `METRIC_KEYS` list as retrieval metrics: (1) Retrieval metrics are percentage-formatted (`* 100`) in print/display; judge metrics are raw 0–1 floats and must not be multiplied. The `as_pct` flag must exclude all `JUDGE_METRIC_KEYS`. (2) When neither compared run used `--judge`, all judge metric deltas are `None`. Printing `—` for every judge row is noisy. The fix is to skip judge rows in `print_report()` when `va is None and vb is None`.
**Take-away:** When sharing a metric-key list across heterogeneous metric types (percentage vs raw score), always check the `as_pct` logic for every new metric type. Conditionally skip rows with all-None values in text reports; hide sections in UIs.

---

## [2026-03-19] LLM-generated eval datasets: skip giant articles, tag adversarial batches in `notes`

**Context:** Building `evals/generate_golden_dataset.py` to auto-generate a golden dataset from Qdrant.
**What happened / insight:** Article 4 is 60k chars (129 definitions) — sending it to GPT in a prompt hits context limits and produces useless output (it would just summarise definitions, not generate meaningful retrieval test cases). It also has a dedicated fast-path in the query engine, so RAG eval doesn't apply. Similarly, adversarial batch cases (Pass 2) need to be distinguishable from article-anchored cases (Pass 1) for resumability checks — tagging `notes` with `batch=<id>` lets the script skip already-processed batches on re-run without a separate tracking file.
**Take-away:** Always skip articles with known special handling (fast-paths, giant size) from RAG eval generation. For resumability in multi-pass generators, embed a stable identifier in a freetext field of the output record (e.g. `notes`) rather than maintaining a separate state file — it survives crashes and partial outputs equally well.

---

## [2026-03-19] Qdrant payload text lives in `_node_content` JSON blob, not top-level `text`

**Context:** Extracting article text from Qdrant payloads in `generate_golden_dataset.py` and `definitions_store.py`.
**What happened / insight:** LlamaIndex stores node text inside a `_node_content` JSON blob (e.g. `{"text": "...", "metadata": {...}}`), not in a top-level `text` field. Some payloads have both (top-level `text` is populated for some node types); others have only `_node_content`. Any code that reads Qdrant payloads directly (outside LlamaIndex) must handle both patterns.
**Take-away:** Always use this two-step fallback when extracting text from raw Qdrant payloads: `text = payload.get("text", "") or ""; if not text: text = json.loads(payload.get("_node_content", "{}")).get("text", "")`. This pattern is the canonical one for this codebase — see `definitions_store.py:163-172` and `generate_golden_dataset.py:extract_text_from_payload()`.

---

## [2026-03-18] `_DEF_QUERY_RE` is a false-positive trap for abbreviation-expanded queries

**Context:** Writing `classify()` in `QueryOrchestrator` — calling `_normalise_query()` before `_detect_definition_query()` so that article refs are canonicalised before classification.
**What happened / insight:** "What is CET1?" normalises to "What is CET1 (Common Equity Tier 1)?". The parenthetical `(Common Equity Tier 1)` contains `(` which is not in the term capture group `[\w\s\-]*?`, so the regex cannot advance past it and fails to match. The query that looks like a definition query becomes CRR_SPECIFIC post-normalisation. This is the *correct* outcome (the term is already expanded), but tests that expect `classify("What is CET1?") == DEFINITION` will fail.
**Take-away:** When writing classification tests involving abbreviation-expanded queries, check what `_normalise_query` produces first. "What is CET1?" → CRR_SPECIFIC (because expansion breaks the regex); "What is the definition of institution?" → DEFINITION (no expansion, clean term). Use unexpanded terms or `define X` phrasing when you need a reliable DEFINITION classification in tests.

---

## [2026-03-18] Tests referencing module-level singletons must be updated when singletons change

**Context:** Refactoring `api/main.py` to replace `_query_engine.query()` routing with `_orchestrator.query()`. `test_api_endpoints.py` mocked `mod._query_engine.query` in every test.
**What happened / insight:** After the refactor, all `TestQueryEndpoint` and `TestQueryHistoryField` tests that set `mod._query_engine.query = MagicMock(...)` still "worked" at mock-assignment time but `_orchestrator.query()` was never intercepted — the orchestrator held its own reference to the engine and called its own `.query()` method, bypassing the mock entirely. Tests either got `AttributeError` (as_retriever on None) or returned unexpected results.
**Take-away:** When a new facade/orchestrator wraps an existing singleton, update all test fixtures that mock the old singleton's methods to mock the facade's methods instead. Also update the `client` fixture to patch the facade's `.load()` rather than the underlying engine's `.load()`.

---

## [2026-03-18] f-string expressions cannot span adjacent string literal boundaries

**Context:** Writing `yield` statements in `query_stream()` with long `json.dumps(dict)` calls that exceeded one line.
**What happened / insight:** Writing `f"data: {json.dumps({'key': val, " f"'key2': val2})}\n\n"` (split across two adjacent f-string literals) is a syntax/runtime error — the `json.dumps(...)` expression is not split at a valid Python boundary. Adjacent string literal concatenation is resolved at compile time per literal; f-string expressions `{...}` are local to each literal.
**Take-away:** For long `json.dumps` calls inside f-strings, always build the dict as a variable first, then do `f"data: {json.dumps(event)}\n\n"` on a single line. Never split a `{expression}` across adjacent f-string literals.

---

## [2026-03-18] Regex with `[\w\s]` capture + lazy quantifier + `\?`/`$` end anchor matches arbitrarily long phrases

**Context:** Writing `_DEF_QUERY_RE` to detect definition queries like "what is institution?" — the term capture group used `[\w\s\-]*?` (lazy) with end condition `\?|$`.
**What happened / insight:** "What are the CET1 requirements under Article 92?" matched because the lazy quantifier expanded "CET1 requirements under Article 92" (5 words) until `\?` was satisfied at the sentence-final `?`. Any pattern of the form `what is/are X?` where X can include spaces will over-capture unless the term is bounded.
**Take-away:** After extracting a term with a regex that allows embedded spaces, apply a word-count guard as a post-capture sanity check: `if len(term.split()) > 4: return None`. Real CRR definition terms top out at 4–5 words; longer captures are false positives.

---

## [2026-03-18] Test sample text must not contain incidental occurrences of the split token

**Context:** Writing `TestParse` for `DefinitionsStore._parse()`, which splits Article 4 text on `\((\d+)\)\s+` boundaries.
**What happened / insight:** The sample text for definition 3 contained `"of Article 4(1) of Directive 2014/65/EU"`. The `(` before `1` is preceded by `4` which is `\w`, so `(?<!\w)\(1\)` should not match — but `"point (1) of"` (from the original phrasing "as defined in point (1) of") does match because `(` there is preceded by a space. This created a spurious 5th definition, breaking count/index assertions.
**Take-away:** When writing parser unit tests, audit the sample text for any incidental occurrence of the split pattern — not just in the obvious places. For `_DEF_SPLIT_RE` specifically, avoid embedded `(N) keyword` phrases (parenthesised number followed by a space) anywhere in test fixture text that isn't a real definition boundary.

---

## [2026-03-18] All `__new__`-based test helpers must be updated when new instance attributes are added

**Context:** Adding `self._defs` to `QueryEngine.__init__` to hold a `DefinitionsStore`. Existing `TestSynthesisNodeMerging._build_query_engine()` creates a `QueryEngine` via `object.__new__(QueryEngine)`, bypassing `__init__`.
**What happened / insight:** Three tests failed with `AttributeError: 'QueryEngine' object has no attribute '_defs'` because the `__new__`-based helper didn't set it. The same issue would arise for any test helper that uses `__new__` to instantiate the class.
**Take-away:** After adding any new `self.X = ...` to a class's `__init__`, grep for `__new__(ClassName)` and `QueryEngine.__new__` (or equivalent) in the test suite and add the new attribute there too. A missing attribute in a `__new__`-constructed instance will always surface as `AttributeError` at runtime, not at construction time.

---

## [2026-03-18] Tests that assert on cross-ref fetch order must be updated when an article is skipped

**Context:** `TestExpandCrossReferences.test_refs_fetched_in_numeric_order` used refs `"114,26,4"` and asserted `fetched == ["4", "26", "114"]`. After adding the Article 4 skip guard in `_expand_cross_references()`, the test broke.
**What happened / insight:** Behavioural guards (skip Article 4, skip already-seen refs, etc.) change the expected fetch sequence. Tests that pin the exact list of fetched articles need to be updated whenever the guard list changes.
**Take-away:** When adding a skip rule to `_expand_cross_references()`, immediately update any test whose ref fixture includes the skipped article in the expected output list. Add a comment in the test explaining why that article is absent: `# Article 4 skipped by definitions fast-path guard`.

---

## [2026-03-18] `import openai` and `from llama_index.llms.openai import OpenAI` coexist without conflict
**Context:** Adding the raw OpenAI sync client to `query_engine.py`, which already imports LlamaIndex's `OpenAI` wrapper.
**What happened / insight:** `import openai` (the package) and `from llama_index.llms.openai import OpenAI` (a class) live in different namespaces. You call the raw client as `openai.OpenAI(...)` and the LlamaIndex wrapper as `OpenAI(...)`. They do not shadow each other. Test patch target for the raw client is `src.query.query_engine.openai.OpenAI`.
**Take-away:** Import the raw `openai` package as `import openai` when you need it alongside LlamaIndex's wrapper; call it as `openai.OpenAI(...)` to avoid naming ambiguity. Always patch `src.query.query_engine.openai.OpenAI` (not `openai.OpenAI`) in unit tests.

---

## [2026-03-18] Tests that assert `engine.synthesize()` was called must be updated when switching to direct OpenAI calls
**Context:** Migrating `QueryEngine.query()` from `engine.synthesize()` to direct `openai.OpenAI(...).chat.completions.create()`.
**What happened / insight:** Three existing `TestSynthesisNodeMerging` tests asserted that `engine.synthesize()` was called with specific node lists. After the migration those calls never happen, so the tests failed with `AuthenticationError` (the real OpenAI client was invoked with the fake `"test"` key). The fix was to mock `openai.OpenAI` in each test and check the prompt content passed to `chat.completions.create()` instead.
**Take-away:** When changing the synthesis path, immediately search for tests that assert on the old path (e.g. `mock_engine.synthesize.call_args`) and update them to mock the new path before running the suite.

---

## [2026-03-18] LlamaIndex synthesizer is synchronous — bypass it for streaming
**Context:** Implementing GPT-4o token streaming on the `/api/query/stream` endpoint.
**What happened / insight:** LlamaIndex's `engine.synthesize()` is a blocking synchronous call that returns a complete response object. There is no streaming variant in the retriever-query-engine path. To stream tokens, you must call `AsyncOpenAI` directly, build the prompt manually from the retrieved nodes, and stream the completion.
**Take-away:** When adding streaming to any LlamaIndex-based pipeline, plan to bypass synthesis and call the LLM SDK directly. Extract retrieval as a separate reusable method first.

---

## [2026-03-18] Sync retrieval blocks the async event loop — use asyncio.to_thread
**Context:** The FastAPI streaming endpoint is `async def`, but `QueryEngine.retrieve()` is synchronous (Qdrant + BGE-M3 calls).
**What happened / insight:** Calling a long-running sync function directly inside an `async` FastAPI handler blocks the entire event loop, preventing other requests from being served during retrieval (~5–15s). The fix is `await asyncio.to_thread(sync_fn, *args)`.
**Take-away:** Any sync I/O or CPU-bound work inside an async FastAPI handler must be wrapped with `asyncio.to_thread()`. This applies to Qdrant queries, BGE-M3 inference, and anything else that blocks.

---

## [2026-03-18] SSE token content must be JSON-encoded to handle special characters
**Context:** Building the SSE `data:` lines for streaming tokens.
**What happened / insight:** GPT-4o tokens can contain newlines, quotes, and backslashes. Concatenating them raw into `f"data: {{'type': 'token', 'content': {delta}}}\n\n"` breaks JSON parsing on the frontend when the token contains a quote or backslash.
**Take-away:** Always use `json.dumps({'type': 'token', 'content': delta})` when building SSE data lines — never string-format the token content directly.

---

## [2026-03-18] Frontend SSE: buffer across chunk boundaries before parsing
**Context:** Reading the SSE stream in `postQueryStream` with `ReadableStream`.
**What happened / insight:** Network chunks do not align with SSE event boundaries. A single `data: ...\n\n` event can be split across two `reader.read()` calls. Naive `\n\n` splitting on each chunk will drop or corrupt events.
**Take-away:** Maintain a string `buffer` across reads. After each `read()`, append decoded bytes to the buffer, then split on `"\n\n"`, process all complete events, and keep the last (potentially incomplete) fragment in the buffer for the next iteration.

---

## [2026-03-18] Multilingual regex: extend keywords AND their surrounding grammar together

**Context:** Extending `ART_RUN_RE` and `EXTERNAL_CONTEXT_RE` in `legal-text-parser.ts` to support Italian article references.

**What happened / insight:** Adding `Articol[oi]` to match Italian article keywords without also adding Italian conjunctions (`e`, `o`) would have left "Articoli 92 e 93" only partially matched — the `e 93` tail would be silently dropped, producing a single-article link instead of two. Similarly, extending `EXTERNAL_CONTEXT_RE` for Italian keywords without adding the Italian preposition variants (`del/della/dello/dell'`) would still have linkified "Articolo 4 del Regolamento" as a CRR internal link. In both cases the keyword variant and its grammatical context form an inseparable unit.

**Take-away:** When adding multilingual support to a regex, audit the full grammatical context — prepositions, conjunctions, and inflections — not just the keyword stem. Write a test for each new grammatical form before shipping.

---

## [2026-03-18] Vitest is the right test runner for Next.js TypeScript libraries

**Context:** Adding a unit test file for `legal-text-parser.ts`; `frontend/package.json` had no test runner.

**What happened / insight:** Next.js projects don't ship with a test runner. Jest requires `ts-jest` or Babel config to handle TypeScript; Vitest works out of the box on `.ts` files with zero config and runs in `~380 ms` for a pure-logic module with no DOM dependency.

**Take-away:** For frontend utility modules (no React, no DOM), prefer `vitest` with no config file. Add `"test": "vitest run"` to `package.json` scripts.

---

## [2026-03-17] LlamaIndex `Settings` lazy resolver raises `ImportError` in restricted envs (e.g. Colab)

`Settings.embed_model`, `.llm`, `.chunk_size`, and `.chunk_overlap` are lazy properties that attempt to resolve the OpenAI default model. In environments without `llama-index-embeddings-openai` (e.g. a fresh Colab), accessing these public properties raises `ImportError` before `_configure_settings()` can set the BGE-M3 model. This was triggered inside `_settings_scope()` which snapshotted Settings attrs before the index build.

**Fix:** Access LlamaIndex `Settings` private backing attrs (`_embed_model`, `_llm`, `_transformations`) inside context-manager snapshots to bypass lazy resolution. Guard `chunk_size`/`chunk_overlap` with `try/except`. The permanent fix is adding `llama-index-embeddings-openai` to the install list — LlamaIndex imports it as a default even when a non-OpenAI model is used.

---

## [2026-03-17] BeautifulSoup `<p>` handler must walk children for formula-in-context preservation

Using `elem.find("img")` + early return inside the `<p>` handler silently discards any text before and after the formula. The fix is to iterate `elem.children` in document order, collecting text nodes and `<img>` placeholders into a token list. This pattern is consistent with how the rest of the `walk()` function handles div children.

---

## [2026-03-17] Layout-A grid-list `get_text()` collapses nested sub-point structure

`cols[1].get_text(" ", strip=True)` concatenates all descendants into one flat string, losing sub-point label/text separation and formula placeholders embedded in nested divs. The fix: classify `cols[1]` children — collect direct text/inline into a `col_parts` list (prefix), then call `walk()` on any `<div>` children so nested sub-points are emitted as separate `parts` entries.

---

## [2026-03-17] Annex regex overmatch: `^anx_` matches sub-annex IDs containing dots

`re.compile(r"^anx_")` matches `anx_IV.1`, `anx_IV.1.a`, etc. in addition to top-level annexes, creating duplicate Qdrant points. The safe pattern is `^anx_[^.]+$` (no dot in the ID suffix). Note: fixing this requires `--reset` + re-ingest because stale sub-annex points are not overwritten by upsert.

---

## [2026-03-17] Cross-reference regex needs a "run" approach, not single-number capture

The original `(\d+[a-z]?)` pattern (a) limits letter suffixes to one char (`92a` but not `92aa`) and (b) cannot see range/list forms like "Articles 89 to 91". The fix: match the full article-reference run first with a pattern that includes `,|and|or|to` connectors, then extract all numbers from the run with a separate `\d+[a-z]*` pattern and expand "to" ranges.

---

## [2026-03-17] Query-time range expansion improves multi-article retrieval

Queries like "Articles 89 to 91" hit the vector store as a literal phrase and miss articles 90–91. Expanding ranges to explicit "Article N" tokens in `_normalise_query()` before embedding gives BM25 and dense retrieval a chance to fetch each article individually. Cap expansion at 20 articles to avoid runaway query expansion.

---

## [2026-03-17] LlamaIndex Document IDs must be deterministic UUIDs for Qdrant

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

## [2026-03-15] LlamaIndex chunking despite `transformations=[]`

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

---

## [2026-03-17] LlamaIndex Settings gotchas

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

## [2026-03-12] Test suite

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

### Qdrant payload indexes required for metadata filtering

`MetadataFilters` on a field (e.g. `language`, `article`) fail with HTTP 400 unless a payload index
exists on that field in the collection:
```
Bad request: Index required but not found for "language" of one of the following types: [keyword].
```

**Fix:** Call `client.create_payload_index(collection_name=..., field_name=field, field_schema=PayloadSchemaType.KEYWORD)` for every field you filter on. This is idempotent — safe to call on existing collections. Add it to `VectorStore._ensure_payload_indexes()` and call it from `connect()` so it runs for both new and existing collections.

Fields to index for this project: `language`, `article`, `level`.

---

## [2026-03-19] `_retrieve_with_filters`: try DEFAULT before HYBRID to avoid wasted BGEm3 encode calls

**Context:** Direct article lookups and cross-reference expansion use `_retrieve_with_filters`, which originally tried HYBRID mode first.

**What happened / insight:** HYBRID mode requires **two separate** BGEm3 `model.encode()` calls: one for the dense vector (`BGEm3Embedding._get_query_embedding`) and one for the sparse vector (`sparse_query_fn`). Each is a CPU-intensive model forward pass (~2–5 s on CPU). Moreover, HYBRID mode with an exact-match metadata filter (article=X) almost always returns **empty results** — the sparse ANN index finds no approximate neighbours under tight metadata constraints. The code then falls back to DEFAULT mode, which runs yet another BGEm3 encode. Net result: 3 encode calls for the common case, wasting 6–15 s before the actual retrieval succeeds.

**Fix:** Swap the order in `_retrieve_with_filters` to `(DEFAULT, HYBRID)`. DEFAULT (dense-only) applies the metadata filter as a post-filter and reliably returns results. HYBRID is kept as a fallback for semantically loose queries that might benefit from sparse recall.

**Impact:** Direct article lookup latency dropped from ~30–60 s (biblical) to ~5–10 s on CPU. Cross-reference expansion (3 articles × DEFAULT-first) similarly improved.

**Take-away:** For metadata-filtered retrievals (exact article/annex lookup), DEFAULT mode is always the right first choice. HYBRID's sparse encoding adds cost with no retrieval benefit when the filter is highly selective.

---

## [2026-03-12] Major Redesign: DOM-based parser

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

---

## [2026-03-12] EUR-Lex class names differ from documentation

The consolidated CRR HTML (CELEX:02013R0575-20260101) uses **different class names** than older EUR-Lex documents. The actual classes are:

| Class | Content | Count |
|-------|---------|-------|
| `title-division-1` | Numbered heading: `PART ONE`, `TITLE I`, `CHAPTER 1`, `SECTION 1`, `SUB-SECTION 1` | 189 |
| `title-article-norm` | Article number: `Article 92`, `Article 114` | 741 |
| `stitle-article-norm` | Article subtitle (not needed for parsing) | 741 |
| `norm` | Body paragraph text | 11,486 non-empty |

**Not** `sti-art`, `ti-art`, `ti-section-*`, `normal` as in older documents.

The `title-division-1` class covers all hierarchy levels (PART/TITLE/CHAPTER/SECTION/SUB-SECTION) — the level is determined by the first word of the text content, not by a separate class per level.

---

## [2026-03-12] Multilingual EUR-Lex headings

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

## [2026-03-12] AutoMergingRetriever requires ALL nodes in the docstore

`AutoMergingRetriever` climbs the node hierarchy from leaf → parent using the `storage_context.docstore`.
If only leaf nodes are in the vector index (or no docstore is attached at all), merging silently produces no parent context.

**Fix:** Populate `SimpleDocumentStore` with ALL nodes (from `HierarchicalNodeParser`), attach it to `StorageContext`, persist with `storage_context.persist(persist_dir=...)`, and reload it via `StorageContext.from_defaults(persist_dir=...)` on subsequent runs.

---

## [2026-03-12] `Settings.llm` ordering matters — indexer resets it to `None`

`HierarchicalIndexer._configure_settings()` sets `Settings.llm = None` (intentionally, to prevent accidental LLM calls during indexing).
If `QueryEngine` calls `self._configure_settings()` before `self.indexer.load()`, the indexer then overwrites the OpenAI LLM with `None`, and synthesis silently fails (returns empty strings).

**Fix:** Always call `_configure_settings()` **after** `indexer.load()` in `QueryEngine.load()`.

---

## [2026-03-12] EUR-Lex HTML uses CSS classes, not semantic heading tags

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

## [2026-03-12] lxml vs html.parser

`BeautifulSoup(html, "lxml")` raises `FeatureNotFound` if `lxml` is not installed.
Always catch `FeatureNotFound` and fall back to `"html.parser"` so parsing works without the optional C extension.
