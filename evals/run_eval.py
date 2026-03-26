"""
Eval runner: queries the CRR RAG API for every case in the golden dataset
and writes per-case results + a summary to evals/results/.

Usage:
    python -m evals.run_eval                                   # full run
    python -m evals.run_eval --limit 10                        # smoke test
    python -m evals.run_eval --case-ids case_001 case_002      # specific cases
    python -m evals.run_eval --dry-run                         # no HTTP calls
    python -m evals.run_eval --run-name after_reranker         # named run
    python -m evals.run_eval --workers 4                       # parallel
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import threading
import time
from pathlib import Path as _Path

# Load .env so OPENAI_API_KEY is available for the LLM-as-judge
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(_Path(__file__).parent.parent / ".env")
except ImportError:
    pass
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

from evals.metrics import compute_all, compute_all_with_expanded, normalise_article
from evals.judge import JUDGE_METRIC_KEYS, judge_answer

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

DEFAULT_DATASET = Path("evals/cases/golden_dataset.jsonl")
DEFAULT_OUTPUT = Path("evals/results")
DEFAULT_API = "http://localhost:8080"

RETRIEVAL_METRIC_KEYS = [
    "hit_at_1", "recall_at_1", "recall_at_3", "recall_at_5",
    "mrr", "precision_at_3", "precision_at_5",
]
EXPANDED_METRIC_KEYS = [
    "hit_at_1_with_expanded", "recall_at_1_with_expanded",
    "recall_at_3_with_expanded", "recall_at_5_with_expanded",
    "mrr_with_expanded",
]
METRIC_KEYS = RETRIEVAL_METRIC_KEYS + EXPANDED_METRIC_KEYS + JUDGE_METRIC_KEYS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _run_name_default() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _load_dataset(path: Path) -> list[dict]:
    """Load a JSONL file, skipping malformed lines and deduplicating by 'id'.

    When the same id appears more than once (e.g. an error row followed by a
    successful retry), the LAST occurrence wins so retried results take effect.
    """
    ordered_ids: list[str] = []
    rows_by_id: dict[str, dict] = {}
    with open(path, encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSON at %s line %d", path.name, line_num)
                continue
            case_id = row.get("id")
            if not case_id:
                logger.warning("Skipping row without 'id' at %s line %d", path.name, line_num)
                continue
            if case_id not in rows_by_id:
                ordered_ids.append(case_id)
            rows_by_id[case_id] = row  # last occurrence wins
    return [rows_by_id[cid] for cid in ordered_ids]


def _load_done_ids(cases_path: Path) -> set[str]:
    """Return IDs of successfully-completed cases. Error cases are NOT included
    so they will be retried automatically on resume."""
    if not cases_path.exists():
        return set()
    done: set[str] = set()
    with open(cases_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    row = json.loads(line)
                    if row.get("status") == "ok":
                        done.add(row["id"])
                except Exception:
                    pass
    return done


def _write_state(state_path: Path, data: dict) -> None:
    """Atomically write run state JSON via temp file + os.replace."""
    tmp = state_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(str(tmp), str(state_path))


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

def _start_api_server(api_url: str) -> "subprocess.Popen":
    """Spawn uvicorn in the background and return the process."""
    import subprocess as _sp
    from urllib.parse import urlparse
    parsed = urlparse(api_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8080
    cmd = [sys.executable, "-m", "uvicorn", "api.main:app", "--host", host, "--port", str(port)]
    logger.info("Auto-starting API server: %s", " ".join(cmd))
    # Use DEVNULL instead of PIPE — an unread PIPE buffer fills and blocks the
    # child process, which looks like the API freezing under load.
    return _sp.Popen(cmd, stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)


def preflight(api_url: str, timeout: int, auto_start: bool = False) -> None:
    """Abort if the API is not reachable or the index is not loaded.

    If auto_start=True and the API is not up, spawns uvicorn and waits up to
    120 s for the index to finish loading.
    """
    import subprocess as _sp  # noqa: F401 (may be needed below)
    health_url = f"{api_url.rstrip('/')}/health"
    api_proc: "subprocess.Popen | None" = None

    def _check() -> dict | None:
        try:
            r = requests.get(health_url, timeout=5)
            r.raise_for_status()
            return r.json()
        except Exception:
            return None

    data = _check()
    if data is None and auto_start:
        api_proc = _start_api_server(api_url)
        logger.info("Waiting for API to become healthy (up to 120 s)…")
        deadline = time.time() + 120
        while time.time() < deadline:
            time.sleep(3)
            data = _check()
            if data is not None:
                break
        if data is None:
            if api_proc:
                api_proc.terminate()
            logger.error("API did not become healthy within 120 s (%s).", health_url)
            sys.exit(1)

    if data is None:
        logger.error(
            "API health check failed (%s). "
            "Start the server with 'uvicorn api.main:app --port 8080' "
            "or pass --auto-start-api to let the runner start it.",
            health_url,
        )
        sys.exit(1)

    if not data.get("index_loaded"):
        logger.info("API is up but index is still loading — waiting up to 120 s…")
        deadline = time.time() + 120
        while time.time() < deadline:
            time.sleep(5)
            data = _check() or {}
            if data.get("index_loaded"):
                break
        if not data.get("index_loaded"):
            logger.error(
                "Index did not finish loading within 120 s. "
                "Check the API server logs."
            )
            sys.exit(1)

    logger.info(
        "API healthy — index loaded, %d vector store items.",
        data.get("vector_store_items", "?"),
    )


# ---------------------------------------------------------------------------
# Per-case evaluation
# ---------------------------------------------------------------------------

_RETRYABLE_STATUS_CODES = {500, 502, 503, 504}
_RETRY_DELAYS = [5, 15]  # seconds between attempts (2 retries → 3 total attempts)


def evaluate_case(
    case: dict,
    api_url: str,
    timeout: int,
    run_name: str,
    run_timestamp: str,
    use_judge: bool = False,
    judge_model: str = "gpt-4o",
) -> dict:
    """Query the API for one case, compute metrics, return result dict.

    Transient HTTP errors (500, 502, 503, 504) and timeouts are retried
    up to len(_RETRY_DELAYS) times with a short delay between attempts.
    """
    query_url = f"{api_url.rstrip('/')}/api/query"

    last_error_type: str = "exception"
    last_error_msg: str = "unknown"
    last_latency_ms: int = 0
    data: dict | None = None

    for attempt, delay in enumerate([0] + _RETRY_DELAYS):
        if delay:
            logger.info("  retry %d/%d for %s in %ds…", attempt, len(_RETRY_DELAYS), case["id"], delay)
            time.sleep(delay)

        t0 = time.perf_counter()
        try:
            resp = requests.post(
                query_url,
                json={"query": case["question"]},
                timeout=(10, timeout),
            )
            last_latency_ms = round((time.perf_counter() - t0) * 1000)

            if resp.status_code == 503:
                last_error_type = "api_503"
                last_error_msg = "Index not loaded (503)"
                # 503 = index not loaded; no point retrying
                return _error_result(case, last_error_type, last_error_msg, last_latency_ms, run_name, run_timestamp)

            if resp.status_code in _RETRYABLE_STATUS_CODES:
                last_error_type = f"http_{resp.status_code}"
                last_error_msg = f"{resp.status_code} Server Error for url: {query_url}"
                if attempt < len(_RETRY_DELAYS):
                    continue  # retry
                return _error_result(case, last_error_type, last_error_msg, last_latency_ms, run_name, run_timestamp, use_judge)

            resp.raise_for_status()
            data = resp.json()
            break  # success

        except requests.Timeout:
            last_latency_ms = round((time.perf_counter() - t0) * 1000)
            last_error_type = "timeout"
            last_error_msg = "Request timed out"
            if attempt < len(_RETRY_DELAYS):
                continue  # retry
            return _error_result(case, last_error_type, last_error_msg, last_latency_ms, run_name, run_timestamp, use_judge)
        except requests.HTTPError as exc:
            last_latency_ms = round((time.perf_counter() - t0) * 1000)
            last_error_type = f"http_{resp.status_code}"
            last_error_msg = str(exc)
            # Non-retryable HTTP errors (e.g. 400, 404, 422)
            return _error_result(case, last_error_type, last_error_msg, last_latency_ms, run_name, run_timestamp, use_judge)
        except Exception as exc:
            last_latency_ms = round((time.perf_counter() - t0) * 1000)
            last_error_type = "exception"
            last_error_msg = str(exc)
            if attempt < len(_RETRY_DELAYS):
                continue  # retry connection errors
            return _error_result(case, last_error_type, last_error_msg, last_latency_ms, run_name, run_timestamp, use_judge)

    if data is None:
        return _error_result(case, last_error_type, last_error_msg, last_latency_ms, run_name, run_timestamp, use_judge)

    latency_ms = last_latency_ms

    # Parse sources
    sources = data.get("sources", [])
    ranked_sources = [s for s in sources if not s.get("expanded", False)]
    expanded_sources = [s for s in sources if s.get("expanded", False)]

    # Extract article numbers from ranked sources (deduplicated, ordered)
    raw_retrieved = [
        s.get("metadata", {}).get("article", "")
        for s in ranked_sources
        if s.get("metadata", {}).get("article")
    ]
    retrieved_articles = []
    seen: set[str] = set()
    for a in raw_retrieved:
        norm = normalise_article(a)
        if norm not in seen:
            seen.add(norm)
            retrieved_articles.append(norm)

    expanded_articles = list({
        normalise_article(s.get("metadata", {}).get("article", ""))
        for s in expanded_sources
        if s.get("metadata", {}).get("article")
    })

    expected_articles = [normalise_article(a) for a in case.get("expected_articles", [])]

    metrics = compute_all(expected_articles, retrieved_articles)
    expanded_metrics = compute_all_with_expanded(expected_articles, retrieved_articles, expanded_articles)

    sources_raw = [
        {
            "article": normalise_article(s.get("metadata", {}).get("article", "")),
            "article_title": s.get("metadata", {}).get("article_title", ""),
            "score": s.get("score", 0.0),
            "expanded": s.get("expanded", False),
        }
        for s in sources
    ]

    rag_answer = data.get("answer", "")

    # LLM-as-judge (optional)
    judge_scores: dict = {k: None for k in JUDGE_METRIC_KEYS}
    judge_scores["judge_rationale"] = None
    if use_judge:
        reference_answer = case.get("reference_answer", "")
        judge_scores = judge_answer(
            case["question"], rag_answer, reference_answer, model=judge_model
        )

    return {
        "id": case["id"],
        "status": "ok",
        "question": case["question"],
        "expected_articles": expected_articles,
        "category": case.get("category", "unknown"),
        "difficulty": case.get("difficulty", "unknown"),
        "question_type": case.get("question_type", "unknown"),
        "citation_type": case.get("citation_type", "unknown"),
        "language": case.get("language", "en"),
        "is_multi_article": len(expected_articles) > 1,
        "retrieved_articles": retrieved_articles,
        "expanded_articles": expanded_articles,
        **metrics,
        **expanded_metrics,
        **judge_scores,
        "answer": rag_answer,
        "sources_raw": sources_raw,
        "latency_ms": latency_ms,
        "trace_id": data.get("trace_id", ""),
        "error_type": None,
        "error_message": None,
        "run_name": run_name,
        "run_timestamp": run_timestamp,
    }


def _error_result(
    case: dict, error_type: str, error_message: str, latency_ms: int,
    run_name: str, run_timestamp: str,
    use_judge: bool = False,
) -> dict:
    return {
        "id": case["id"],
        "status": "error",
        "question": case["question"],
        "expected_articles": [normalise_article(a) for a in case.get("expected_articles", [])],
        "category": case.get("category", "unknown"),
        "difficulty": case.get("difficulty", "unknown"),
        "question_type": case.get("question_type", "unknown"),
        "citation_type": case.get("citation_type", "unknown"),
        "language": case.get("language", "en"),
        "is_multi_article": len(case.get("expected_articles", [])) > 1,
        "retrieved_articles": [],
        "expanded_articles": [],
        **{k: None for k in RETRIEVAL_METRIC_KEYS},
        **{k: None for k in EXPANDED_METRIC_KEYS},
        **{k: None for k in JUDGE_METRIC_KEYS},
        "judge_rationale": None,
        "answer": "",
        "sources_raw": [],
        "latency_ms": latency_ms,
        "trace_id": "",
        "error_type": error_type,
        "error_message": error_message,
        "run_name": run_name,
        "run_timestamp": run_timestamp,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _mean(values: list[float]) -> Optional[float]:
    clean = [v for v in values if v is not None]
    return round(statistics.mean(clean), 4) if clean else None


def _aggregate(results: list[dict], filter_fn=None) -> dict:
    """Compute mean of all metrics over a (optionally filtered) result list."""
    rows = [r for r in results if r["status"] == "ok"]
    if filter_fn:
        rows = [r for r in rows if filter_fn(r)]
    n = len(rows)
    if n == 0:
        return {"n": 0, **{k: None for k in METRIC_KEYS}}
    return {
        "n": n,
        **{k: _mean([r.get(k) for r in rows]) for k in METRIC_KEYS},
    }


def build_summary(
    results: list[dict],
    run_name: str,
    run_timestamp: str,
    dataset_path: str,
    api_url: str,
    judge_enabled: bool = False,
) -> dict:
    """Build the per-run summary dict."""
    ok = [r for r in results if r["status"] == "ok"]
    failed = [r for r in results if r["status"] != "ok"]
    latencies = [r["latency_ms"] for r in ok if r["latency_ms"] is not None]

    # Category breakdown
    categories = sorted({r["category"] for r in results})
    by_category = {
        cat: _aggregate(results, lambda r, c=cat: r["category"] == c)
        for cat in categories
    }

    # Difficulty breakdown
    by_difficulty = {
        d: _aggregate(results, lambda r, d=d: r["difficulty"] == d)
        for d in ("easy", "medium", "hard")
    }

    # Question type breakdown
    qtypes = sorted({r["question_type"] for r in results})
    by_question_type = {
        qt: _aggregate(results, lambda r, qt=qt: r["question_type"] == qt)
        for qt in qtypes
    }

    # Single vs multi-article
    by_article_count = {
        "single": _aggregate(results, lambda r: not r["is_multi_article"]),
        "multi": _aggregate(results, lambda r: r["is_multi_article"]),
    }

    # Article-cited vs open-ended question framing
    by_citation_type = {
        "article_cited": _aggregate(results, lambda r: r.get("citation_type") == "article_cited"),
        "open_ended": _aggregate(results, lambda r: r.get("citation_type") == "open_ended"),
    }

    return {
        "run_name": run_name,
        "run_timestamp": run_timestamp,
        "dataset_path": str(dataset_path),
        "api_url": api_url,
        "judge_enabled": judge_enabled,
        "total_cases": len(results),
        "successful_cases": len(ok),
        "failed_cases": len(failed),
        "overall": _aggregate(results),
        "by_category": by_category,
        "by_difficulty": by_difficulty,
        "by_question_type": by_question_type,
        "by_article_count": by_article_count,
        "by_citation_type": by_citation_type,
        "p50_latency_ms": round(statistics.median(latencies)) if latencies else None,
        "p90_latency_ms": round(statistics.quantiles(latencies, n=10)[8]) if len(latencies) >= 10 else None,
        "p99_latency_ms": round(statistics.quantiles(latencies, n=100)[98]) if len(latencies) >= 100 else None,
        "mean_latency_ms": round(statistics.mean(latencies)) if latencies else None,
    }


# ---------------------------------------------------------------------------
# Config capture
# ---------------------------------------------------------------------------

def _git_commit() -> str:
    """Return the current HEAD short SHA, or 'unknown' if git is unavailable."""
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _capture_run_config(args: "argparse.Namespace", run_name: str, run_timestamp: str) -> dict:
    """Snapshot all tunable settings into a serialisable dict.

    Reads env vars with their defaults so the config is self-contained —
    no need to inspect .env to understand what a past run used.
    """
    def _bool(key: str, default: str) -> bool:
        return os.getenv(key, default).lower() == "true"

    def _float(key: str, default: float) -> float:
        try:
            return float(os.getenv(key, str(default)))
        except (TypeError, ValueError):
            return default

    def _int(key: str, default: int) -> int:
        try:
            return int(os.getenv(key, str(default)))
        except (TypeError, ValueError):
            return default

    return {
        "run_name": run_name,
        "run_timestamp": run_timestamp,
        "description": args.description,
        "git_commit": _git_commit(),
        "eval": {
            "dataset_path": str(args.dataset),
            "api_url": args.api_url,
            "workers": args.workers,
            "limit": args.limit,
            "case_ids": args.case_ids,
            "judge_enabled": args.judge,
            "judge_model": args.judge_model if args.judge else None,
        },
        "retrieval": {
            "top_k": _int("RETRIEVAL_TOP_K", 12),
            "alpha": _float("RETRIEVAL_ALPHA", 0.5),
            "similarity_cutoff": 0.3,
            "use_reranker": _bool("USE_RERANKER", "false"),
            "reranker_model": os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            "rerank_top_n": _int("RERANK_TOP_N", 6),
            "rerank_blend_alpha": _float("RERANK_BLEND_ALPHA", 0.3),
            "title_boost_weight": _float("TITLE_BOOST_WEIGHT", 0.15),
            "adjacent_tiebreak_delta": _float("ADJACENT_TIEBREAK_DELTA", 0.0),
        },
        "synthesis": {
            "llm_model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            "hard_query_model": os.getenv("HARD_QUERY_MODEL", "gpt-4o"),
            "max_context_chars": _int("MAX_CONTEXT_CHARS", 100_000),
        },
        "query_pipeline": {
            "use_hyde": _bool("USE_HYDE", "false"),
            "use_toc_routing": _bool("USE_TOC_ROUTING", "false"),
            "use_query_enrichment": _bool("USE_QUERY_ENRICHMENT", "true"),
            "use_article_graph": _bool("USE_ARTICLE_GRAPH", "false"),
            "use_mixed_chunking": _bool("USE_MIXED_CHUNKING", "false"),
            "use_paragraph_chunking": _bool("USE_PARAGRAPH_CHUNKING", "false"),
            "use_paragraph_window_reranker": _bool("USE_PARAGRAPH_WINDOW_RERANKER", "false"),
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run CRR RAG eval against the golden dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                   help="Directory to write results into")
    p.add_argument("--api-url", default=DEFAULT_API)
    p.add_argument("--timeout", type=int, default=150,
                   help="Per-request HTTP timeout in seconds (default: 150, must match QUERY_TIMEOUT_SECONDS)")
    p.add_argument("--workers", type=int, default=1,
                   help="Parallel workers (1 = sequential)")
    p.add_argument("--limit", type=int, default=None,
                   help="Evaluate only first N cases")
    p.add_argument("--case-ids", nargs="+", default=None,
                   help="Evaluate only specific case IDs")
    p.add_argument("--run-name", default=None,
                   help="Label for this run (default: timestamp)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print plan without making API calls")
    p.add_argument("--judge", action="store_true",
                   help="Score answers with an LLM judge (gpt-4o) after retrieval eval")
    p.add_argument("--judge-model", default="gpt-4o",
                   help="OpenAI model to use as judge (default: gpt-4o)")
    p.add_argument("--no-resume", action="store_true",
                   help="Ignore any previous partial results and start the run from scratch")
    p.add_argument("--auto-start-api", action="store_true",
                   help="Auto-start the API server (uvicorn) if it is not already running")
    p.add_argument("--log-file", default=None,
                   help="Write log output to this file in addition to stderr")
    p.add_argument("--description", default=None,
                   help="Free-text note describing what this run is testing (saved to config)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.log_file:
        _fh = logging.FileHandler(args.log_file, encoding="utf-8")
        _fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s"))
        logging.getLogger().addHandler(_fh)

    run_name = args.run_name or _run_name_default()
    run_timestamp = _now_iso()
    args.output.mkdir(parents=True, exist_ok=True)

    cases_path = args.output / f"{run_name}_cases.jsonl"
    summary_path = args.output / f"{run_name}_summary.json"
    config_path = args.output / f"{run_name}_config.json"

    # Capture and persist run config immediately so it's available even for dry-runs.
    run_config = _capture_run_config(args, run_name, run_timestamp)
    _write_state(config_path, run_config)

    logger.info("=" * 60)
    logger.info("CRR Eval Runner")
    logger.info("  run     : %s", run_name)
    logger.info("  dataset : %s", args.dataset)
    logger.info("  api     : %s", args.api_url)
    logger.info("  output  : %s", cases_path)
    logger.info("  workers : %d", args.workers)
    logger.info("  dry-run : %s", args.dry_run)
    logger.info("  judge   : %s%s", args.judge,
                f" (model={args.judge_model})" if args.judge else "")
    logger.info("  git     : %s", run_config["git_commit"])
    if args.description:
        logger.info("  note    : %s", args.description)
    logger.info("=" * 60)

    # --no-resume: delete any existing partial output for this run name
    if args.no_resume and cases_path.exists():
        cases_path.unlink()
        logger.info("--no-resume: deleted previous results at %s.", cases_path)

    # Load dataset
    all_cases = _load_dataset(args.dataset)
    logger.info("Loaded %d cases from %s.", len(all_cases), args.dataset)

    # Apply filters
    if args.case_ids:
        id_set = set(args.case_ids)
        all_cases = [c for c in all_cases if c["id"] in id_set]
        logger.info("Filtered to %d cases by --case-ids.", len(all_cases))
    if args.limit:
        all_cases = all_cases[: args.limit]
        logger.info("Limited to first %d cases.", len(all_cases))

    # Resumability
    done_ids = _load_done_ids(cases_path)
    if done_ids:
        logger.info("Resuming: %d cases already evaluated, skipping.", len(done_ids))
    todo = [c for c in all_cases if c["id"] not in done_ids]
    logger.info("%d cases to evaluate.", len(todo))

    if args.dry_run:
        for c in todo:
            logger.info("  [dry-run] Would evaluate %s — %s", c["id"], c["question"][:80])
        logger.info("Dry run complete.")
        return

    # Preflight
    preflight(args.api_url, args.timeout, auto_start=args.auto_start_api)

    # Persist run state so the dashboard can track progress cross-session.
    state_path = args.output / f"{run_name}_state.json"
    _write_state(state_path, {
        "run_name": run_name,
        "status": "running",
        "pid": os.getpid(),
        "planned_total": len(all_cases),
        "started_at": run_timestamp,
    })

    # Evaluate
    results_this_run: list[dict] = []

    # Single-writer discipline: one lock + dedup set prevents duplicate rows
    # from concurrent threads or a late-completing worker after timeout.
    _write_lock = threading.Lock()
    _written_ids: set[str] = set(done_ids)  # seed with already-resumed IDs

    def _write_result(result: dict) -> None:
        """Thread-safe, dedup-guarded append of one result to the cases file."""
        with _write_lock:
            if result["id"] not in _written_ids:
                with open(cases_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(result, ensure_ascii=False) + "\n")
                    fh.flush()
                _written_ids.add(result["id"])

    def _eval_and_write(case: dict) -> dict:
        result = evaluate_case(
            case, args.api_url, args.timeout, run_name, run_timestamp,
            use_judge=args.judge, judge_model=args.judge_model,
        )
        _write_result(result)
        return result

    try:
        from tqdm import tqdm
        progress = tqdm(total=len(todo), unit="case")
    except ImportError:
        progress = None

    try:
        if args.workers == 1:
            for case in todo:
                result = _eval_and_write(case)
                results_this_run.append(result)
                status = "✓" if result["status"] == "ok" else "✗"
                msg = f"{status} {result['id']} hit@1={result.get('hit_at_1')} mrr={result.get('mrr')} ({result['latency_ms']}ms)"
                if progress:
                    progress.set_description(result["id"])
                    progress.update(1)
                else:
                    logger.info(msg)
        else:
            import concurrent.futures as _cf
            # Per-future deadline: HTTP timeout + 30 s grace for overhead.
            _fut_timeout = args.timeout + 30
            # Use explicit pool (not context manager) so we can call shutdown(wait=False)
            # if stuck threads would otherwise block __exit__ forever.
            pool = ThreadPoolExecutor(max_workers=args.workers)
            try:
                futures = {pool.submit(_eval_and_write, c): c for c in todo}
                pending = set(futures.keys())
                while pending:
                    done_set, pending = _cf.wait(
                        pending, timeout=_fut_timeout,
                        return_when=_cf.FIRST_COMPLETED,
                    )
                    if not done_set:
                        # No future completed within _fut_timeout — remaining are stuck.
                        # Record a timeout error for each; _write_result's dedup guard
                        # ensures a late-completing worker cannot add a second row.
                        logger.warning(
                            "%d futures did not complete within %ds — recording as timeout errors.",
                            len(pending), _fut_timeout,
                        )
                        for stuck_fut in pending:
                            case = futures[stuck_fut]
                            result = _error_result(
                                case, "timeout", f"Future timed out after {_fut_timeout}s",
                                _fut_timeout * 1000, run_name, run_timestamp,
                            )
                            _write_result(result)
                            results_this_run.append(result)
                            if progress:
                                progress.set_description(case["id"])
                                progress.update(1)
                        break
                    for fut in done_set:
                        try:
                            result = fut.result()
                        except Exception as exc:
                            case = futures[fut]
                            logger.error("Unexpected error for %s: %s", case["id"], exc)
                            result = _error_result(
                                case, "exception", str(exc), 0, run_name, run_timestamp,
                            )
                            _write_result(result)
                        results_this_run.append(result)
                        if progress:
                            progress.set_description(result["id"])
                            progress.update(1)
            finally:
                pool.shutdown(wait=False)
    except Exception:
        _write_state(state_path, {
            "run_name": run_name,
            "status": "failed",
            "pid": os.getpid(),
            "planned_total": len(all_cases),
            "started_at": run_timestamp,
            "failed_at": _now_iso(),
        })
        raise

    if progress:
        progress.close()

    # Load all results (including previously-done ones) for summary.
    # _load_dataset deduplicates by id, so duplicates from any prior crash are excluded.
    all_results = _load_dataset(cases_path)  # type: ignore[arg-type]

    # Build and write summary (atomic write via temp file).
    summary = build_summary(
        all_results,
        run_name=run_name,
        run_timestamp=run_timestamp,
        dataset_path=args.dataset,
        api_url=args.api_url,
        judge_enabled=args.judge,
    )
    _tmp_summary = summary_path.with_suffix(".tmp")
    _tmp_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(str(_tmp_summary), str(summary_path))

    # Mark run as completed.
    _write_state(state_path, {
        "run_name": run_name,
        "status": "completed",
        "pid": os.getpid(),
        "planned_total": len(all_cases),
        "started_at": run_timestamp,
        "completed_at": _now_iso(),
    })

    # Print scorecard
    ov = summary["overall"]
    logger.info("")
    logger.info("=" * 60)
    logger.info("Run complete: %s", run_name)
    logger.info("  Cases: %d ok / %d failed / %d total",
                summary["successful_cases"], summary["failed_cases"], summary["total_cases"])
    if summary["failed_cases"]:
        pct = summary["failed_cases"] / summary["total_cases"] * 100 if summary["total_cases"] else 0
        logger.warning(
            "  *** %d CASES FAILED (%.1f%%) — metrics reflect only the %d successful cases ***",
            summary["failed_cases"], pct, summary["successful_cases"],
        )
        # Log error breakdown by type
        error_types: dict[str, int] = {}
        for r in all_results:
            if r.get("status") != "ok":
                et = r.get("error_type") or "unknown"
                error_types[et] = error_types.get(et, 0) + 1
        for et, cnt in sorted(error_types.items(), key=lambda x: -x[1]):
            logger.warning("    %s: %d case(s)", et, cnt)
    logger.info("  Hit@1:      %.1f%%", (ov.get("hit_at_1") or 0) * 100)
    logger.info("  Recall@3:   %.1f%%", (ov.get("recall_at_3") or 0) * 100)
    logger.info("  Recall@5:   %.1f%%", (ov.get("recall_at_5") or 0) * 100)
    logger.info("  MRR:        %.3f", ov.get("mrr") or 0)

    ct = summary.get("by_citation_type", {})
    ac = ct.get("article_cited", {})
    oe = ct.get("open_ended", {})
    logger.info("")
    logger.info("  --- By citation type ---")
    logger.info("  article_cited  (n=%s)  Hit@1=%.1f%%  Recall@3=%.1f%%  MRR=%.3f",
                ac.get("n", 0),
                (ac.get("hit_at_1") or 0) * 100,
                (ac.get("recall_at_3") or 0) * 100,
                ac.get("mrr") or 0)
    logger.info("  open_ended     (n=%s)  Hit@1=%.1f%%  Recall@3=%.1f%%  MRR=%.3f",
                oe.get("n", 0),
                (oe.get("hit_at_1") or 0) * 100,
                (oe.get("recall_at_3") or 0) * 100,
                oe.get("mrr") or 0)

    if args.judge:
        logger.info("")
        logger.info("  --- LLM Judge scores ---")
        logger.info("  Correctness:   %.3f", ov.get("judge_correctness") or 0)
        logger.info("  Completeness:  %.3f", ov.get("judge_completeness") or 0)
        logger.info("  Faithfulness:  %.3f", ov.get("judge_faithfulness") or 0)

    logger.info("")
    logger.info("  P50 latency: %s ms", summary.get("p50_latency_ms"))
    logger.info("  Summary → %s", summary_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main(sys.argv[1:])
