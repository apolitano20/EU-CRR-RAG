"""
Compare two CRR RAG eval runs: print a delta scorecard to stdout and
optionally write a machine-readable comparison JSON.

The `build_comparison()` function is also imported by evals/dashboard.py
to power the dedicated Compare Runs page.

Usage:
    python -m evals.compare run_A run_B
    python -m evals.compare run_A run_B --output evals/results/cmp_A_vs_B.json
    python -m evals.compare --list
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

RESULTS_DIR = Path("evals/results")

RETRIEVAL_METRIC_KEYS = [
    "hit_at_1", "recall_at_1", "recall_at_3", "recall_at_5",
    "mrr", "precision_at_3", "precision_at_5",
]
JUDGE_METRIC_KEYS = ["judge_correctness", "judge_completeness", "judge_faithfulness"]
METRIC_KEYS = RETRIEVAL_METRIC_KEYS + JUDGE_METRIC_KEYS
METRIC_LABELS = {
    "hit_at_1":              "Hit@1",
    "recall_at_1":           "Recall@1",
    "recall_at_3":           "Recall@3",
    "recall_at_5":           "Recall@5",
    "mrr":                   "MRR",
    "precision_at_3":        "Prec@3",
    "precision_at_5":        "Prec@5",
    "judge_correctness":     "Judge Correctness",
    "judge_completeness":    "Judge Completeness",
    "judge_faithfulness":    "Judge Faithfulness",
}

# A delta below this (in fractional units, not pp) is flagged as a regression.
REGRESSION_THRESHOLD = -0.01   # -1 pp
IMPROVEMENT_THRESHOLD =  0.01  # +1 pp


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_summary(name_or_path: str) -> Path:
    p = Path(name_or_path)
    if p.exists():
        return p
    candidate = RESULTS_DIR / f"{name_or_path}_summary.json"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"Cannot find run '{name_or_path}'. "
        f"Expected '{candidate}' or a direct path to a summary JSON."
    )


def _load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _compare_slice(slice_a: dict, slice_b: dict) -> dict:
    """Compute per-metric {a, b, delta} for one breakdown slice."""
    metrics: dict[str, dict] = {}
    for m in METRIC_KEYS:
        va = slice_a.get(m)
        vb = slice_b.get(m)
        delta = round(vb - va, 4) if (va is not None and vb is not None) else None
        metrics[m] = {"a": va, "b": vb, "delta": delta}
    return {
        "n_a": slice_a.get("n", 0),
        "n_b": slice_b.get("n", 0),
        **metrics,
    }


def _compare_breakdown(summary_a: dict, summary_b: dict, key: str) -> dict:
    da = summary_a.get(key, {})
    db = summary_b.get(key, {})
    return {
        slice_key: _compare_slice(da.get(slice_key, {}), db.get(slice_key, {}))
        for slice_key in sorted(set(da) | set(db))
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_comparison(summary_a: dict, summary_b: dict) -> dict:
    """
    Build a full comparison dict between two run summaries.

    Returns a dict with keys:
        run_a, run_b, generated_at,
        overall, by_category, by_difficulty, by_question_type, by_article_count,
        regressions, improvements, latency
    """
    overall = _compare_slice(
        summary_a.get("overall", {}),
        summary_b.get("overall", {}),
    )

    by_category      = _compare_breakdown(summary_a, summary_b, "by_category")
    by_difficulty    = _compare_breakdown(summary_a, summary_b, "by_difficulty")
    by_question_type = _compare_breakdown(summary_a, summary_b, "by_question_type")
    by_article_count = _compare_breakdown(summary_a, summary_b, "by_article_count")

    regressions = [
        {"metric": m, **overall[m]}
        for m in METRIC_KEYS
        if overall[m]["delta"] is not None and overall[m]["delta"] < REGRESSION_THRESHOLD
    ]
    improvements = [
        {"metric": m, **overall[m]}
        for m in METRIC_KEYS
        if overall[m]["delta"] is not None and overall[m]["delta"] > IMPROVEMENT_THRESHOLD
    ]

    return {
        "run_a": summary_a.get("run_name", "?"),
        "run_b": summary_b.get("run_name", "?"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall": overall,
        "by_category": by_category,
        "by_difficulty": by_difficulty,
        "by_question_type": by_question_type,
        "by_article_count": by_article_count,
        "regressions": regressions,
        "improvements": improvements,
        "latency": {
            "p50_a":  summary_a.get("p50_latency_ms"),
            "p50_b":  summary_b.get("p50_latency_ms"),
            "p90_a":  summary_a.get("p90_latency_ms"),
            "p90_b":  summary_b.get("p90_latency_ms"),
            "mean_a": summary_a.get("mean_latency_ms"),
            "mean_b": summary_b.get("mean_latency_ms"),
        },
    }


# ---------------------------------------------------------------------------
# CLI output
# ---------------------------------------------------------------------------

def _fmt(val: float | None, as_pct: bool = True) -> str:
    if val is None:
        return "—"
    return f"{val * 100:.1f}%" if as_pct else f"{val:.4f}"


def _fmt_delta(delta: float | None, as_pct: bool = True) -> str:
    if delta is None:
        return "—"
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta * 100:.1f}pp" if as_pct else f"{sign}{delta:.4f}"


def print_report(cmp: dict) -> None:
    a, b = cmp["run_a"], cmp["run_b"]
    W = 70
    print()
    print("=" * W)
    print(f"  Eval comparison:  A = {a}")
    print(f"                    B = {b}")
    print("=" * W)

    # Overall scorecard
    print(f"\n  {'Metric':<22} {'A':>10} {'B':>10} {'Delta':>10}  ")
    print("  " + "-" * 58)
    ov = cmp["overall"]
    for m in METRIC_KEYS:
        if m not in ov:
            continue
        va    = ov[m]["a"]
        vb    = ov[m]["b"]
        delta = ov[m]["delta"]
        # Skip judge metrics entirely if both runs have no judge scores
        if m in JUDGE_METRIC_KEYS and va is None and vb is None:
            continue
        as_pct = m not in ("mrr",) and m not in JUDGE_METRIC_KEYS
        arrow = " ↑" if (delta or 0) > IMPROVEMENT_THRESHOLD else (
                " ↓" if (delta or 0) < REGRESSION_THRESHOLD else "  ")
        print(
            f"  {METRIC_LABELS[m]:<22} "
            f"{_fmt(va, as_pct):>10} "
            f"{_fmt(vb, as_pct):>10} "
            f"{_fmt_delta(delta, as_pct):>10}{arrow}"
        )

    # Latency
    lat = cmp.get("latency", {})
    print(f"\n  {'Latency':<16} {'A':>10} {'B':>10}")
    print("  " + "-" * 38)
    for lbl, ka, kb in [
        ("P50 (ms)",  "p50_a",  "p50_b"),
        ("P90 (ms)",  "p90_a",  "p90_b"),
        ("Mean (ms)", "mean_a", "mean_b"),
    ]:
        va_l = lat.get(ka, "—")
        vb_l = lat.get(kb, "—")
        print(f"  {lbl:<16} {str(va_l):>10} {str(vb_l):>10}")

    # Regression / improvement summary
    print()
    if cmp["regressions"]:
        items = ", ".join(
            f"{METRIC_LABELS[r['metric']]} {_fmt_delta(r['delta'])}"
            for r in cmp["regressions"]
        )
        print(f"  ⚠️  Regressions : {items}")
    if cmp["improvements"]:
        items = ", ".join(
            f"{METRIC_LABELS[r['metric']]} {_fmt_delta(r['delta'])}"
            for r in cmp["improvements"]
        )
        print(f"  ✅  Improvements: {items}")
    if not cmp["regressions"] and not cmp["improvements"]:
        print("  No significant changes (±1 pp threshold across all metrics).")
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _list_runs() -> list[str]:
    if not RESULTS_DIR.exists():
        return []
    return sorted(
        p.stem.replace("_summary", "")
        for p in RESULTS_DIR.glob("*_summary.json")
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare two CRR RAG eval runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m evals.compare --list\n"
            "  python -m evals.compare baseline_run new_run\n"
            "  python -m evals.compare baseline_run new_run -o evals/results/cmp.json\n"
        ),
    )
    p.add_argument("run_a", nargs="?", help="Baseline run name or path to _summary.json")
    p.add_argument("run_b", nargs="?", help="Candidate run name or path to _summary.json")
    p.add_argument("--output", "-o", type=Path, default=None,
                   help="Write comparison JSON to this path")
    p.add_argument("--list", "-l", action="store_true",
                   help="List available runs and exit")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.list:
        runs = _list_runs()
        if not runs:
            print("No completed runs found in evals/results/")
        else:
            print("Available runs:")
            for r in runs:
                print(f"  {r}")
        return

    if not args.run_a or not args.run_b:
        print("Error: provide two run names (or --list to see available runs).", file=sys.stderr)
        sys.exit(1)

    if args.run_a == args.run_b:
        print("Error: run_a and run_b must be different.", file=sys.stderr)
        sys.exit(1)

    try:
        path_a = _find_summary(args.run_a)
        path_b = _find_summary(args.run_b)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    summary_a = _load_summary(path_a)
    summary_b = _load_summary(path_b)
    cmp = build_comparison(summary_a, summary_b)

    print_report(cmp)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(cmp, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Comparison JSON written → {args.output}")


if __name__ == "__main__":
    main(sys.argv[1:])
