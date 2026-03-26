"""
CRR RAG Eval Dashboard — Streamlit app.

Two modes:
  1. Dataset Review  — browse and annotate golden dataset cases (no eval needed)
  2. Eval Results    — scorecard, breakdowns, drill-down (requires a completed run)

Usage:
    streamlit run evals/dashboard.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RESULTS_DIR       = Path("evals/results")
DATASET_PATH      = Path("evals/cases/golden_dataset.jsonl")
TEST_DATASET_PATH = Path("evals/cases/test_dataset.jsonl")
REVIEW_PATH       = Path("evals/cases/review_status.json")

KNOWN_DATASETS = {
    "Golden dataset (173 cases)": DATASET_PATH,
    "Test set — held-out (55 cases)": TEST_DATASET_PATH,
}

METRIC_KEYS = [
    "hit_at_1", "recall_at_1", "recall_at_3", "recall_at_5",
    "mrr", "precision_at_3", "precision_at_5",
]
METRIC_LABELS = {
    "hit_at_1":      "Hit@1",
    "recall_at_1":   "Recall@1",
    "recall_at_3":   "Recall@3",
    "recall_at_5":   "Recall@5",
    "mrr":           "MRR",
    "precision_at_3":"Prec@3",
    "precision_at_5":"Prec@5",
}

JUDGE_METRIC_KEYS = ["judge_correctness", "judge_completeness", "judge_faithfulness"]
JUDGE_METRIC_LABELS = {
    "judge_correctness":   "Correctness",
    "judge_completeness":  "Completeness",
    "judge_faithfulness":  "Faithfulness",
}

REVIEW_OPTIONS  = ["— unreviewed —", "✅ approved", "🔵 nuance_loss", "⚠️ needs_fix", "❌ remove"]
REVIEW_STATUSES = {"— unreviewed —": None, "✅ approved": "approved",
                   "🔵 nuance_loss": "nuance_loss",
                   "⚠️ needs_fix": "needs_fix", "❌ remove": "remove"}
STATUS_REVERSE  = {v: k for k, v in REVIEW_STATUSES.items()}
STATUS_COLOURS  = {"approved": "#d4edda", "nuance_loss": "#cce5ff", "needs_fix": "#fff3cd", "remove": "#f8d7da"}

# ---------------------------------------------------------------------------
# Data normalization
# ---------------------------------------------------------------------------

def _normalize_case(c: dict) -> dict:
    """Ensure all required fields exist with safe defaults."""
    c.setdefault("id", "")
    c.setdefault("category", "unknown")
    c.setdefault("difficulty", "medium")
    c.setdefault("question_type", "unknown")
    c.setdefault("language", "en")
    c.setdefault("question", "")
    c.setdefault("reference_answer", "")
    c.setdefault("expected_articles", [])
    c.setdefault("notes", "")
    return c

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _fetch_article_text(art_num: str) -> dict:
    """Fetch text for a single article from Qdrant. Returns {title, text} or {error}."""
    import sys
    import json as _json
    import traceback
    from pathlib import Path as _Path

    project_root = str(_Path(__file__).parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    try:
        from dotenv import load_dotenv
        load_dotenv(_Path(__file__).parent.parent / ".env")
        from src.indexing.vector_store import VectorStore
        vs = VectorStore()
        vs.connect_readonly()
        payloads = vs.scroll_payloads(language="en")
        title, texts = "", []
        for p in payloads:
            if str(p.get("article", "")).strip() != art_num:
                continue
            text = p.get("text", "") or ""
            if not text:
                raw = p.get("_node_content", "")
                if raw:
                    try:
                        text = _json.loads(raw).get("text", "") or ""
                    except Exception:
                        pass
            if text:
                texts.append(text)
            if not title:
                title = p.get("article_title", "") or ""
        return {"title": title, "text": "\n\n".join(texts)}
    except Exception:
        return {"title": "", "text": "", "error": traceback.format_exc()}


@st.cache_data(show_spinner="Loading articles from Qdrant…")
def _load_article_index() -> dict[str, dict]:
    """Return {article_num: {title, text}} pulled from Qdrant. Raises on failure."""
    import sys
    import json as _json
    from pathlib import Path as _Path

    # Ensure project root is on sys.path so `src` is importable regardless of CWD
    project_root = str(_Path(__file__).parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from dotenv import load_dotenv
    env_path = _Path(__file__).parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)

    from src.indexing.vector_store import VectorStore

    vs = VectorStore()
    vs.connect_readonly()
    payloads = vs.scroll_payloads(language="en")

    index: dict[str, dict] = {}
    for p in payloads:
        art = str(p.get("article", "")).strip()
        if not art:
            continue
        text = p.get("text", "") or ""
        if not text:
            raw = p.get("_node_content", "")
            if raw:
                try:
                    text = _json.loads(raw).get("text", "") or ""
                except Exception:
                    pass
        title = p.get("article_title", "")
        if art not in index:
            index[art] = {"title": title, "text": text}
        else:
            index[art]["text"] = index[art]["text"] + "\n\n" + text
            if not index[art]["title"] and title:
                index[art]["title"] = title
    return index


@st.cache_data(show_spinner=False)
def _load_golden_dataset() -> list[dict]:
    if not DATASET_PATH.exists():
        return []
    cases = []
    with open(DATASET_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    cases.append(_normalize_case(json.loads(line)))
                except Exception:
                    pass
    return cases


@st.cache_data(show_spinner=False)
def _load_cases(path: str) -> list[dict]:
    """Load any JSONL dataset by path string. Returns normalised case list."""
    result = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        result.append(_normalize_case(json.loads(line)))
                    except Exception:
                        pass
    except Exception:
        pass
    return result


def _load_review_status() -> dict[str, dict]:
    """Load review annotations from disk. Returns {case_id: {status, note}}."""
    if not REVIEW_PATH.exists():
        return {}
    try:
        return json.loads(REVIEW_PATH.read_text(encoding="utf-8"))
    except Exception:
        st.warning(f"Could not load review status from `{REVIEW_PATH}`. Starting fresh.")
        return {}


def _save_review_status(status: dict[str, dict]) -> None:
    REVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write: write to temp then rename
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=REVIEW_PATH.parent, prefix=".review_tmp_", suffix=".json"
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            json.dump(status, fh, indent=2, ensure_ascii=False)
        os.replace(tmp_path, REVIEW_PATH)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


@st.cache_data(show_spinner=False)
def _load_run(cases_path: str, file_mtime: float = 0.0) -> pd.DataFrame:
    rows = []
    try:
        with open(cases_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        pass
    except Exception as exc:
        st.error(f"Could not load run file `{cases_path}`: {exc}")
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def _load_summary(summary_path: str, file_mtime: float = 0.0) -> dict:
    try:
        return json.loads(Path(summary_path).read_text(encoding="utf-8"))
    except Exception as exc:
        st.error(f"Could not load summary file `{summary_path}`: {exc}")
        return {}


@st.cache_data(show_spinner=False)
def _load_config(config_path: str, file_mtime: float = 0.0) -> dict:
    try:
        return json.loads(Path(config_path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def _discover_runs() -> list[tuple[str, Path, Path]]:
    if not RESULTS_DIR.exists():
        return []
    runs = []
    for sp in sorted(RESULTS_DIR.glob("*_summary.json"), reverse=True):
        run_name = sp.stem.replace("_summary", "")
        cp = RESULTS_DIR / f"{run_name}_cases.jsonl"
        if cp.exists():
            runs.append((run_name, sp, cp))
    return runs


def _discover_incomplete_runs() -> list[dict]:
    """Return state dicts for runs that have a state.json but no summary (partial/failed/running)."""
    if not RESULTS_DIR.exists():
        return []
    completed_names = {sp.stem.replace("_summary", "") for sp in RESULTS_DIR.glob("*_summary.json")}
    incomplete = []
    for sf in sorted(RESULTS_DIR.glob("*_state.json"), reverse=True):
        run_name = sf.stem.replace("_state", "")
        if run_name in completed_names:
            continue
        try:
            state = json.loads(sf.read_text(encoding="utf-8"))
            incomplete.append(state)
        except Exception:
            pass
    return incomplete


# ---------------------------------------------------------------------------
# ── PAGE 1: Dataset Review ──────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def page_dataset_review() -> None:
    import re as _re

    st.header("Dataset Review")
    st.caption(
        "Browse and annotate the 173 golden-dataset cases before wiring them into "
        "the eval pipeline. Mark each case as **approved**, **nuance_loss**, **needs_fix**, or **remove**."
    )

    cases = _load_golden_dataset()
    if not cases:
        st.error(f"Dataset not found at `{DATASET_PATH}`. Run the generator first.")
        return

    # ── Session-state initialisation ─────────────────────────────────────
    if "review_status" not in st.session_state:
        st.session_state.review_status = _load_review_status()
    if "selected_case_id" not in st.session_state:
        st.session_state.selected_case_id = cases[0]["id"] if cases else ""
    # Cache for on-demand article fetches: {art_num: {title, text} | {error}}
    if "fetched_articles" not in st.session_state:
        st.session_state.fetched_articles = {}

    # If a non-table action (Next/Prev/selectbox/save) triggered the last rerun,
    # clear the table's stale row selection so it doesn't override navigation.
    if st.session_state.pop("rv_table_reset", False):
        st.session_state.pop("rv_table", None)

    review = st.session_state.review_status

    # ── Sidebar filters ──────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("---")
        st.subheader("Review Filters")
        cats    = sorted({c["category"] for c in cases})
        diffs   = sorted({c["difficulty"] for c in cases})
        qtypes  = sorted({c["question_type"] for c in cases})

        cat_f    = st.multiselect("Category",      cats,   default=cats,   key="rv_cat")
        diff_f   = st.multiselect("Difficulty",    diffs,  default=diffs,  key="rv_diff")
        qtype_f  = st.multiselect("Question Type", qtypes, default=qtypes, key="rv_qtype")
        status_f = st.multiselect(
            "Review Status",
            ["unreviewed", "approved", "nuance_loss", "needs_fix", "remove"],
            default=["unreviewed", "approved", "nuance_loss", "needs_fix", "remove"],
            key="rv_status",
        )

    # ── Progress summary ─────────────────────────────────────────────────
    total     = len(cases)
    reviewed     = sum(1 for c in cases if review.get(c["id"], {}).get("status"))
    approved     = sum(1 for c in cases if review.get(c["id"], {}).get("status") == "approved")
    nuance_loss  = sum(1 for c in cases if review.get(c["id"], {}).get("status") == "nuance_loss")
    needs_fix    = sum(1 for c in cases if review.get(c["id"], {}).get("status") == "needs_fix")
    to_remove    = sum(1 for c in cases if review.get(c["id"], {}).get("status") == "remove")

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Total",                   total)
    col2.metric("Reviewed",                f"{reviewed}/{total}")
    col3.metric("✅ Approved",              approved)
    col4.metric("🔵 Acceptable w/ Nuance", nuance_loss)
    col5.metric("⚠️ Needs fix",             needs_fix)
    col6.metric("❌ Remove",                to_remove)

    st.markdown("---")

    # ── Filter cases ─────────────────────────────────────────────────────
    def _status(c: dict) -> str:
        return review.get(c["id"], {}).get("status") or "unreviewed"

    filtered = [
        c for c in cases
        if c["category"]       in cat_f
        and c["difficulty"]    in diff_f
        and c["question_type"] in qtype_f
        and _status(c)         in status_f
    ]

    if not filtered:
        st.info("No cases match the current filters.")
        return

    st.markdown(f"Showing **{len(filtered)}** of {total} cases.")

    # ── Resolve selected_case_id → index ──────────────────────────────────
    case_ids = [c["id"] for c in filtered]
    sel_id = st.session_state.selected_case_id
    if sel_id not in case_ids:
        sel_id = case_ids[0]
        st.session_state.selected_case_id = sel_id
    idx = case_ids.index(sel_id)

    # ── Overview table ───────────────────────────────────────────────────
    table_rows = []
    for c in filtered:
        rv   = review.get(c["id"], {})
        arts = c.get("expected_articles", [])
        table_rows.append({
            "ID":               c["id"],
            "Status":           STATUS_REVERSE.get(rv.get("status"), "— unreviewed —"),
            "Category":         c["category"],
            "Difficulty":       c["difficulty"],
            "Type":             c["question_type"],
            "Articles":         " | ".join(arts),
            "Question":         c["question"][:100],
            "Note":             rv.get("note", ""),
        })
    table_df = pd.DataFrame(table_rows)

    st.caption("Click a row to open it in the inspector below.")
    selection = st.dataframe(
        table_df,
        use_container_width=True,
        hide_index=True,
        height=320,
        on_select="rerun",
        selection_mode="single-row",
        key="rv_table",
    )

    # Row-click: update selected_case_id
    clicked_rows = []
    if hasattr(selection, "selection"):
        clicked_rows = selection.selection.get("rows", [])
    if clicked_rows:
        clicked_idx = min(clicked_rows[0], len(filtered) - 1)
        new_id = case_ids[clicked_idx]
        if new_id != st.session_state.selected_case_id:
            st.session_state.selected_case_id = new_id
            st.rerun()

    # ── Inspector ────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Inspect & Annotate")

    # Navigation bar — Prev / selectbox / Next
    nav_cols = st.columns([1, 6, 1])
    with nav_cols[0]:
        if st.button("◀ Prev", disabled=(idx == 0)):
            new_id = case_ids[idx - 1]
            st.session_state.selected_case_id = new_id
            st.session_state.rv_selectbox = new_id
            st.session_state.rv_table_reset = True
            st.rerun()
    with nav_cols[2]:
        if st.button("Next ▶", disabled=(idx >= len(filtered) - 1)):
            new_id = case_ids[idx + 1]
            st.session_state.selected_case_id = new_id
            st.session_state.rv_selectbox = new_id
            st.session_state.rv_table_reset = True
            st.rerun()
    with nav_cols[1]:
        if "rv_selectbox" not in st.session_state:
            st.session_state.rv_selectbox = sel_id
        jumped_to = st.selectbox(
            "Jump to case:",
            case_ids,
            label_visibility="collapsed",
            key="rv_selectbox",
        )
        if jumped_to != sel_id:
            st.session_state.selected_case_id = jumped_to
            st.session_state.rv_table_reset = True
            st.rerun()

    current_case   = filtered[idx]
    selected_id    = current_case["id"]
    current_review = review.get(selected_id, {})

    # ── Case detail ──────────────────────────────────────────────────────
    left, right = st.columns([3, 2])
    with left:
        st.markdown(
            f"**{selected_id}** &nbsp;|&nbsp; "
            f"`{current_case.get('category','—')}` &nbsp;|&nbsp; "
            f"`{current_case.get('difficulty','—')}` &nbsp;|&nbsp; "
            f"`{current_case.get('question_type','—')}` &nbsp;|&nbsp; "
            f"lang: `{current_case.get('language','en')}`"
        )

        st.markdown("**Question**")
        st.write(current_case["question"])

        st.markdown("**Reference Answer**")
        st.write(current_case.get("reference_answer", "—"))

        # ── Notes / cross-references ──────────────────────────────────
        notes = current_case.get("notes", "")
        if notes:
            # Separate batch tag from human-readable notes
            batch_tag = ""
            human_notes = notes
            batch_match = _re.match(r"^(batch=\S+)(.*)", notes, _re.DOTALL)
            if batch_match:
                batch_tag    = batch_match.group(1)
                human_notes  = batch_match.group(2).strip(" ;,")

            if batch_tag:
                st.caption(f"Generator batch: `{batch_tag}`")
            if human_notes:
                st.info(f"**Cross-references / notes:** {human_notes}")

            # Highlight any article numbers in notes that aren't in expected_articles
            expected_set   = set(current_case.get("expected_articles", []))
            extra_articles = sorted(set(_re.findall(r"\b(\d{1,3}[a-z]?)\b", notes)) - expected_set)
            # filter out obvious non-article numbers (single digits that are paragraph refs)
            extra_articles = [a for a in extra_articles if int(_re.sub(r"[a-z]","",a)) > 3]
            if extra_articles:
                st.caption(
                    "Articles mentioned in notes but not in expected list: "
                    + ", ".join(f"**{a}**" for a in extra_articles)
                    + " — consider whether these should be added to `expected_articles`."
                )

        # ── Expected articles (ground-truth cross-references) ─────────
        st.markdown("---")
        st.markdown("**Ground-truth articles** *(must be retrieved for this question)*")
        st.caption(
            "These are the CRR cross-references the RAG system must surface. "
            "The eval measures Recall@k / Hit@1 against this list."
        )

        expected = current_case.get("expected_articles", [])
        if not expected:
            st.write("—")
        else:
            for art_num in expected:
                fetched = st.session_state.fetched_articles.get(art_num)
                if fetched is None:
                    # Not yet fetched — show load button
                    btn_col, lbl_col = st.columns([1, 5])
                    with btn_col:
                        if st.button(
                            "📥 Load",
                            key=f"fetch_{selected_id}_{art_num}",
                            help=f"Fetch Article {art_num} text from Qdrant",
                        ):
                            with st.spinner(f"Fetching Article {art_num}…"):
                                st.session_state.fetched_articles[art_num] = _fetch_article_text(art_num)
                            st.rerun()
                    with lbl_col:
                        st.markdown(f"**Article {art_num}**")
                elif fetched.get("error"):
                    st.error(f"Article {art_num}: could not fetch — {fetched['error'][:300]}")
                else:
                    title = fetched.get("title", "")
                    text  = fetched.get("text", "")
                    header = f"Article {art_num}" + (f" — {title}" if title else "")
                    with st.expander(header, expanded=(len(expected) == 1)):
                        if text:
                            st.text(text[:4000] + (" …[truncated]" if len(text) > 4000 else ""))
                        else:
                            st.warning("No text found for this article in Qdrant.")

    with right:
        st.markdown("**Your Review**")
        current_label = STATUS_REVERSE.get(current_review.get("status"), "— unreviewed —")
        with st.form(key=f"rv_form_{selected_id}"):
            new_label = st.radio(
                "Mark as:",
                REVIEW_OPTIONS,
                index=REVIEW_OPTIONS.index(current_label),
            )
            new_note = st.text_area(
                "Note (optional):",
                value=current_review.get("note", ""),
                height=100,
                placeholder="e.g. 'Wrong expected article — should be 27 not 26'",
            )
            save_col, clear_col = st.columns(2)
            with save_col:
                save_pressed = st.form_submit_button("💾 Save", use_container_width=True)
            with clear_col:
                clear_pressed = st.form_submit_button("🗑 Clear", use_container_width=True)

        if save_pressed:
            review[selected_id] = {"status": REVIEW_STATUSES[new_label], "note": new_note}
            _save_review_status(review)
            st.session_state.rv_table_reset = True
            st.rerun()
        if clear_pressed:
            review.pop(selected_id, None)
            _save_review_status(review)
            st.session_state.rv_table_reset = True
            st.rerun()

    # ── Export ───────────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("Export review results"):
        approved_ids     = [c["id"] for c in cases if review.get(c["id"], {}).get("status") == "approved"]
        nuance_loss_ids  = [c["id"] for c in cases if review.get(c["id"], {}).get("status") == "nuance_loss"]
        flagged_ids      = [c["id"] for c in cases if review.get(c["id"], {}).get("status") == "needs_fix"]
        remove_ids       = [c["id"] for c in cases if review.get(c["id"], {}).get("status") == "remove"]
        unreviewed_ids   = [c["id"] for c in cases if not review.get(c["id"], {}).get("status")]

        st.markdown(f"- ✅ Approved:                   **{len(approved_ids)}**")
        st.markdown(f"- 🔵 Acceptable w/ Nuance Loss:  **{len(nuance_loss_ids)}**")
        st.markdown(f"- ⚠️ Needs fix:                  **{len(flagged_ids)}**")
        st.markdown(f"- ❌ Remove:                     **{len(remove_ids)}**")
        st.markdown(f"- ○ Unreviewed:                 **{len(unreviewed_ids)}**")

        if remove_ids:
            st.markdown("**Cases marked for removal:**")
            st.code("\n".join(remove_ids))
            st.caption(
                "Run `python -m evals.review_apply` to remove these from golden_dataset.jsonl "
                "(tool to be implemented)."
            )


# ---------------------------------------------------------------------------
# ── PAGE 2: Eval Results ────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def _section_scorecard(summary: dict, compare_summary: dict | None) -> None:
    st.subheader("Scorecard")
    ov  = summary.get("overall", {})
    cov = compare_summary.get("overall", {}) if compare_summary else {}

    # Retrieval metrics row
    cols = st.columns(len(METRIC_KEYS))
    for col, m in zip(cols, METRIC_KEYS):
        val  = ov.get(m)
        cval = cov.get(m)
        display = f"{val*100:.1f}%" if val is not None else "—"
        delta = None
        if compare_summary and val is not None and cval is not None:
            delta = f"{(val - cval)*100:+.1f}%"
        col.metric(label=METRIC_LABELS[m], value=display, delta=delta)

    # Answer quality (judge) metrics row — only when judge was enabled
    judge_vals = {k: ov.get(k) for k in JUDGE_METRIC_KEYS}
    has_judge = any(v is not None for v in judge_vals.values())
    if has_judge:
        st.markdown("**Answer Quality (LLM Judge)**")
        judge_cols = st.columns(len(JUDGE_METRIC_KEYS))
        for col, m in zip(judge_cols, JUDGE_METRIC_KEYS):
            val  = ov.get(m)
            cval = cov.get(m) if compare_summary else None
            display = f"{val:.3f}" if val is not None else "—"
            delta = None
            if compare_summary and val is not None and cval is not None:
                delta = f"{(val - cval):+.3f}"
            col.metric(label=JUDGE_METRIC_LABELS[m], value=display, delta=delta)


def _breakdown_table(data: dict[str, dict], show_judge: bool = False) -> None:
    rows = []
    for key, vals in data.items():
        row = {"Slice": key, "N": vals.get("n", 0)}
        for m in METRIC_KEYS:
            v = vals.get(m)
            row[METRIC_LABELS[m]] = f"{v*100:.1f}%" if v is not None else "—"
        if show_judge:
            for m in JUDGE_METRIC_KEYS:
                v = vals.get(m)
                row[JUDGE_METRIC_LABELS[m]] = f"{v:.3f}" if v is not None else "—"
        rows.append(row)
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _comparison_table(data_a: dict, data_b: dict, label_col: str, show_judge: bool = False) -> None:
    """Comparison table showing all retrieval metrics with per-metric delta columns."""
    metrics = list(METRIC_KEYS)
    if show_judge:
        metrics += list(JUDGE_METRIC_KEYS)

    _all_labels = {**METRIC_LABELS, **JUDGE_METRIC_LABELS}
    _judge_set  = set(JUDGE_METRIC_KEYS)

    # Build column list so we know delta column indices for highlighting
    fixed_cols  = [label_col, "N (A)", "N (B)"]
    metric_cols = []
    for m in metrics:
        short = _all_labels[m]
        metric_cols += [f"{short} A", f"{short} B", f"Δ {short}"]
    all_cols = fixed_cols + metric_cols

    rows = []
    for key in sorted(set(data_a) | set(data_b)):
        va_row = data_a.get(key, {})
        vb_row = data_b.get(key, {})
        row: dict = {
            label_col: key,
            "N (A)": va_row.get("n", "—"),
            "N (B)": vb_row.get("n", "—"),
        }
        for m in metrics:
            short = _all_labels[m]
            va = va_row.get(m)
            vb = vb_row.get(m)
            is_judge = m in _judge_set
            delta = (va - vb) if (va is not None and vb is not None) else None
            if is_judge:
                row[f"{short} A"] = f"{va:.3f}" if va is not None else "—"
                row[f"{short} B"] = f"{vb:.3f}" if vb is not None else "—"
                row[f"Δ {short}"] = f"{delta:+.3f}" if delta is not None else "—"
            else:
                row[f"{short} A"] = f"{va*100:.1f}%" if va is not None else "—"
                row[f"{short} B"] = f"{vb*100:.1f}%" if vb is not None else "—"
                row[f"Δ {short}"] = f"{delta*100:+.1f}%" if delta is not None else "—"
        rows.append(row)

    df = pd.DataFrame(rows, columns=all_cols)
    delta_cols = [c for c in all_cols if c.startswith("Δ ")]

    def _delta_style(val: str, cap: float = 15.0) -> str:
        """Map a delta string (e.g. '+4.2%') to a background/text CSS style.

        Magnitude is normalised against `cap` pp (clamped to [0, 1]), then
        linearly interpolated between a near-white and a saturated hue so that
        small deltas are barely tinted and large deltas are richly coloured.

        Positive  → green spectrum  (#e8f5e9 … #1b5e20)
        Negative  → red spectrum    (#ffebee … #7f0000)
        |Δ| < 0.5 → no colour (within noise)
        """
        if not isinstance(val, str) or val == "—":
            return ""
        try:
            num = float(val.replace("%", "").replace("+", ""))
        except ValueError:
            return ""
        if abs(num) < 0.5:
            return ""
        t = min(abs(num) / cap, 1.0)          # 0 → pale, 1 → saturated
        if num > 0:
            # light #e8f5e9 → dark #1b5e20
            r = int(232 + t * (27  - 232))
            g = int(245 + t * (94  - 245))
            b = int(233 + t * (32  - 233))
            text = "white" if t > 0.55 else "#1b5e20"
        else:
            # light #ffebee → dark #7f0000
            r = int(255 + t * (127 - 255))
            g = int(235 + t * (0   - 235))
            b = int(238 + t * (0   - 238))
            text = "white" if t > 0.45 else "#7f0000"
        return f"background-color:rgb({r},{g},{b});color:{text}"

    def _hl(row: pd.Series) -> list[str]:
        styles = [""] * len(row)
        for dc in delta_cols:
            if dc not in df.columns:
                continue
            idx = df.columns.get_loc(dc)
            styles[idx] = _delta_style(row.iloc[idx])
        return styles

    st.dataframe(df.style.apply(_hl, axis=1), use_container_width=True, hide_index=True)


def _count_valid_results(path: Path) -> int:
    """Count unique processed case IDs in a JSONL file (ok + error, dedup-aware)."""
    ids: set[str] = set()
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        ids.add(json.loads(line)["id"])
                    except Exception:
                        pass
    except FileNotFoundError:
        pass
    return len(ids)


def _count_results_by_status(path: Path) -> tuple[int, int]:
    """Return (ok_count, error_count) for unique case IDs in a JSONL file."""
    seen: dict[str, str] = {}  # id -> status (first occurrence wins)
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        row = json.loads(line)
                        case_id = row["id"]
                        if case_id not in seen:
                            seen[case_id] = row.get("status", "error")
                    except Exception:
                        pass
    except FileNotFoundError:
        pass
    ok = sum(1 for s in seen.values() if s == "ok")
    return ok, len(seen) - ok


def _count_valid_dataset_cases(path: Path) -> int:
    """Count unique valid case IDs in the golden dataset JSONL."""
    ids: set[str] = set()
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        ids.add(json.loads(line)["id"])
                    except Exception:
                        pass
    except FileNotFoundError:
        pass
    return len(ids)


def _is_pid_alive(pid: int) -> bool:
    """Return True if a process with the given PID is running."""
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _kill_eval_proc(proc: "subprocess.Popen | None", run_name: str) -> None:
    """Kill an eval subprocess and mark its state file as 'stopped'."""
    if proc is not None:
        try:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
        except Exception:
            pass
    # Also kill by PID from state file (handles orphan case where proc is None)
    state_path = RESULTS_DIR / f"{run_name}_state.json"
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            pid = state.get("pid", -1)
            if pid > 0 and _is_pid_alive(pid):
                try:
                    import signal as _sig
                    os.kill(pid, _sig.SIGTERM)
                except Exception:
                    pass
            state["status"] = "stopped"
            state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass
    for key in ("eval_proc", "eval_log_file", "eval_run_name", "eval_total_cases"):
        st.session_state.pop(key, None)


def _tail_log_file(path: Path, max_lines: int = 30) -> str:
    """Return the last *max_lines* lines of a log file, or a waiting message."""
    try:
        with open(path, encoding="utf-8") as fh:
            lines = fh.readlines()
        return "".join(lines[-max_lines:]) if lines else "(log file is empty)"
    except FileNotFoundError:
        return "Waiting for log output…"


def _run_eval_panel(empty: bool = True) -> None:
    """
    Shared UI panel for launching a new eval run and tracking its progress.
    Renders inline — call from page_eval_results_empty or as an expander.
    """
    # ── Orphan detection: check for externally-started runs (cross-session) ──
    proc: subprocess.Popen | None = st.session_state.get("eval_proc")
    if proc is None:
        for sf in sorted(RESULTS_DIR.glob("*_state.json"), reverse=True) if RESULTS_DIR.exists() else []:
            try:
                state = json.loads(sf.read_text(encoding="utf-8"))
            except Exception:
                continue
            if state.get("status") != "running":
                continue
            orphan_pid = state.get("pid", -1)
            if not _is_pid_alive(orphan_pid):
                # Process is dead — mark crashed so it does not block future launches.
                state["status"] = "crashed"
                try:
                    sf.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
                except Exception:
                    pass
                continue
            # A live orphan run found — surface it in the UI.
            orphan_name = state.get("run_name", sf.stem.replace("_state", ""))
            orphan_total = state.get("planned_total", 0)
            orphan_cases = RESULTS_DIR / f"{orphan_name}_cases.jsonl"
            orphan_ok, orphan_err = _count_results_by_status(orphan_cases)
            orphan_done = orphan_ok + orphan_err
            pct = min(orphan_done / orphan_total, 1.0) if orphan_total > 0 else 0
            err_note = f", **{orphan_err} errors**" if orphan_err else ""
            _pcol, _scol = st.columns([5, 1])
            with _pcol:
                st.progress(pct, text=f"Running **{orphan_name}** … {orphan_ok} ok{err_note} / {orphan_total} cases (detected from disk)")
            with _scol:
                if st.button("⏹ Stop", key="stop_orphan", type="secondary", use_container_width=True):
                    _kill_eval_proc(None, orphan_name)
                    st.warning(f"Eval **{orphan_name}** stopped.")
                    st.rerun()
                    return
            orphan_log = RESULTS_DIR / f"{orphan_name}.log"
            if orphan_log.exists():
                with st.expander("Live logs", expanded=False):
                    st.code(_tail_log_file(orphan_log, max_lines=30), language=None)
            time.sleep(1)
            st.rerun()
            return

    # ── If a run is currently in progress ────────────────────────────────
    if proc is not None:
        retcode = proc.poll()
        run_name   = st.session_state.get("eval_run_name", "")
        cases_path = RESULTS_DIR / f"{run_name}_cases.jsonl"
        n_ok, n_err = _count_results_by_status(cases_path)
        done = n_ok + n_err
        # Prefer planned_total from the runner's state.json (accurate post-filtering).
        state_path = RESULTS_DIR / f"{run_name}_state.json"
        total = st.session_state.get("eval_total_cases", 173)
        if state_path.exists():
            try:
                total = json.loads(state_path.read_text(encoding="utf-8")).get("planned_total", total)
            except Exception:
                pass
        pct        = min(done / total, 1.0) if total > 0 else 0

        err_note = f", **{n_err} errors**" if n_err else ""
        _pcol, _scol = st.columns([5, 1])
        with _pcol:
            st.progress(pct, text=f"Running **{run_name}** … {n_ok} ok{err_note} / {total} cases processed")
        with _scol:
            if st.button("⏹ Stop", key="stop_eval", type="secondary", use_container_width=True):
                _kill_eval_proc(proc, run_name)
                st.warning(f"Eval **{run_name}** stopped ({n_ok} ok, {n_err} errors recorded so far).")
                st.rerun()
                return

        log_path: Path | None = st.session_state.get("eval_log_file")
        if log_path is not None:
            with st.expander("Live logs", expanded=True):
                st.code(_tail_log_file(log_path, max_lines=30), language=None)

        if retcode is None:
            # Still running — schedule a rerun in ~1 s
            time.sleep(1)
            st.rerun()
        elif retcode == 0:
            st.session_state.pop("eval_proc", None)
            st.session_state.pop("eval_log_file", None)
            if n_err:
                st.warning(f"⚠️ Eval **{run_name}** complete — {n_ok} ok, **{n_err} failed** (see scorecard for details).")
            else:
                st.success(f"✅ Eval **{run_name}** complete — {n_ok} / {total} cases ok.")
            time.sleep(0.5)
            st.rerun()
        else:
            st.session_state.pop("eval_proc", None)
            error_text = _tail_log_file(log_path, max_lines=50) if log_path else "(no log file)"
            st.session_state.pop("eval_log_file", None)
            st.error(f"Eval failed (exit code {retcode}).")
            with st.expander("Error log", expanded=True):
                st.code(error_text, language=None)
        return   # don't show the launch form while a run is active

    # ── Launch form ───────────────────────────────────────────────────────
    if empty:
        st.markdown("### No eval runs found yet")
        st.markdown(
            "Once a run completes this page will show a **scorecard**, "
            "**breakdowns** by difficulty/category/question-type, and a **case drill-down**."
        )
        st.markdown("---")

    dataset_label = st.radio(
        "Dataset",
        list(KNOWN_DATASETS.keys()),
        horizontal=True,
        key="eval_dataset_choice",
        help="Golden dataset is used for tuning; Test set is the held-out evaluation.",
    )
    selected_dataset_path = KNOWN_DATASETS[dataset_label]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        run_name_input = st.text_input(
            "Run name", placeholder="e.g. baseline  (leave blank for timestamp)"
        )
    with c2:
        workers = st.slider("Parallel workers", min_value=1, max_value=8, value=1)
    with c3:
        limit = st.number_input(
            "Limit cases (0 = all)", min_value=0, max_value=1000, value=0, step=10
        )
    with c4:
        req_timeout = st.number_input(
            "Request timeout (s)", min_value=30, max_value=600, value=120, step=30,
            help="Per-query HTTP timeout. Increase if the API is slow (BGE-M3 on CPU needs ~30-60 s).",
        )

    api_url = st.text_input("API URL", value="http://localhost:8080")
    description_input = st.text_input(
        "Description (optional)",
        placeholder="e.g. Testing RETRIEVAL_ALPHA=0.7 + TOP_K=20 for diluted_embedding failures",
        help="Saved to config.json alongside results — useful for the experiment log",
    )

    col_judge, col_resume, col_autostart = st.columns(3)
    with col_judge:
        enable_judge = st.checkbox(
            "LLM-as-judge (gpt-4o)",
            value=False,
            help="Score each answer on correctness, completeness, faithfulness",
        )
    with col_resume:
        no_resume = st.checkbox(
            "Fresh start (ignore previous results)",
            value=False,
            help="Delete any partial results for this run name and start from case 1",
        )
    with col_autostart:
        auto_start_api = st.checkbox(
            "Auto-start API if not running",
            value=True,
            help="Spawn uvicorn automatically and wait for the index to load",
        )

    # Show live API health status
    try:
        import requests as _req
        _r = _req.get(f"{api_url.rstrip('/')}/health", timeout=2)
        _d = _r.json()
        if _d.get("index_loaded"):
            st.success(f"API healthy — {_d.get('vector_store_items', '?')} index items")
        else:
            st.warning("API is up but index not yet loaded.")
    except Exception:
        st.warning("API not reachable — will auto-start if checkbox is enabled.")

    col_btn, col_note = st.columns([1, 4])
    with col_btn:
        start = st.button("▶ Run Eval", type="primary", use_container_width=True)
    with col_note:
        st.caption("Auto-start API is on by default — the runner will spawn uvicorn if needed.")

    if start:
        run_name = run_name_input.strip() or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        total    = int(limit) if limit > 0 else _count_valid_dataset_cases(selected_dataset_path)

        cmd = [
            sys.executable, "-m", "evals.run_eval",
            "--run-name", run_name,
            "--workers", str(workers),
            "--api-url", api_url,
            "--timeout", str(int(req_timeout)),
        ]
        if selected_dataset_path != DATASET_PATH:
            cmd += ["--dataset", str(selected_dataset_path)]
        if limit > 0:
            cmd += ["--limit", str(limit)]
        if enable_judge:
            cmd += ["--judge"]
        if no_resume:
            cmd += ["--no-resume"]
        if auto_start_api:
            cmd += ["--auto-start-api"]
        if description_input.strip():
            cmd += ["--description", description_input.strip()]

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        log_path = RESULTS_DIR / f"{run_name}.log"
        cmd += ["--log-file", str(log_path)]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        st.session_state["eval_proc"]         = proc
        st.session_state["eval_run_name"]     = run_name
        st.session_state["eval_total_cases"]  = total
        st.session_state["eval_log_file"]     = log_path
        st.rerun()


def _section_visual_analysis(filtered_df: pd.DataFrame, ok_df: pd.DataFrame, summary: dict) -> None:
    """Render the Visual Analysis section between Breakdowns and Case Drill-Down."""
    st.subheader("Visual Analysis")

    if not HAS_PLOTLY:
        st.info("Install plotly for charts: pip install plotly")
        return

    if filtered_df.empty:
        st.warning("No cases match the current filters.")
        return

    # ── Tier 1: KPI row ──────────────────────────────────────────────────────
    total_ok    = len(ok_df)
    total_filt  = len(filtered_df)

    def _mean(col: str) -> float | None:
        if col in filtered_df.columns and filtered_df[col].notna().any():
            return filtered_df[col].mean()
        return None

    kpi_cols = st.columns(6)
    kpi_cols[0].metric("Cases (filtered/total)", f"{total_filt} / {total_ok}")
    for i, (col, label) in enumerate([
        ("hit_at_1",   "Hit@1"),
        ("recall_at_1","Recall@1"),
        ("recall_at_3","Recall@3"),
        ("recall_at_5","Recall@5"),
        ("mrr",        "MRR"),
    ], start=1):
        v = _mean(col)
        kpi_cols[i].metric(label, f"{v*100:.1f}%" if v is not None else "—")

    lat_col = "latency_ms"
    if lat_col in filtered_df.columns and filtered_df[lat_col].notna().any():
        p50 = filtered_df[lat_col].quantile(0.5)
        st.metric("Latency p50", f"{p50:.0f} ms")

    has_judge = any(
        m in filtered_df.columns and filtered_df[m].notna().any()
        for m in JUDGE_METRIC_KEYS
    )
    if has_judge:
        j_cols = st.columns(3)
        for i, (col, label) in enumerate(zip(
            ["judge_correctness", "judge_completeness", "judge_faithfulness"],
            ["Correctness", "Completeness", "Faithfulness"],
        )):
            v = _mean(col)
            j_cols[i].metric(label, f"{v:.2f}" if v is not None else "—")

    # ── Facet dimension selector ──────────────────────────────────────────────
    available_dims = [d for d in ["difficulty", "question_type", "category", "citation_type"]
                      if d in filtered_df.columns]
    if not available_dims:
        st.info("No grouping dimensions found in this run.")
        return

    facet_dim = st.selectbox(
        "Slice charts by", available_dims, index=0, key="chart_facet_dim"
    )

    # ── Tier 2: Four chart tabs ───────────────────────────────────────────────
    tab_metrics, tab_latency, tab_failures, tab_judge = st.tabs(
        ["Metric Distributions", "Latency", "Failure Analysis", "Judge Scores"]
    )

    # ── Tab 1: Metric Distributions ──────────────────────────────────────────
    with tab_metrics:
        recall_metrics = [c for c in ["recall_at_1","recall_at_3","recall_at_5"]
                          if c in filtered_df.columns]
        if recall_metrics and facet_dim in filtered_df.columns:
            # Grouped bar: Recall@k by facet
            rows = []
            for grp, grp_df in filtered_df.groupby(facet_dim):
                for col in recall_metrics:
                    rows.append({
                        "Metric": METRIC_LABELS.get(col, col),
                        facet_dim: str(grp),
                        "Mean": grp_df[col].mean() if grp_df[col].notna().any() else 0,
                    })
            recall_bar_df = pd.DataFrame(rows)
            fig1 = px.bar(
                recall_bar_df, x="Metric", y="Mean", color=facet_dim,
                barmode="group", title="Recall@k by group",
                labels={"Mean": "Mean value"},
            )
            fig1.update_layout(height=350, yaxis_tickformat=".0%")
            st.plotly_chart(fig1, use_container_width=True)

        if "hit_at_1" in filtered_df.columns and facet_dim in filtered_df.columns:
            # Stacked 100% bar: hit vs miss proportion by facet
            hit_rows = []
            for grp, grp_df in filtered_df.groupby(facet_dim):
                n     = len(grp_df)
                hits  = int(grp_df["hit_at_1"].sum())
                misses = n - hits
                hit_rows.append({facet_dim: str(grp), "Result": "Hit",  "Count": hits,   "Pct": hits / n * 100})
                hit_rows.append({facet_dim: str(grp), "Result": "Miss", "Count": misses, "Pct": misses / n * 100})
            hit_df = pd.DataFrame(hit_rows)
            fig2 = px.bar(
                hit_df, x=facet_dim, y="Pct", color="Result",
                barmode="stack", title="Hit@1 proportion by group",
                color_discrete_map={"Hit": "#28a745", "Miss": "#dc3545"},
                labels={"Pct": "Percentage (%)"},
            )
            fig2.update_layout(height=350)
            st.plotly_chart(fig2, use_container_width=True)

    # ── Tab 2: Latency ────────────────────────────────────────────────────────
    with tab_latency:
        if lat_col in filtered_df.columns and filtered_df[lat_col].notna().any():
            p50v = filtered_df[lat_col].quantile(0.5)
            p90v = filtered_df[lat_col].quantile(0.9)

            color_arg = facet_dim if facet_dim in filtered_df.columns else None
            fig3 = px.histogram(
                filtered_df, x=lat_col, nbins=25,
                color=color_arg,
                title="Latency distribution",
                labels={lat_col: "Latency (ms)"},
            )
            fig3.add_vline(x=p50v, line_dash="dash", line_color="blue",
                           annotation_text=f"p50={p50v:.0f}ms")
            fig3.add_vline(x=p90v, line_dash="dash", line_color="red",
                           annotation_text=f"p90={p90v:.0f}ms")
            fig3.update_layout(height=350)
            st.plotly_chart(fig3, use_container_width=True)

            if facet_dim in filtered_df.columns:
                fig4 = px.box(
                    filtered_df, x=facet_dim, y=lat_col,
                    title="Latency by group", color=facet_dim,
                    labels={lat_col: "Latency (ms)"},
                )
                fig4.update_layout(height=350, showlegend=False)
                st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("No latency data available.")

    # ── Tab 3: Failure Analysis ───────────────────────────────────────────────
    with tab_failures:
        if "hit_at_1" in filtered_df.columns and facet_dim in filtered_df.columns:
            fail_rows = []
            for grp, grp_df in filtered_df.groupby(facet_dim):
                fail_rate = 1 - grp_df["hit_at_1"].mean()
                fail_rows.append({facet_dim: str(grp), "Failure Rate": fail_rate})
            fail_df = pd.DataFrame(fail_rows).sort_values("Failure Rate", ascending=False)
            fig5 = px.bar(
                fail_df, x="Failure Rate", y=facet_dim,
                orientation="h", title="Failure rate by group (worst first)",
                labels={"Failure Rate": "Failure Rate (1 - Hit@1)"},
                color="Failure Rate", color_continuous_scale="RdYlGn_r",
            )
            fig5.update_layout(height=350, xaxis_tickformat=".0%")
            st.plotly_chart(fig5, use_container_width=True)

        recall_col = "recall_at_3"
        mrr_col    = "mrr"
        if all(c in filtered_df.columns for c in [recall_col, mrr_col, "id", "question"]):
            scatter_df = filtered_df[[recall_col, mrr_col, "id", "question", facet_dim]].dropna()
            fig6 = px.scatter(
                scatter_df, x=recall_col, y=mrr_col,
                color=facet_dim,
                hover_data={"id": True, "question": True},
                title="Recall@3 vs MRR per case",
                labels={recall_col: "Recall@3", mrr_col: "MRR"},
            )
            fig6.update_layout(height=350)
            st.plotly_chart(fig6, use_container_width=True)

    # ── Tab 4: Judge Scores ───────────────────────────────────────────────────
    with tab_judge:
        if not has_judge:
            st.info("Run eval with --judge to enable judge score charts.")
        else:
            judge_present = [m for m in JUDGE_METRIC_KEYS
                             if m in filtered_df.columns and filtered_df[m].notna().any()]
            if facet_dim in filtered_df.columns and judge_present:
                melt_cols = [facet_dim] + judge_present
                judge_melt = filtered_df[melt_cols].melt(
                    id_vars=facet_dim, var_name="Metric", value_name="Score"
                )
                judge_melt["Metric"] = judge_melt["Metric"].map(JUDGE_METRIC_LABELS)
                fig7 = px.box(
                    judge_melt, x="Metric", y="Score",
                    color=facet_dim, title="Judge scores by group",
                )
                fig7.update_layout(height=350)
                st.plotly_chart(fig7, use_container_width=True)

            recall_col = "recall_at_3"
            corr_col   = "judge_correctness"
            if all(c in filtered_df.columns for c in [recall_col, corr_col]):
                scatter2_df = filtered_df[[recall_col, corr_col, facet_dim]].dropna()
                if not scatter2_df.empty:
                    fig8 = px.scatter(
                        scatter2_df, x=recall_col, y=corr_col,
                        color=facet_dim,
                        title="Correctness vs Recall@3",
                        labels={recall_col: "Recall@3", corr_col: "Correctness"},
                        trendline="ols",
                    )
                    fig8.update_layout(height=350)
                    st.plotly_chart(fig8, use_container_width=True)

    # ── Tier 3: Heatmap ───────────────────────────────────────────────────────
    heatmap_dims = [d for d in ["category", "question_type"] if d in filtered_df.columns]
    if len(heatmap_dims) == 2:
        st.markdown("#### Cross-dimension Heatmap")
        metric_options = {METRIC_LABELS[m]: m for m in METRIC_KEYS if m in filtered_df.columns}
        chosen_label   = st.selectbox("Heatmap metric", list(metric_options.keys()),
                                      index=list(metric_options.keys()).index("Recall@3")
                                      if "Recall@3" in metric_options else 0,
                                      key="heatmap_metric")
        chosen_metric  = metric_options[chosen_label]

        pivot = filtered_df.pivot_table(
            values=chosen_metric, index="category", columns="question_type",
            aggfunc="mean", observed=True,
        )
        # Grey out cells with < 2 cases
        count_pivot = filtered_df.pivot_table(
            values=chosen_metric, index="category", columns="question_type",
            aggfunc="count", observed=True,
        )
        pivot[count_pivot < 2] = float("nan")

        n_rows = len(pivot)
        n_cols = len(pivot.columns)
        fig9 = px.imshow(
            pivot, color_continuous_scale="RdYlGn", zmin=0, zmax=1,
            title=f"Mean {chosen_label}: Category × Question Type",
            labels={"color": chosen_label},
            text_auto=".2f",
            aspect="auto",
        )
        fig9.update_traces(textfont_size=14)
        fig9.update_layout(
            height=max(400, 80 * n_rows + 100),
            xaxis=dict(tickangle=-30, tickfont_size=12),
            yaxis=dict(tickfont_size=12),
        )
        st.plotly_chart(fig9, use_container_width=True)
        st.caption("Grey cells = fewer than 2 cases.")


def page_eval_results_empty() -> None:
    st.header("Eval Results")
    _run_eval_panel(empty=True)


def page_eval_results(runs: list) -> None:
    st.header("Eval Results")

    # Show any incomplete (partial/failed/crashed) runs as a notice.
    incomplete = _discover_incomplete_runs()
    if incomplete:
        for inc in incomplete:
            status = inc.get("status", "unknown")
            name = inc.get("run_name", "?")
            planned = inc.get("planned_total", "?")
            if status == "failed":
                st.warning(f"Run **{name}** failed ({planned} cases planned). Use '▶ Start' with the same name to resume.")
            elif status == "crashed":
                st.warning(f"Run **{name}** crashed ({planned} cases planned). Use '▶ Start' with the same name to resume.")

    with st.expander("▶ Start a new eval run", expanded=False):
        _run_eval_panel(empty=False)

    run_names = [r[0] for r in runs]

    with st.sidebar:
        st.markdown("---")
        st.subheader("Run Selection")
        selected_run = st.selectbox("Primary run", run_names, index=0, key="er_primary")
        compare_run  = st.selectbox(
            "Compare against",
            ["— none —"] + [r for r in run_names if r != selected_run],
            index=0, key="er_compare",
        )
        compare_run = None if compare_run == "— none —" else compare_run

        st.markdown("---")
        st.subheader("Drill-Down Filters")

    _, primary_sp, primary_cp = next(r for r in runs if r[0] == selected_run)
    df      = _load_run(str(primary_cp), file_mtime=primary_cp.stat().st_mtime if primary_cp.exists() else 0.0)
    summary = _load_summary(str(primary_sp), file_mtime=primary_sp.stat().st_mtime if primary_sp.exists() else 0.0)

    primary_cfg_path = RESULTS_DIR / f"{selected_run}_config.json"
    config = _load_config(
        str(primary_cfg_path),
        file_mtime=primary_cfg_path.stat().st_mtime if primary_cfg_path.exists() else 0.0,
    )

    compare_summary = None
    if compare_run:
        _, csp, _ = next(r for r in runs if r[0] == compare_run)
        compare_summary = _load_summary(str(csp), file_mtime=csp.stat().st_mtime if csp.exists() else 0.0)

    # Run metadata
    s = summary
    desc = config.get("description") or ""
    n_failed = s.get("failed_cases", 0) or 0
    n_total  = s.get("total_cases", 0) or 0
    info_fn  = st.warning if n_failed else st.info
    info_fn(
        f"**Run:** {s.get('run_name', '—')}  |  "
        f"**Cases:** {s.get('successful_cases', '—')} ok / {n_failed} failed / {n_total} total  |  "
        f"**Latency p50/p90:** {s.get('p50_latency_ms','—')} / {s.get('p90_latency_ms','—')} ms  |  "
        f"**API:** {s.get('api_url','—')}"
        + (f"  \n📝 {desc}" if desc else "")
    )
    if n_failed:
        pct_failed = round(n_failed / n_total * 100, 1) if n_total else 0
        st.error(
            f"⚠️ **{n_failed} cases failed ({pct_failed}%)** — metrics are computed only over the "
            f"{s.get('successful_cases', '?')} successful cases and may not reflect full dataset performance. "
            "Check the Cases tab below for `error_type` details."
        )

    if config:
        with st.expander("⚙️ Run configuration", expanded=False):
            r = config.get("retrieval", {})
            q = config.get("query_pipeline", {})
            syn = config.get("synthesis", {})
            ev = config.get("eval", {})
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Retrieval**")
                st.markdown(
                    f"- `TOP_K` = {r.get('top_k', '—')}\n"
                    f"- `RERANK_TOP_N` = {r.get('rerank_top_n', '—')}\n"
                    f"- `ALPHA` = {r.get('alpha', '—')}\n"
                    f"- `USE_RERANKER` = {r.get('use_reranker', '—')}\n"
                    f"- `RERANKER_MODEL` = `{r.get('reranker_model', '—')}`\n"
                    f"- `BLEND_ALPHA` = {r.get('rerank_blend_alpha', '—')}\n"
                    f"- `TITLE_BOOST` = {r.get('title_boost_weight', '—')}\n"
                    f"- `ADJ_TIEBREAK_DELTA` = {r.get('adjacent_tiebreak_delta', '—')}"
                )
            with col2:
                st.markdown("**Synthesis & Pipeline**")
                st.markdown(
                    f"- `LLM` = `{syn.get('llm_model', '—')}`\n"
                    f"- `HARD_QUERY_MODEL` = `{syn.get('hard_query_model', '—')}`\n"
                    f"- `USE_HYDE` = {q.get('use_hyde', '—')}\n"
                    f"- `USE_TOC_ROUTING` = {q.get('use_toc_routing', '—')}\n"
                    f"- `USE_ENRICHMENT` = {q.get('use_query_enrichment', '—')}"
                )
            with col3:
                st.markdown("**Run**")
                st.markdown(
                    f"- `workers` = {ev.get('workers', '—')}\n"
                    f"- `judge` = {ev.get('judge_enabled', '—')}\n"
                    f"- `judge_model` = `{ev.get('judge_model', '—')}`\n"
                    f"- `git` = `{config.get('git_commit', '—')}`"
                )

    st.markdown("---")
    _section_scorecard(summary, compare_summary)
    st.markdown("---")

    # Breakdowns
    st.subheader("Breakdowns")
    tab_cat, tab_diff, tab_qtype, tab_size = st.tabs(
        ["By Category", "By Difficulty", "By Question Type", "Single vs Multi-Article"]
    )

    for tab, key, order in [
        (tab_cat,  "by_category",      None),
        (tab_diff, "by_difficulty",    ["easy", "medium", "hard"]),
        (tab_qtype,"by_question_type", None),
    ]:
        with tab:
            data = summary.get(key, {})
            if order:
                data = {k: data[k] for k in order if k in data}
            if compare_summary:
                _comparison_table(data, compare_summary.get(key, {}), "Slice",
                                  show_judge=summary.get("judge_enabled", False))
            else:
                chart_df = pd.DataFrame(
                    [{"Slice": k, "Recall@3 (%)": round((v.get("recall_at_3") or 0)*100, 1)}
                     for k, v in data.items()]
                ).sort_values("Recall@3 (%)", ascending=True)
                if not chart_df.empty:
                    st.bar_chart(chart_df.set_index("Slice")["Recall@3 (%)"],
                                 use_container_width=True, horizontal=True)
                _breakdown_table(data, show_judge=summary.get("judge_enabled", False))

    with tab_size:
        data = {k.capitalize(): v for k, v in summary.get("by_article_count", {}).items()}
        if compare_summary:
            cdata = {k.capitalize(): v for k, v in compare_summary.get("by_article_count", {}).items()}
            _comparison_table(data, cdata, "Slice",
                              show_judge=summary.get("judge_enabled", False))
        else:
            _breakdown_table(data, show_judge=summary.get("judge_enabled", False))

    st.markdown("---")

    # Build ok_df and sidebar filters here so they apply to both visual analysis and drill-down
    if df.empty or "status" not in df.columns:
        st.warning("No data in this run file.")
        return

    ok_df = df[df["status"] == "ok"].copy()
    if ok_df.empty:
        err_df = df[df["status"] != "ok"]
        st.error(
            f"All {len(err_df)} cases in this run failed. "
            f"No retrieval metrics to display."
        )
        if "error_type" in err_df.columns:
            breakdown = err_df["error_type"].value_counts().to_dict()
            st.write("**Error breakdown:**", breakdown)
        if "error_message" in err_df.columns:
            sample = err_df["error_message"].dropna().iloc[0] if not err_df["error_message"].dropna().empty else ""
            if sample:
                st.write(f"**Sample error:** {sample}")
        st.info(
            "If errors are timeouts, restart the API and wait for the "
            "'BGE-M3 warm-up complete' log line before running evals. "
            "You can also increase the **Request timeout** in the Run Eval panel."
        )
        # Still show the error cases in a drill-down table so the user can inspect them
        st.subheader("Error Cases")
        err_show = err_df[["id", "question", "error_type", "error_message", "latency_ms",
                            "category", "difficulty", "question_type"]].copy()
        err_show = err_show.rename(columns={"error_type": "Type", "error_message": "Message"})
        st.dataframe(err_show, use_container_width=True, hide_index=True, height=400)
        return

    with st.sidebar:
        cats    = sorted(ok_df["category"].unique())      if "category"      in ok_df.columns else []
        diffs   = [d for d in ("easy","medium","hard") if d in ok_df.get("difficulty", pd.Series()).unique()]
        qtypes  = sorted(ok_df["question_type"].unique()) if "question_type" in ok_df.columns else []
        cat_f   = st.multiselect("Category",      cats,   default=cats,   key="er_cat")
        diff_f  = st.multiselect("Difficulty",    diffs,  default=diffs,  key="er_diff")
        qtype_f = st.multiselect("Question Type", qtypes, default=qtypes, key="er_qtype")

    filtered_df = ok_df.copy()
    # Always apply .isin() — empty selection = zero rows (consistent with Dataset Review)
    if "category" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["category"].isin(cat_f)]
    if "difficulty" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["difficulty"].isin(diff_f)]
    if "question_type" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["question_type"].isin(qtype_f)]

    # Visual Analysis section
    _section_visual_analysis(filtered_df, ok_df, summary)

    st.markdown("---")

    # Drill-down
    st.subheader("Case Drill-Down")

    failures_only = st.checkbox("Failures only (Hit@1 = 0)")
    if failures_only:
        filtered_df = filtered_df[filtered_df["hit_at_1"] == 0]

    st.markdown(f"Showing **{len(filtered_df)}** cases.")

    if "er_selected_id" not in st.session_state:
        st.session_state.er_selected_id = None

    # Include judge columns when they have data
    show_cols = ["id", "question", "expected_articles", "retrieved_articles",
                 "hit_at_1", "recall_at_3", "mrr", "latency_ms", "category", "difficulty", "question_type"]
    judge_cols_present = [m for m in JUDGE_METRIC_KEYS
                          if m in filtered_df.columns and filtered_df[m].notna().any()]
    show_cols += judge_cols_present
    present = [c for c in show_cols if c in filtered_df.columns]
    show_df = filtered_df[present].copy()
    if "question" in show_df.columns:
        show_df["question"] = show_df["question"].str[:90]
    for col in ("expected_articles", "retrieved_articles"):
        if col in show_df.columns:
            show_df[col] = show_df[col].apply(lambda v: ", ".join(v) if isinstance(v, list) else str(v))
    for m in judge_cols_present:
        show_df[m] = show_df[m].apply(
            lambda v: round(float(v), 3) if v is not None and str(v) != "nan" else None
        )
    show_df = show_df.rename(columns={m: JUDGE_METRIC_LABELS[m] for m in judge_cols_present})

    def _colour_hit(val):
        if val == 1: return "background-color:#d4edda"
        if val == 0: return "background-color:#f8d7da"
        return ""

    styled = show_df.style.map(_colour_hit, subset=["hit_at_1"]) if "hit_at_1" in show_df else show_df.style

    st.caption("Click a row to inspect it below.")
    selection = st.dataframe(
        styled, use_container_width=True, hide_index=True, height=320,
        on_select="rerun", selection_mode="single-row", key="er_drill_table",
    )

    _, exp_col = st.columns([6, 1])
    with exp_col:
        st.download_button(
            "⬇️ Export CSV",
            data=show_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{selected_run}_drilldown.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # Resolve row click → selected_id
    case_ids = filtered_df["id"].tolist()
    if not case_ids:
        return

    clicked_rows = []
    if hasattr(selection, "selection"):
        clicked_rows = selection.selection.get("rows", [])
    if clicked_rows:
        clicked_id = filtered_df.iloc[min(clicked_rows[0], len(filtered_df) - 1)]["id"]
        if clicked_id != st.session_state.er_selected_id:
            st.session_state.er_selected_id = clicked_id
            st.rerun()

    sel_id = st.session_state.get("er_selected_id")
    if sel_id not in case_ids:
        sel_id = case_ids[0]
        st.session_state.er_selected_id = sel_id

    # Case inspector — load reference answers from whichever dataset this run used
    st.markdown("---")
    dataset_src = config.get("dataset_path", str(DATASET_PATH))
    golden = {c["id"]: c for c in _load_cases(dataset_src)}
    selected_id = sel_id
    row = filtered_df[filtered_df["id"] == selected_id].iloc[0]
    golden_case = golden.get(selected_id, {})

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Question**"); st.write(row.get("question", ""))
        st.markdown("**Expected articles**"); st.write(", ".join(row.get("expected_articles", [])))
        st.markdown("**Retrieved articles**")
        expected_set = set(row.get("expected_articles", []))
        for i, a in enumerate(row.get("retrieved_articles", [])[:8], 1):
            st.write(f"{i}. {'✅' if a in expected_set else '❌'} Article {a}")
    with col_b:
        st.markdown("**Metrics**")
        metrics_dict = {METRIC_LABELS[m]: row.get(m) for m in METRIC_KEYS}
        # Append judge scores when present
        judge_vals_row = {JUDGE_METRIC_LABELS[m]: row.get(m) for m in JUDGE_METRIC_KEYS}
        if any(v is not None for v in judge_vals_row.values()):
            metrics_dict.update(judge_vals_row)
        st.table(pd.DataFrame.from_dict(metrics_dict, orient="index", columns=["Value"]))
        st.write(f"Latency: {row.get('latency_ms')} ms")

    st.markdown("**LLM Answer**"); st.write(row.get("answer", "—"))

    # Judge rationale (when available)
    rationale = row.get("judge_rationale")
    if rationale and str(rationale).strip() and str(rationale) != "None":
        st.markdown("**Judge Rationale**")
        st.info(str(rationale))

    if golden_case.get("reference_answer"):
        st.markdown("**Reference Answer**"); st.write(golden_case["reference_answer"])

    sources_raw = row.get("sources_raw", [])
    if isinstance(sources_raw, str):
        try: sources_raw = json.loads(sources_raw)
        except Exception: sources_raw = []
    if sources_raw:
        st.markdown("**Sources**")
        st.dataframe(
            pd.DataFrame(sources_raw)[["article", "article_title", "score", "expanded"]],
            use_container_width=True, hide_index=True,
        )


# ---------------------------------------------------------------------------
# ── PAGE 3: Compare Runs ────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def _cmp_breakdown_table(cmp_data: dict, metric: str, label_col: str) -> None:
    """Render a comparison table for one breakdown dimension."""
    rows = []
    for key, vals in cmp_data.items():
        va    = vals.get(metric, {}).get("a")
        vb    = vals.get(metric, {}).get("b")
        delta = vals.get(metric, {}).get("delta")
        sign  = "+" if (delta or 0) >= 0 else ""
        rows.append({
            label_col:                   key,
            "N (A)":                     vals.get("n_a", 0),
            "N (B)":                     vals.get("n_b", 0),
            f"{METRIC_LABELS[metric]} A": f"{va * 100:.1f}%" if va is not None else "—",
            f"{METRIC_LABELS[metric]} B": f"{vb * 100:.1f}%" if vb is not None else "—",
            "Delta":                     f"{sign}{(delta or 0) * 100:.1f}pp" if delta is not None else "—",
        })
    if not rows:
        st.write("No data.")
        return

    df = pd.DataFrame(rows)

    def _hl(row: pd.Series) -> list[str]:
        styles = [""] * len(row)
        if "Delta" in df.columns:
            idx = df.columns.get_loc("Delta")
            raw = row.iloc[idx]
            if isinstance(raw, str) and raw.startswith("+"):
                styles[idx] = "background-color:#d4edda;color:#155724"
            elif isinstance(raw, str) and raw.startswith("-"):
                styles[idx] = "background-color:#f8d7da;color:#721c24"
        return styles

    st.dataframe(df.style.apply(_hl, axis=1), use_container_width=True, hide_index=True)


def page_compare_runs(runs: list) -> None:
    st.header("Run Comparison")

    if len(runs) < 2:
        st.info(
            "Need at least **2 completed runs** to compare. "
            "Run the eval pipeline twice — e.g. before and after a change — "
            "then return here.\n\n"
            "```bash\npython -m evals.run_eval --run-name baseline\n"
            "# make your change …\n"
            "python -m evals.run_eval --run-name after_change\n```"
        )
        return

    run_names = [r[0] for r in runs]

    with st.sidebar:
        st.markdown("---")
        st.subheader("Runs to Compare")
        run_a_name = st.selectbox(
            "Baseline (A)", run_names,
            index=min(1, len(run_names) - 1),
            key="cmp_a",
        )
        remaining = [r for r in run_names if r != run_a_name]
        run_b_name = st.selectbox("Candidate (B)", remaining, index=0, key="cmp_b")

    _, sp_a, _ = next(r for r in runs if r[0] == run_a_name)
    _, sp_b, _ = next(r for r in runs if r[0] == run_b_name)

    summary_a = _load_summary(str(sp_a), file_mtime=sp_a.stat().st_mtime if sp_a.exists() else 0.0)
    summary_b = _load_summary(str(sp_b), file_mtime=sp_b.stat().st_mtime if sp_b.exists() else 0.0)

    cfg_path_a = RESULTS_DIR / f"{run_a_name}_config.json"
    cfg_path_b = RESULTS_DIR / f"{run_b_name}_config.json"
    config_a = _load_config(str(cfg_path_a), file_mtime=cfg_path_a.stat().st_mtime if cfg_path_a.exists() else 0.0)
    config_b = _load_config(str(cfg_path_b), file_mtime=cfg_path_b.stat().st_mtime if cfg_path_b.exists() else 0.0)

    try:
        from evals.compare import build_comparison, METRIC_KEYS as _CMP_METRIC_KEYS
        cmp = build_comparison(summary_a, summary_b)
    except ImportError:
        st.error("Could not import `evals.compare`. Make sure `compare.py` exists in `evals/`.")
        return

    # ── Run info banners ─────────────────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        s = summary_a
        desc_a = config_a.get("description") or ""
        st.info(
            f"**A — baseline:** `{s.get('run_name', '?')}`  \n"
            f"{s.get('successful_cases', '?')} cases  |  p50: {s.get('p50_latency_ms', '—')} ms"
            + (f"  \n📝 {desc_a}" if desc_a else "")
        )
    with col_b:
        s = summary_b
        desc_b = config_b.get("description") or ""
        st.info(
            f"**B — candidate:** `{s.get('run_name', '?')}`  \n"
            f"{s.get('successful_cases', '?')} cases  |  p50: {s.get('p50_latency_ms', '—')} ms"
            + (f"  \n📝 {desc_b}" if desc_b else "")
        )

    # ── Config diff ───────────────────────────────────────────────────────
    if config_a and config_b:
        def _flat(cfg: dict) -> dict:
            """Flatten retrieval/synthesis/query_pipeline into a single-level dict."""
            out = {}
            for section in ("retrieval", "synthesis", "query_pipeline"):
                for k, v in cfg.get(section, {}).items():
                    out[k] = v
            out["workers"] = cfg.get("eval", {}).get("workers")
            out["git_commit"] = cfg.get("git_commit")
            return out

        flat_a, flat_b = _flat(config_a), _flat(config_b)
        all_keys = sorted(set(flat_a) | set(flat_b))
        diff_rows = [
            {"Setting": k, "A": flat_a.get(k, "—"), "B": flat_b.get(k, "—")}
            for k in all_keys
            if flat_a.get(k) != flat_b.get(k)
        ]
        if diff_rows:
            with st.expander("⚙️ Config diff (changed settings only)", expanded=True):
                st.dataframe(pd.DataFrame(diff_rows), use_container_width=True, hide_index=True)
        else:
            st.caption("⚙️ Configs are identical (or config files not available for both runs).")

    # ── Regression / improvement banner ──────────────────────────────────
    if cmp["regressions"]:
        items = ", ".join(
            f"{METRIC_LABELS[r['metric']]} {(r['delta'] or 0) * 100:+.1f}pp"
            for r in cmp["regressions"]
        )
        st.error(f"⚠️ Regressions detected: {items}")
    if cmp["improvements"]:
        items = ", ".join(
            f"{METRIC_LABELS[r['metric']]} {(r['delta'] or 0) * 100:+.1f}pp"
            for r in cmp["improvements"]
        )
        st.success(f"✅ Improvements: {items}")
    if not cmp["regressions"] and not cmp["improvements"]:
        st.info("No significant changes detected (±1 pp threshold).")

    st.markdown("---")

    # ── Overall scorecard ────────────────────────────────────────────────
    st.subheader("Overall Scorecard (B vs A)")
    ov   = cmp["overall"]
    cols = st.columns(len(METRIC_KEYS))
    for col, m in zip(cols, METRIC_KEYS):
        va    = ov[m]["a"]
        vb    = ov[m]["b"]
        delta = ov[m]["delta"]
        as_pct = m != "mrr"
        val_str   = f"{vb * 100:.1f}%" if (vb is not None and as_pct) else (f"{vb:.4f}" if vb is not None else "—")
        delta_str = None
        if delta is not None:
            sign = "+" if delta >= 0 else ""
            delta_str = f"{sign}{delta * 100:.1f}pp" if as_pct else f"{sign}{delta:.4f}"
        col.metric(METRIC_LABELS[m], val_str, delta_str)

    # ── Latency comparison ───────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Latency")
    lat      = cmp.get("latency", {})
    lat_cols = st.columns(3)
    for col, (lbl, ka, kb) in zip(lat_cols, [
        ("P50 (ms)",  "p50_a",  "p50_b"),
        ("P90 (ms)",  "p90_a",  "p90_b"),
        ("Mean (ms)", "mean_a", "mean_b"),
    ]):
        va_l = lat.get(ka)
        vb_l = lat.get(kb)
        delta_lat = (vb_l - va_l) if (va_l is not None and vb_l is not None) else None
        delta_str = (f"{'+' if (delta_lat or 0) >= 0 else ''}{delta_lat} ms"
                     if delta_lat is not None else None)
        col.metric(lbl, f"{vb_l} ms" if vb_l is not None else "—", delta_str)

    st.markdown("---")

    # ── Breakdown comparison ─────────────────────────────────────────────
    st.subheader("Breakdowns")

    metric_pick = st.selectbox(
        "Metric to compare",
        METRIC_KEYS,
        format_func=lambda m: METRIC_LABELS[m],
        index=METRIC_KEYS.index("recall_at_3"),
        key="cmp_metric_pick",
    )

    tab_cat, tab_diff, tab_qtype, tab_size = st.tabs(
        ["By Category", "By Difficulty", "By Question Type", "Single vs Multi-Article"]
    )
    with tab_cat:
        _cmp_breakdown_table(cmp["by_category"], metric_pick, "Category")
    with tab_diff:
        ordered = {k: cmp["by_difficulty"][k] for k in ("easy", "medium", "hard")
                   if k in cmp["by_difficulty"]}
        _cmp_breakdown_table(ordered, metric_pick, "Difficulty")
    with tab_qtype:
        _cmp_breakdown_table(cmp["by_question_type"], metric_pick, "Question Type")
    with tab_size:
        _cmp_breakdown_table(cmp["by_article_count"], metric_pick, "Article Count")

    # ── Export ───────────────────────────────────────────────────────────
    st.markdown("---")
    import json as _json
    cmp_json = _json.dumps(cmp, indent=2, ensure_ascii=False)
    st.download_button(
        label="⬇️ Download comparison JSON",
        data=cmp_json,
        file_name=f"compare_{run_a_name}_vs_{run_b_name}.json",
        mime="application/json",
    )


# ---------------------------------------------------------------------------
# ── App entry point ─────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="CRR RAG Eval Dashboard",
        page_icon="📊",
        layout="wide",
    )
    st.title("📊 CRR RAG Eval Dashboard")

    runs = _discover_runs()

    page = st.sidebar.radio(
        "Page",
        ["📝 Dataset Review", "📈 Eval Results", "🔀 Compare Runs"],
        key="main_page",
    )

    if page == "📝 Dataset Review":
        page_dataset_review()
    elif page == "📈 Eval Results":
        if runs:
            page_eval_results(runs)
        else:
            page_eval_results_empty()
    else:
        page_compare_runs(runs)


if __name__ == "__main__":
    main()
