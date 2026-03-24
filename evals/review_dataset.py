"""
evals/review_dataset.py — Automated quality review of the golden dataset.

For each case runs:
  1. Regex    — citation_type correctness (no API/LLM needed)
  2. LLM      — expected_articles completeness, numerical accuracy,
                reference answer completeness, difficulty calibration
               (needs article text from GET /api/article/{id} + gpt-4o)

Outputs:
  evals/cases/review_results.jsonl  — per-case flags (machine-readable)
  evals/cases/review_report.md      — human-readable summary

Usage:
    python -m evals.review_dataset                    # full run (needs API + OpenAI key)
    python -m evals.review_dataset --limit 20         # smoke test
    python -m evals.review_dataset --no-llm           # regex + coverage stats only
    python -m evals.review_dataset --api-url http://localhost:8080
    python -m evals.review_dataset --case-ids case_034 case_082
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

import requests

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

DATASET_PATH   = Path("evals/cases/golden_dataset.jsonl")
OUTPUT_JSONL   = Path("evals/cases/review_results.jsonl")
OUTPUT_REPORT  = Path("evals/cases/review_report.md")
DEFAULT_API    = "http://localhost:8080"

# Truncate article text fed to LLM — avoids token-limit issues on long articles
# (e.g. Article 4 definitions). Covers ~2 000 tokens comfortably.
_MAX_ARTICLE_CHARS = 8_000

# ---------------------------------------------------------------------------
# Regex check: citation_type
# ---------------------------------------------------------------------------

# Matches cross-regulation references like:
#   "Articles 10 to 14 of Regulation (EU) No 1093/2010"
#   "Article 17 of Directive 2013/36/EU"
# These must be stripped before checking for CRR article citations.
_CROSS_REG_RE = re.compile(
    r"\bArticles?\s+[\d\w][\w\s,/–-]*?\bof\s+(?:Regulation|Directive)\b[^.;]*",
    re.IGNORECASE,
)

# Matches a bare CRR article reference after cross-regulation refs have been removed.
_ARTICLE_RE = re.compile(r"\bArticle\s+\d+[a-z]?\b", re.IGNORECASE)


def _strip_cross_reg_refs(text: str) -> str:
    """Remove references to articles of other regulations/directives."""
    return _CROSS_REG_RE.sub("", text)


def check_citation_type(case: dict) -> Optional[dict]:
    """
    citation_type must be 'article_cited' when the question explicitly names
    a CRR article number, and 'open_ended' otherwise.

    References to articles of *other* regulations (e.g. Regulation (EU) No
    1093/2010, CRD IV) are stripped before the check — those article numbers
    are irrelevant to the CRR citation_type classification.
    """
    question  = case.get("question", "")
    declared  = case.get("citation_type", "")
    crr_only  = _strip_cross_reg_refs(question)
    has_ref   = bool(_ARTICLE_RE.search(crr_only))

    if has_ref and declared == "open_ended":
        return {
            "check": "citation_type",
            "severity": "medium",
            "message": "Question mentions an article number but citation_type is 'open_ended'.",
        }
    if not has_ref and declared == "article_cited":
        return {
            "check": "citation_type",
            "severity": "medium",
            "message": "citation_type is 'article_cited' but no article number found in question.",
        }
    return None


# ---------------------------------------------------------------------------
# Article text fetching
# ---------------------------------------------------------------------------

def _fetch_article(api_url: str, article_id: str) -> Optional[str]:
    """Return 'Article N — Title\\n\\n<text>' or None on failure / 404."""
    url = f"{api_url.rstrip('/')}/api/article/{article_id}"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        data  = r.json()
        title = data.get("article_title", "")
        text  = data.get("text", "")
        header = f"Article {article_id}" + (f" — {title}" if title else "")
        return f"{header}\n\n{text}"
    except Exception as exc:
        logger.warning("Could not fetch Article %s: %s", article_id, exc)
        return None


def _fetch_articles_for_case(api_url: str, case: dict) -> dict[str, str]:
    """Return {article_id: truncated_text} for all expected_articles."""
    out: dict[str, str] = {}
    for art in case.get("expected_articles", []):
        text = _fetch_article(api_url, art)
        if text:
            out[art] = text[:_MAX_ARTICLE_CHARS]
    return out


# ---------------------------------------------------------------------------
# LLM review
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a precise reviewer of evaluation datasets for a CRR (EU Capital Requirements Regulation) RAG system.

You receive:
- A question
- A reference answer (written by an LLM, may contain errors)
- The actual article text(s) from the CRR
- Current metadata

Identify genuine quality issues only. Do NOT flag stylistic differences, paraphrasing, or minor phrasing variations.

Return ONLY a JSON object with exactly this structure:

{
  "expected_articles": {
    "verdict": "ok" | "incomplete",
    "missing": ["article numbers that are load-bearing to the answer but absent from expected_articles"],
    "note": "one-line explanation, or empty string"
  },
  "numerical_accuracy": {
    "verdict": "ok" | "flag",
    "issues": [
      {
        "value": "the specific value as written in the reference answer",
        "note": "what the article text actually says, or that the value is absent"
      }
    ]
  },
  "completeness": {
    "verdict": "ok" | "flag",
    "gaps": [
      "description of a material condition, threshold, or sub-rule present in the article text but absent from the reference answer"
    ]
  },
  "difficulty": {
    "verdict": "ok" | "flag",
    "suggested": "easy | medium | hard",
    "note": "one-line explanation, or empty string"
  }
}

Rules:
- expected_articles: only covers articles of the CRR itself (Regulation (EU) No 575/2013). \
Do NOT flag articles from other regulations or directives (e.g. Regulation (EU) No 1093/2010, \
CRD IV / Directive 2013/36/EU, EBA founding regulation, etc.) — those are external references \
and must never appear in expected_articles. Only flag CRR article numbers that are load-bearing \
to the answer but genuinely absent from expected_articles.
- numerical_accuracy: only flag when a specific number, percentage, threshold, date, or count in \
the reference answer is wrong or absent in the article text. Ignore correct values. \
IMPORTANT: the CRR text uses "x %" (with a space before the percent sign); treat "x%" and "x %" \
as identical — do NOT flag a value solely because of this spacing difference.
- completeness: only flag genuinely material omissions — missing required conditions, dropped \
sub-rules that change the answer. Ignore minor elaboration.
- difficulty: flag only when the mismatch is obvious (e.g. labelled 'easy' but requires 3+ \
cross-references, or labelled 'hard' but is a single sentence lookup).
- Return empty lists / "ok" verdicts when no issue is found. Never invent issues.
"""

_USER_TEMPLATE = """\
QUESTION:
{question}

REFERENCE ANSWER:
{reference_answer}

ARTICLE TEXT(S):
{article_texts}

METADATA:
- expected_articles : {expected_articles}
- difficulty        : {difficulty}
- question_type     : {question_type}
- citation_type     : {citation_type}
"""


# Default model per provider — override with --model
_DEFAULT_MODELS = {
    "openai":  "gpt-4o",
    "gemini":  "gemini-2.0-flash",
}

# Gemini exposes an OpenAI-compatible endpoint, so we reuse the OpenAI client.
_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def _build_client(provider: str) -> tuple:
    """
    Return (OpenAI client, resolved api_key) for the given provider.
    Raises RuntimeError if the required env var is missing.
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("openai package not installed — run: pip install openai")

    if provider == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set in environment / .env")
        return OpenAI(api_key=api_key, base_url=_GEMINI_BASE_URL), api_key

    # default: openai
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment / .env")
    return OpenAI(api_key=api_key), api_key


def _run_llm_review(
    case: dict,
    article_texts: dict[str, str],
    provider: str,
    model: str,
) -> Optional[dict]:
    """Call the LLM reviewer. Returns parsed dict or None on any failure."""
    try:
        client, _ = _build_client(provider)
    except RuntimeError as exc:
        logger.error("%s", exc)
        return None

    combined = "\n\n---\n\n".join(
        f"[Article {art_id}]\n{text}" for art_id, text in article_texts.items()
    )

    user_msg = _USER_TEMPLATE.format(
        question=case.get("question", ""),
        reference_answer=case.get("reference_answer", ""),
        article_texts=combined,
        expected_articles=json.dumps(case.get("expected_articles", [])),
        difficulty=case.get("difficulty", ""),
        question_type=case.get("question_type", ""),
        citation_type=case.get("citation_type", ""),
    )

    try:
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=1024,
            timeout=60,
        )
    except Exception as exc:
        logger.error("LLM call failed for %s: %s", case.get("id"), exc)
        return None

    raw = response.choices[0].message.content or ""
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("JSON parse error for %s: %s | raw=%r", case.get("id"), exc, raw[:200])
        return None


# ---------------------------------------------------------------------------
# Per-case review
# ---------------------------------------------------------------------------

def review_case(
    case: dict,
    api_url: str,
    run_llm: bool,
    provider: str,
    model: str,
) -> dict:
    case_id      = case.get("id", "unknown")
    flags: list[dict] = []

    # --- Check 1: citation_type (pure regex) ---
    flag = check_citation_type(case)
    if flag:
        flags.append(flag)

    # --- Checks 2-5: LLM (needs article text from API) ---
    article_texts: dict[str, str] = {}
    llm_result: Optional[dict] = None

    if run_llm:
        article_texts = _fetch_articles_for_case(api_url, case)

        if not article_texts:
            flags.append({
                "check": "article_fetch",
                "severity": "info",
                "message": (
                    f"Article text unavailable for {case.get('expected_articles')} "
                    "— LLM checks skipped."
                ),
            })
        else:
            llm_result = _run_llm_review(case, article_texts, provider=provider, model=model)
            if llm_result:
                # expected_articles completeness
                ea = llm_result.get("expected_articles", {})
                if ea.get("verdict") == "incomplete" and ea.get("missing"):
                    flags.append({
                        "check": "expected_articles",
                        "severity": "high",
                        "message": (
                            f"Possibly missing from expected_articles: {ea['missing']}."
                            + (f" {ea['note']}" if ea.get("note") else "")
                        ),
                    })

                # numerical accuracy
                num = llm_result.get("numerical_accuracy", {})
                if num.get("verdict") == "flag":
                    for issue in num.get("issues", []):
                        flags.append({
                            "check": "numerical_accuracy",
                            "severity": "high",
                            "message": f"'{issue.get('value')}' — {issue.get('note')}",
                        })

                # completeness
                comp = llm_result.get("completeness", {})
                if comp.get("verdict") == "flag":
                    for gap in comp.get("gaps", []):
                        flags.append({
                            "check": "completeness",
                            "severity": "medium",
                            "message": gap,
                        })

                # difficulty calibration
                diff = llm_result.get("difficulty", {})
                if diff.get("verdict") == "flag":
                    flags.append({
                        "check": "difficulty",
                        "severity": "low",
                        "message": (
                            f"Suggested difficulty: {diff.get('suggested')}."
                            + (f" {diff['note']}" if diff.get("note") else "")
                        ),
                    })

    needs_review = any(f["severity"] in ("high", "medium") for f in flags)

    return {
        "id":                   case_id,
        "question_snippet":     case.get("question", "")[:120],
        "expected_articles":    case.get("expected_articles", []),
        "category":             case.get("category"),
        "difficulty":           case.get("difficulty"),
        "question_type":        case.get("question_type"),
        "citation_type":        case.get("citation_type"),
        "flags":                flags,
        "needs_review":         needs_review,
        "article_text_fetched": list(article_texts.keys()),
        "llm_checked":          llm_result is not None,
    }


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def build_report(results: list[dict]) -> str:
    total       = len(results)
    flagged     = [r for r in results if r["needs_review"]]
    high_cases  = [r for r in results if any(f["severity"] == "high"   for f in r["flags"])]
    med_cases   = [r for r in results if any(f["severity"] == "medium" for f in r["flags"])]
    low_cases   = [r for r in results if any(f["severity"] == "low"    for f in r["flags"])]
    llm_checked = sum(1 for r in results if r["llm_checked"])

    lines = [
        "# Golden Dataset — Automated Review Report",
        "",
        f"| | |",
        f"|---|---|",
        f"| Total cases | {total} |",
        f"| LLM-checked (article text available) | {llm_checked} |",
        f"| **Cases needing review** | **{len(flagged)}** |",
        f"| — High severity | {len(high_cases)} |",
        f"| — Medium severity | {len(med_cases)} |",
        f"| — Low severity (FYI) | {len(low_cases)} |",
        "",
    ]

    # Findings grouped by check type
    check_order = [
        ("expected_articles",  "Missing from `expected_articles`"),
        ("numerical_accuracy", "Numerical accuracy issues"),
        ("completeness",       "Reference answer completeness gaps"),
        ("citation_type",      "`citation_type` mismatch"),
        ("difficulty",         "Difficulty calibration"),
        ("article_fetch",      "Article text not available (LLM checks skipped)"),
    ]

    for check_key, check_label in check_order:
        affected = [r for r in results if any(f["check"] == check_key for f in r["flags"])]
        if not affected:
            continue

        lines += [f"## {check_label} ({len(affected)} cases)", ""]
        for r in affected:
            for flag in r["flags"]:
                if flag["check"] != check_key:
                    continue
                sev  = flag["severity"].upper()
                arts = ", ".join(r["expected_articles"]) or "—"
                lines.append(f"- **{r['id']}** (Art. {arts}) `[{sev}]`")
                lines.append(f"  {flag['message']}")
        lines.append("")

    # Coverage distribution
    lines += ["## Coverage distribution", ""]

    cat_counts  = Counter(r["category"]      for r in results)
    diff_counts = Counter(r["difficulty"]    for r in results)
    qt_counts   = Counter(r["question_type"] for r in results)
    ct_counts   = Counter(r["citation_type"] for r in results)

    lines += ["**By category**", ""]
    lines += ["| Category | n |", "|---|---|"]
    for cat, n in sorted(cat_counts.items(), key=lambda x: -x[1]):
        lines.append(f"| {cat} | {n} |")
    lines.append("")

    lines += ["**By difficulty**", ""]
    lines += ["| Difficulty | n |", "|---|---|"]
    for d in ("easy", "medium", "hard"):
        lines.append(f"| {d} | {diff_counts.get(d, 0)} |")
    lines.append("")

    lines += ["**By question type**", ""]
    lines += ["| Question type | n |", "|---|---|"]
    for qt, n in qt_counts.most_common():
        lines.append(f"| {qt} | {n} |")
    lines.append("")

    lines += ["**By citation type**", ""]
    lines += ["| Citation type | n |", "|---|---|"]
    for ct, n in ct_counts.most_common():
        lines.append(f"| {ct} | {n} |")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Automated quality review of the CRR golden dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset",       type=Path, default=DATASET_PATH)
    p.add_argument("--output-jsonl",  type=Path, default=OUTPUT_JSONL)
    p.add_argument("--output-report", type=Path, default=OUTPUT_REPORT)
    p.add_argument("--api-url",       default=DEFAULT_API,
                   help="CRR API URL for fetching article text (default: %(default)s)")
    p.add_argument("--limit",         type=int, default=None,
                   help="Review only first N cases")
    p.add_argument("--case-ids",      nargs="+", default=None,
                   help="Review only specific case IDs")
    p.add_argument("--no-llm",        action="store_true",
                   help="Run regex checks only — no API or LLM calls")
    p.add_argument("--provider",      default="openai", choices=["openai", "gemini"],
                   help="LLM provider for review checks (default: %(default)s). "
                        "Use 'gemini' to avoid correlated blind-spots with the "
                        "OpenAI-generated dataset. Requires GEMINI_API_KEY in .env.")
    p.add_argument("--model",         default=None,
                   help="Model override (default: gpt-4o for openai, "
                        "gemini-2.0-flash for gemini)")
    return p.parse_args(argv)


def _load_dataset(path: Path) -> list[dict]:
    cases: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    cases.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return cases


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    cases = _load_dataset(args.dataset)
    logger.info("Loaded %d cases from %s", len(cases), args.dataset)

    if args.case_ids:
        id_set = set(args.case_ids)
        cases  = [c for c in cases if c.get("id") in id_set]
        logger.info("Filtered to %d cases by --case-ids.", len(cases))
    if args.limit:
        cases = cases[: args.limit]
        logger.info("Limited to first %d cases.", len(cases))

    provider = args.provider
    model    = args.model or _DEFAULT_MODELS.get(provider, "gpt-4o")

    logger.info(
        "Reviewing %d cases  [llm=%s, provider=%s, model=%s]",
        len(cases), not args.no_llm,
        provider if not args.no_llm else "—",
        model    if not args.no_llm else "—",
    )

    results: list[dict] = []
    for i, case in enumerate(cases, 1):
        logger.info("[%d/%d] %s", i, len(cases), case.get("id"))
        result = review_case(
            case,
            api_url=args.api_url,
            run_llm=not args.no_llm,
            provider=provider,
            model=model,
        )
        results.append(result)
        if not args.no_llm:
            time.sleep(0.3)  # light throttle — avoids 429s on burst

    # Write JSONL
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("Per-case results → %s", args.output_jsonl)

    # Write markdown report
    report = build_report(results)
    args.output_report.write_text(report, encoding="utf-8")
    logger.info("Report → %s", args.output_report)

    # Console summary
    flagged    = [r for r in results if r["needs_review"]]
    high_flags = [r for r in results if any(f["severity"] == "high" for f in r["flags"])]
    logger.info("=" * 55)
    logger.info("Reviewed %d cases — %d flagged (%d high, %d medium)",
                len(results), len(flagged),
                len(high_flags),
                len([r for r in results if any(f["severity"] == "medium" for f in r["flags"])]))
    for r in high_flags:
        first_high = next(f for f in r["flags"] if f["severity"] == "high")
        logger.info("  [HIGH] %s — %s", r["id"], first_high["message"][:90])
    logger.info("=" * 55)


if __name__ == "__main__":
    main(sys.argv[1:])
