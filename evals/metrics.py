"""
Retrieval quality metrics for the CRR RAG eval pipeline.

All functions operate on normalised article-number strings.
"""
from __future__ import annotations

import re


def normalise_article(raw: str) -> str:
    """Normalise an article number for comparison.

    '429A' -> '429a'
    ' 26 ' -> '26'
    '26(1)' -> '26'   (strip parenthetical paragraph refs)
    """
    s = str(raw).strip().lower()
    s = re.sub(r"\(.*", "", s).strip()
    return s


def deduplicate_ranked(articles: list[str]) -> list[str]:
    """Return ordered list with first occurrence of each article kept."""
    seen: set[str] = set()
    result: list[str] = []
    for a in articles:
        if a not in seen:
            seen.add(a)
            result.append(a)
    return result


def hit_at_k(expected: list[str], retrieved: list[str], k: int) -> int:
    """1 if any expected article appears in top-k retrieved, else 0."""
    expected_set = set(expected)
    return int(any(a in expected_set for a in retrieved[:k]))


def recall_at_k(expected: list[str], retrieved: list[str], k: int) -> float:
    """Fraction of expected articles found in top-k retrieved."""
    if not expected:
        return 0.0
    expected_set = set(expected)
    found = sum(1 for a in retrieved[:k] if a in expected_set)
    return found / len(expected_set)


def precision_at_k(expected: list[str], retrieved: list[str], k: int) -> float:
    """Fraction of top-k retrieved articles that are expected."""
    if k == 0:
        return 0.0
    expected_set = set(expected)
    hits = sum(1 for a in retrieved[:k] if a in expected_set)
    return hits / k


def mrr(expected: list[str], retrieved: list[str]) -> float:
    """Mean Reciprocal Rank: 1/rank of first matching article (1-indexed).

    Returns 0.0 if no expected article appears in retrieved.
    """
    expected_set = set(expected)
    for rank, a in enumerate(retrieved, start=1):
        if a in expected_set:
            return 1.0 / rank
    return 0.0


def compute_all(expected: list[str], retrieved: list[str]) -> dict:
    """Compute the full suite of retrieval metrics for one case."""
    norm_exp = [normalise_article(a) for a in expected]
    norm_ret = deduplicate_ranked([normalise_article(a) for a in retrieved])
    return {
        "hit_at_1":      hit_at_k(norm_exp, norm_ret, 1),
        "recall_at_1":   recall_at_k(norm_exp, norm_ret, 1),
        "recall_at_3":   recall_at_k(norm_exp, norm_ret, 3),
        "recall_at_5":   recall_at_k(norm_exp, norm_ret, 5),
        "mrr":           mrr(norm_exp, norm_ret),
        "precision_at_3": precision_at_k(norm_exp, norm_ret, 3),
        "precision_at_5": precision_at_k(norm_exp, norm_ret, 5),
    }


def compute_all_with_expanded(
    expected: list[str],
    retrieved: list[str],
    expanded: list[str],
) -> dict:
    """Metrics treating expanded articles as additional retrieved results (appended after ranked).

    Expanded articles are lower-priority than primary retrieved results but still count
    toward recall. This measures the combined effectiveness of retrieval + cross-ref expansion
    and exposes the true gap vs. the artificially deflated Recall numbers when expansion is ignored.
    """
    norm_exp = [normalise_article(a) for a in expected]
    combined = deduplicate_ranked(
        [normalise_article(a) for a in retrieved] +
        [normalise_article(a) for a in expanded]
    )
    return {
        "hit_at_1_with_expanded":    hit_at_k(norm_exp, combined, 1),
        "recall_at_1_with_expanded": recall_at_k(norm_exp, combined, 1),
        "recall_at_3_with_expanded": recall_at_k(norm_exp, combined, 3),
        "recall_at_5_with_expanded": recall_at_k(norm_exp, combined, 5),
        "mrr_with_expanded":         mrr(norm_exp, combined),
    }
