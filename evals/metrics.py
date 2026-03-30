"""
Retrieval quality metrics for the CRR RAG eval pipeline.

All functions operate on normalised article-number strings.
"""
from __future__ import annotations

import re

# Matches sub-article numbers like "429a", "429b", "132c" — digits followed
# by one or more lowercase letters.  Used by article_family() below.
_SUB_ARTICLE_RE = re.compile(r"^(\d+)[a-z]+$")


def normalise_article(raw: str) -> str:
    """Normalise an article number for comparison.

    '429A' -> '429a'
    ' 26 ' -> '26'
    '26(1)' -> '26'   (strip parenthetical paragraph refs)
    """
    s = str(raw).strip().lower()
    s = re.sub(r"\(.*", "", s).strip()
    return s


def article_family(article: str) -> str:
    """Return the article family key for sub-article-aware comparison.

    Sub-articles normalise to their parent:
        "429a" -> "429",  "429b" -> "429",  "132c" -> "132"
    Plain numeric articles and annexes are returned unchanged:
        "429"  -> "429",  "92"   -> "92"

    Used by hit_at_k_family() so that retrieving 429 when the gold is 429b
    (or vice versa) counts as a family hit.
    """
    m = _SUB_ARTICLE_RE.match(article)
    return m.group(1) if m else article


def hit_at_k_family(expected: list[str], retrieved: list[str], k: int) -> int:
    """1 if any top-k retrieved article is in the same family as any expected article.

    Two articles are in the same family when they share a parent (e.g. 429 and
    429b both normalise to "429").  This counts within-family swaps (429 retrieved
    when 429b expected) as hits so stochastic reranker ties inside a sub-article
    cluster do not produce artificial misses.
    """
    expected_families = {article_family(a) for a in expected}
    return int(any(article_family(a) in expected_families for a in retrieved[:k]))


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
        "hit_at_1":        hit_at_k(norm_exp, norm_ret, 1),
        "hit_at_1_family": hit_at_k_family(norm_exp, norm_ret, 1),
        "recall_at_1":     recall_at_k(norm_exp, norm_ret, 1),
        "recall_at_3":     recall_at_k(norm_exp, norm_ret, 3),
        "recall_at_5":     recall_at_k(norm_exp, norm_ret, 5),
        "mrr":             mrr(norm_exp, norm_ret),
        "precision_at_3":  precision_at_k(norm_exp, norm_ret, 3),
        "precision_at_5":  precision_at_k(norm_exp, norm_ret, 5),
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
