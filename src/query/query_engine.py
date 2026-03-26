"""
QueryEngine: Qdrant native hybrid retrieval (dense + sparse via BGE-M3)
with cross-reference expansion and GPT-4o synthesis.
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import openai  # raw sync client — distinct from llama_index.llms.openai.OpenAI

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core.vector_stores.types import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQueryMode,
)
from llama_index.llms.openai import OpenAI

from src.indexing.bge_m3_sparse import BGEm3Embedding
from src.indexing.index_builder import HierarchicalIndexer
from src.indexing.vector_store import VectorStore

logger = logging.getLogger(__name__)

RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "12"))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "6"))
SIMILARITY_CUTOFF = 0.3
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
# Blend weight for retrieval score vs reranker score (0=pure reranker, 1=pure retrieval).
# 0.3 means 30% retrieval signal + 70% reranker signal.
RERANK_BLEND_ALPHA: float = float(os.getenv("RERANK_BLEND_ALPHA", "0.3"))
LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
# Approximate token budget for synthesis context (~4 chars/token, leave 4k headroom).
# Prevents rate-limit errors on large articles (e.g. Article 4 definitions).
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", str(100_000)))
RETRIEVAL_ALPHA: float = float(os.getenv("RETRIEVAL_ALPHA", "0.5"))
TITLE_BOOST_WEIGHT: float = float(os.getenv("TITLE_BOOST_WEIGHT", "0.15"))
# Maximum score gap between rank-1 and rank-2 nodes for the adjacent-article tiebreaker to fire.
# 0.0 = disabled. Recommended starting value: 0.05.
ADJACENT_TIEBREAK_DELTA: float = float(os.getenv("ADJACENT_TIEBREAK_DELTA", "0.0"))
# Max paragraph windows to score per article in ParagraphWindowReranker (latency guard).
PARAGRAPH_WINDOW_MAX_WINDOWS: int = int(os.getenv("PARAGRAPH_WINDOW_MAX_WINDOWS", "4"))
# When true, the index contains PARAGRAPH-level chunks in addition to ARTICLE docs.
# Retrieval filters to PARAGRAPH chunks; context assembly fetches the parent ARTICLE docs.
USE_PARAGRAPH_CHUNKING: bool = os.getenv("USE_PARAGRAPH_CHUNKING", "false").lower() == "true"
# When true, both ARTICLE and PARAGRAPH chunks compete in retrieval (no chunk_type filter).
# An ArticleDeduplicatorPostprocessor keeps only the best-scored chunk per article, and
# context assembly upgrades any winning PARAGRAPH chunk to its parent ARTICLE text.
USE_MIXED_CHUNKING: bool = os.getenv("USE_MIXED_CHUNKING", "false").lower() == "true"
# When true, cross-reference expansion uses the pre-computed ArticleGraph for BFS traversal
# with ref-type prioritisation instead of flat alphabetical CSV expansion.
USE_ARTICLE_GRAPH: bool = os.getenv("USE_ARTICLE_GRAPH", "true").lower() == "true"
_HISTORY_MAX_TURNS = 5
# When EVAL_MODE=true all internal LLM calls use temperature=0 for reproducibility.
EVAL_MODE: bool = os.getenv("EVAL_MODE", "false").lower() == "true"
_EVAL_KWARGS: dict = {"temperature": 0} if EVAL_MODE else {}


_reranker_lock = threading.Lock()
"""Serialises cross-encoder predict() calls across concurrent workers.

CrossEncoder (sentence-transformers) is NOT thread-safe: concurrent predict()
calls on the same model instance can corrupt internal PyTorch state, causing
non-Python native crashes that bypass except Exception handlers.  This lock
serialises ALL reranker predictions without requiring a separate model instance
per thread, keeping memory usage constant regardless of worker count.
"""


class BlendedReranker(BaseNodePostprocessor):
    """Cross-encoder reranker with score blending.

    Instead of letting the reranker fully override retrieval scores, blends:
        final = alpha * retrieval_score + (1 - alpha) * normalised_reranker_score

    This preserves strong retrieval signals (preventing rank flips on adjacent
    articles) while still benefiting from cross-encoder cross-attention.
    Reranker logits are min-max normalised within each batch before blending.
    """

    def __init__(self, model: str, top_n: int, alpha: float) -> None:
        super().__init__()
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(model)
        self._top_n = top_n
        self._alpha = alpha  # weight for retrieval score

    @classmethod
    def class_name(cls) -> str:
        return "BlendedReranker"

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        if not nodes or query_bundle is None:
            return nodes

        query = query_bundle.query_str
        pairs = [(query, n.node.get_content()) for n in nodes]
        with _reranker_lock:
            reranker_scores = self._model.predict(pairs)

        # Normalise reranker logits to [0, 1]
        lo, hi = min(reranker_scores), max(reranker_scores)
        span = hi - lo if hi > lo else 1.0
        norm_scores = [(s - lo) / span for s in reranker_scores]

        for node, norm_score in zip(nodes, norm_scores):
            retrieval_score = node.score or 0.0
            node.score = self._alpha * retrieval_score + (1 - self._alpha) * norm_score

        return sorted(nodes, key=lambda n: n.score or 0.0, reverse=True)[: self._top_n]


class ArticleTitleBoostPostprocessor(BaseNodePostprocessor):
    """Post-rerank score boost based on article title / query token overlap.

    Runs *after* the cross-encoder reranker (which already scored all nodes).
    Applies a small additive bonus proportional to how many title tokens appear
    in the query, then re-sorts and truncates to top_n.

    This rescues open-ended queries ("What is TREA?", "pledged government bonds")
    where embedding similarity alone can't bridge the vocabulary gap but the
    article title ("Total Risk Exposure Amount") is a strong discriminative signal.
    """

    _STOPWORDS: frozenset = frozenset({
        "requirements", "requirement", "article", "articles", "institution",
        "institutions", "regulation", "regulations", "general", "specific",
        "provisions", "provision", "the", "of", "for", "and", "or", "in",
        "to", "a", "an", "on", "by", "with", "that", "this", "is", "are",
        "be", "been", "have", "has", "not", "which", "from", "at", "as",
        "its", "their", "it", "shall", "may", "must",
    })

    def __init__(self, boost_weight: float, top_n: int) -> None:
        super().__init__()
        self._boost_weight = boost_weight
        self._top_n = top_n

    @classmethod
    def class_name(cls) -> str:
        return "ArticleTitleBoostPostprocessor"

    def _tokenize(self, text: str) -> set[str]:
        tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
        return tokens - self._STOPWORDS

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        if not nodes or query_bundle is None or self._boost_weight == 0:
            return nodes[: self._top_n]

        query_tokens = self._tokenize(query_bundle.query_str)
        if not query_tokens:
            return nodes[: self._top_n]

        for node in nodes:
            title = node.node.metadata.get("article_title", "") or ""
            if not title:
                continue
            title_tokens = self._tokenize(title)
            if not title_tokens:
                continue
            matched = query_tokens & title_tokens
            if not matched:
                continue
            # min() denominator: titles are short (2-5 words), so 2/3 title words = 0.67
            # instead of diluting against a 15-token query (Jaccard would give ~0.13)
            match_ratio = len(matched) / min(len(query_tokens), len(title_tokens))
            boost = self._boost_weight * match_ratio
            node.score = (node.score or 0.0) + boost

        return sorted(nodes, key=lambda n: n.score or 0.0, reverse=True)[: self._top_n]


class AdjacentArticleTiebreakerPostprocessor(BaseNodePostprocessor):
    """Post-rerank tiebreaker for adjacent-article rank-flip failures.

    When the top-2 nodes are "adjacent" articles (same numeric base, e.g. 429/429a,
    or consecutive integers, e.g. 114/115) and their score gap is below `delta`,
    prefer the article whose title has more query-token overlap.

    Fires only on near-ties between neighbours — safe to leave enabled in production
    since it is a no-op for all other cases.

    Enabled by setting ADJACENT_TIEBREAK_DELTA > 0 (e.g. 0.05).
    """

    _STOPWORDS: frozenset = frozenset({
        "requirements", "requirement", "article", "articles", "institution",
        "institutions", "regulation", "regulations", "general", "specific",
        "provisions", "provision", "the", "of", "for", "and", "or", "in",
        "to", "a", "an", "on", "by", "with", "that", "this", "is", "are",
        "be", "been", "have", "has", "not", "which", "from", "at", "as",
        "its", "their", "it", "shall", "may", "must",
    })

    def __init__(self, delta: float) -> None:
        super().__init__()
        self._delta = delta

    @classmethod
    def class_name(cls) -> str:
        return "AdjacentArticleTiebreakerPostprocessor"

    @staticmethod
    def _article_base(article_str: str) -> int:
        """Extract the leading integer from an article string (e.g. '429a' → 429)."""
        m = re.match(r"(\d+)", article_str.strip())
        return int(m.group(1)) if m else -1

    def _are_adjacent(self, art_a: str, art_b: str) -> bool:
        """True if articles share the same numeric base or are consecutive integers."""
        if not art_a or not art_b:
            return False
        base_a = self._article_base(art_a)
        base_b = self._article_base(art_b)
        if base_a < 0 or base_b < 0:
            return False
        # Same base (e.g. 429 and 429a) or consecutive (e.g. 114 and 115)
        return base_a == base_b or abs(base_a - base_b) == 1

    def _title_overlap(self, title: str, query_tokens: set[str]) -> int:
        tokens = set(re.findall(r"[a-z0-9]+", title.lower())) - self._STOPWORDS
        return len(tokens & query_tokens)

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        if len(nodes) < 2 or query_bundle is None or self._delta <= 0:
            return nodes

        top, second = nodes[0], nodes[1]
        score_gap = (top.score or 0.0) - (second.score or 0.0)
        if score_gap > self._delta:
            return nodes

        art_top = top.node.metadata.get("article", "")
        art_second = second.node.metadata.get("article", "")
        if not self._are_adjacent(art_top, art_second):
            return nodes

        query_tokens = set(re.findall(r"[a-z0-9]+", query_bundle.query_str.lower())) - self._STOPWORDS
        title_top = top.node.metadata.get("article_title", "") or ""
        title_second = second.node.metadata.get("article_title", "") or ""
        overlap_top = self._title_overlap(title_top, query_tokens)
        overlap_second = self._title_overlap(title_second, query_tokens)

        if overlap_second > overlap_top:
            logger.debug(
                "AdjacentTiebreaker: swapped Art.%s (overlap=%d) above Art.%s (overlap=%d), gap=%.4f",
                art_second, overlap_second, art_top, overlap_top, score_gap,
            )
            nodes[0], nodes[1] = nodes[1], nodes[0]

        return nodes


class ArticleDeduplicatorPostprocessor(BaseNodePostprocessor):
    """Deduplicate retrieved nodes to at most one chunk per article.

    In mixed-chunking mode both ARTICLE and PARAGRAPH chunks compete in retrieval.
    Without deduplication, a long article with many paragraphs can flood the top-k
    and crowd out other articles entirely.

    Strategy per article:
    - If only ARTICLE or only PARAGRAPH chunks present, keep the highest-scored one.
    - If both are present and the ARTICLE chunk is within ``_PREFER_ARTICLE_MARGIN``
      of the best PARAGRAPH chunk, prefer the ARTICLE (avoids a synthesis fetch-parent
      round-trip and gives the reranker the full article text to score).
    - Otherwise keep whichever chunk scored highest.

    Context assembly is responsible for upgrading any surviving PARAGRAPH chunk to
    its parent ARTICLE text before synthesis.
    """

    # If the ARTICLE chunk score is at most this far below the best PARAGRAPH chunk
    # score, the ARTICLE chunk wins (prefer full context over marginal score gain).
    _PREFER_ARTICLE_MARGIN: float = 0.02

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def class_name(cls) -> str:
        return "ArticleDeduplicatorPostprocessor"

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        if not nodes:
            return nodes

        # Group by article number; annexes/non-article nodes keyed by node_id.
        best_article: dict[str, NodeWithScore] = {}   # ARTICLE chunk winner per key
        best_para: dict[str, NodeWithScore] = {}      # PARAGRAPH chunk winner per key

        for node in nodes:
            meta = node.node.metadata
            article = meta.get("article", "")
            key = article if article else node.node.node_id
            chunk_type = meta.get("chunk_type", "ARTICLE")

            if chunk_type == "PARAGRAPH":
                if key not in best_para or (node.score or 0.0) > (best_para[key].score or 0.0):
                    best_para[key] = node
            else:
                if key not in best_article or (node.score or 0.0) > (best_article[key].score or 0.0):
                    best_article[key] = node

        all_keys = set(best_article) | set(best_para)
        result: list[NodeWithScore] = []
        for key in all_keys:
            art_node = best_article.get(key)
            para_node = best_para.get(key)

            if art_node is None:
                result.append(para_node)  # type: ignore[arg-type]
            elif para_node is None:
                result.append(art_node)
            else:
                art_score = art_node.score or 0.0
                para_score = para_node.score or 0.0
                # Prefer ARTICLE if it's within the margin of the best PARAGRAPH
                if para_score - art_score <= self._PREFER_ARTICLE_MARGIN:
                    result.append(art_node)
                else:
                    result.append(para_node)

        return sorted(result, key=lambda n: n.score or 0.0, reverse=True)


class ParagraphWindowReranker(BaseNodePostprocessor):
    """Paragraph-aware cross-encoder reranker.

    Instead of scoring the full article text against the query, splits each
    article into paragraph windows and uses the *best window score* as the
    article's rerank score.  This prevents long articles from diluting the
    relevant paragraph's signal in the cross-encoder cross-attention.

    Enabled by USE_PARAGRAPH_WINDOW_RERANKER=true.
    """

    _MIN_WINDOW_CHARS: int = 30

    def __init__(self, model: str, top_n: int, alpha: float, max_windows: int) -> None:
        super().__init__()
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(model)
        self._top_n = top_n
        self._alpha = alpha
        self._max_windows = max_windows

    @classmethod
    def class_name(cls) -> str:
        return "ParagraphWindowReranker"

    def _split_windows(self, text: str) -> list[str]:
        """Split article text into paragraph windows, evenly sampled up to max_windows."""
        raw = re.split(r"\n\n+", text)
        windows = [w.strip() for w in raw if len(w.strip()) >= self._MIN_WINDOW_CHARS]
        if not windows:
            return [text]
        if len(windows) <= self._max_windows:
            return windows
        # Evenly sample across the article so we cover early, middle, and late provisions
        step = len(windows) / self._max_windows
        return [windows[int(i * step)] for i in range(self._max_windows)]

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        if not nodes or query_bundle is None:
            return nodes

        query = query_bundle.query_str
        node_windows = [self._split_windows(n.node.get_content()) for n in nodes]
        all_pairs = [(query, w) for windows in node_windows for w in windows]

        with _reranker_lock:
            all_scores = list(self._model.predict(all_pairs))

        lo, hi = min(all_scores), max(all_scores)
        span = hi - lo if hi > lo else 1.0

        idx = 0
        for node, windows in zip(nodes, node_windows):
            count = len(windows)
            window_scores = all_scores[idx: idx + count]
            idx += count
            best_idx = max(range(count), key=lambda i: window_scores[i])
            best_raw = window_scores[best_idx]
            best_norm = (best_raw - lo) / span
            retrieval_score = node.score or 0.0
            node.score = self._alpha * retrieval_score + (1 - self._alpha) * best_norm
            # Store the best-matching window in metadata for debugging / future display use
            node.node.metadata["best_paragraph"] = windows[best_idx]

        return sorted(nodes, key=lambda n: n.score or 0.0, reverse=True)[: self._top_n]


_FALSE_PREMISE_RULE = (
    "FALSE PREMISE RULE: If the question asks whether X qualifies as / is equivalent to / "
    "satisfies / can be treated as Y, and the context clearly shows they are different concepts "
    "or different requirements, your Direct Answer MUST begin with 'No.' and immediately state "
    "what the CRR actually requires. Do not hedge or partially affirm a false premise. "
    "Apply ONLY when the context clearly contradicts the question's assumption; answer normally otherwise.\n"
    "Examples of correct false-premise handling (do not reproduce these verbatim — use the "
    "actual context to answer):\n"
    "  Q: Can government bonds received as collateral be recognised as variation margin for "
    "leverage ratio purposes?\n"
    "  A: No. Article 429c(3) restricts variation margin recognition to cash collateral only; "
    "non-cash instruments such as government bonds do not qualify.\n"
    "  Q: If liquid assets cover long-term funding needs over 12 months, is the institution "
    "compliant with the liquidity coverage requirement?\n"
    "  A: No. Article 412 governs a 30-day stressed outflow window (LCR); long-term structural "
    "funding stability is a separate requirement under Article 413 (NSFR).\n"
    "  Q: Is a 'personal investment company' considered a financial customer for liquidity "
    "outflows?\n"
    "  A: No. Article 411(3) defines personal investment company as a distinct category, "
    "separate from financial customer under Article 411(1).\n"
)

_LEGAL_QA_TEMPLATE = PromptTemplate(
    "You are a regulatory compliance expert specialising in EU prudential banking regulation "
    "(CRR – Regulation (EU) No 575/2013).\n\n"
    "Use ONLY the context below to answer the question. "
    "Do not speculate or introduce information not present in the context.\n"
    "CRITICAL CITATION RULE: Cite ONLY article numbers whose text appears verbatim in the "
    "context provided below. If the question asks about a specific article but that article "
    "is not present in the context, state: 'Article X was not found in the provided context.'\n"
    + _FALSE_PREMISE_RULE +
    "If the context does not contain enough information to answer, respond with ONLY: "
    "'The provided context does not contain sufficient information to answer this question.'\n\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n\n"
    "Question: {query_str}\n\n"
    "Respond using the following structure. Omit any section that is not applicable.\n\n"
    "**Direct Answer**\n"
    "A concise answer to the question in 1–3 sentences.\n\n"
    "**Key Provisions**\n"
    "Bullet-point summary of the relevant rules, thresholds, or definitions drawn from the context. "
    "Each bullet must reference the Article it comes from (e.g. '- Article 92(1)(a): …'). "
    "Only cite articles whose text appears in the context above.\n\n"
    "**Conditions, Exceptions & Definitions**\n"
    "Any qualifications, carve-outs, special treatments, or defined terms that affect the answer.\n\n"
    "**Article References**\n"
    "Comma-separated list of every Article cited above. "
    "Only include articles present in the context (e.g. 'Article 92, Article 93, Article 128').\n\n"
    "Answer:"
)


_LEGAL_QA_TEMPLATE_WITH_HISTORY = PromptTemplate(
    "You are a regulatory compliance expert specialising in EU prudential banking regulation "
    "(CRR – Regulation (EU) No 575/2013).\n\n"
    "Use ONLY the context below to answer the question. "
    "Do not speculate or introduce information not present in the context.\n"
    "CRITICAL CITATION RULE: Cite ONLY article numbers whose text appears verbatim in the "
    "context provided below. If the question asks about a specific article but that article "
    "is not present in the context, state: 'Article X was not found in the provided context.'\n"
    + _FALSE_PREMISE_RULE +
    "If the context does not contain enough information to answer, respond with ONLY: "
    "'The provided context does not contain sufficient information to answer this question.'\n\n"
    "Prior conversation:\n{history_str}\n\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n\n"
    "Question: {query_str}\n\n"
    "Respond using the following structure. Omit any section that is not applicable.\n\n"
    "**Direct Answer**\n"
    "A concise answer to the question in 1–3 sentences.\n\n"
    "**Key Provisions**\n"
    "Bullet-point summary of the relevant rules, thresholds, or definitions drawn from the context. "
    "Each bullet must reference the Article it comes from (e.g. '- Article 92(1)(a): …'). "
    "Only cite articles whose text appears in the context above.\n\n"
    "**Conditions, Exceptions & Definitions**\n"
    "Any qualifications, carve-outs, special treatments, or defined terms that affect the answer.\n\n"
    "**Article References**\n"
    "Comma-separated list of every Article cited above. "
    "Only include articles present in the context (e.g. 'Article 92, Article 93, Article 128').\n\n"
    "Answer:"
)


def _format_history(history: list[dict], max_turns: int = _HISTORY_MAX_TURNS) -> str:
    """Format conversation history into a readable string for prompt injection.

    Args:
        history: List of dicts with 'question' and 'answer' keys.
        max_turns: Maximum number of recent turns to include.

    Returns:
        Formatted string, or empty string if history is empty.
    """
    if not history:
        return ""
    turns = history[-max_turns:]
    formatted = []
    for turn in turns:
        formatted.append(f"Q: {turn['question']}\nA: {turn['answer']}")
    return "\n\n---\n\n".join(formatted)


def _rewrite_query_with_history(
    query: str,
    history: list[dict],
    api_key: Optional[str],
    model: str = LLM_MODEL,
) -> str:
    """Rewrite a follow-up query into a standalone question using conversation history.

    Args:
        query: The current user query (possibly a follow-up).
        history: List of prior Q&A turns.
        api_key: OpenAI API key.
        model: LLM model to use for rewriting.

    Returns:
        Rewritten standalone query, or the original query if rewriting fails.
    """
    history_str = _format_history(history)
    prompt = (
        "You are helping rewrite a follow-up question into a complete, self-contained question "
        "that can be understood without any prior context.\n\n"
        "Conversation history:\n"
        f"{history_str}\n\n"
        "Follow-up question: " + query + "\n\n"
        "Rewrite the follow-up question as a complete, standalone question in the SAME LANGUAGE "
        "as the original question. If the question is already self-contained, return it unchanged. "
        "Return ONLY the rewritten question, nothing else."
    )
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            timeout=30.0,
            **_EVAL_KWARGS,
        )
        rewritten = (response.choices[0].message.content or "").strip()
        return rewritten if rewritten else query
    except Exception as exc:
        logger.warning("Query rewrite failed (%s) — using original query", exc)
        return query


# CRR-specific abbreviation expansion: inserted before embedding so the model sees full terms.
# Matched case-sensitively so generic words (e.g. "at1" in prose) are not expanded.
_ABBREV_MAP: dict[str, str] = {
    "CET1": "Common Equity Tier 1",
    "AT1": "Additional Tier 1",
    "T2": "Tier 2 capital",
    "LCR": "Liquidity Coverage Ratio",
    "NSFR": "Net Stable Funding Ratio",
    "MREL": "Minimum Requirement for own funds and Eligible Liabilities",
    "RWA": "risk-weighted assets",
    "IRB": "Internal Ratings-Based approach",
    "CVA": "Credit Valuation Adjustment",
    "CCR": "Counterparty Credit Risk",
    "EAD": "Exposure at Default",
    "LGD": "Loss Given Default",
    "ECAI": "External Credit Assessment Institution",
    "SFT": "Securities Financing Transaction",
    "CCP": "Central Counterparty",
    "QCCP": "Qualifying Central Counterparty",
    "EBA": "European Banking Authority",
    "STS": "Simple Transparent and Standardised securitisation",
    "SEC-IRBA": "Securitisation Internal Ratings-Based Approach",
    "SEC-SA": "Securitisation Standardised Approach",
    "SA": "Standardised Approach",
    "CIU": "Collective Investment Undertaking",
    "NPE": "Non-Performing Exposure",
    "TREA": "Total Risk Exposure Amount",
    "HQLA": "High Quality Liquid Asset",
    "PD": "Probability of Default",
    "CF": "Conversion Factor",
    "SME": "Small and Medium-sized Enterprise",
    "IPRE": "Income-Producing Real Estate",
    "HVCRE": "High-Volatility Commercial Real Estate",
}
_ABBREV_RE = re.compile(r"\b(" + "|".join(re.escape(k) for k in _ABBREV_MAP) + r")\b")

# Legal paraphrase → canonical CRR term expansion.
# Appends the canonical term inline so both BM25 and dense retrieval get better signals
# when users describe concepts with common synonyms not in the CRR text.
_SYNONYM_MAP: dict[str, str] = {
    "preference shares": "Additional Tier 1 instruments",
    "perpetual bonds": "Additional Tier 1 instruments",
    "subordinated notes": "Tier 2 instruments",
    "subordinated debt": "Tier 2 instruments",
    "bail-in bonds": "eligible liabilities",
    "minority interests": "minority interest capital instruments",
    "non-performing loans": "Non-Performing Exposures",
    "non-performing loan": "Non-Performing Exposure",
    "securitisation position": "securitisation exposure",
    "enterprises": "corporates",
    "enterprise": "corporate",
    # run_10 additions — surgical CRR-specific mappings only (broad terms removed after regression)
    "local authority": "regional government or local authority",
    "local public authority": "regional government or local authority",
    "local authorities": "regional governments or local authorities",
    "pledged assets": "encumbered assets",
    "pledged collateral": "encumbered assets",
    "core capital": "Common Equity Tier 1",
    # run_26 additions — targets diluted_embedding failures where plain-language
    # synonyms caused embedding/BM25 misses against precise CRR vocabulary.
    "accumulated earnings": "retained earnings",
    "maximum allowable exposure": "large exposure",
    "pledged as collateral": "encumbered",
    "easily sellable": "liquid assets high quality",
    "available liquid resources": "liquid assets",
    "solvency threshold": "initial capital own funds",
    "internal permission for tailored risk evaluation": "Internal Ratings-Based approach IRB",
    "blend of assets": "mixed pool",
}
_SYNONYM_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _SYNONYM_MAP) + r")\b",
    re.IGNORECASE,
)

# Normalise shorthand article references to the canonical "Article N" form used in metadata.
# Handles: "art. 92", "Art 92", "article92", "§92" → "Article 92"
_ART_RE = re.compile(r"\b(?:art(?:icle|\.)?)\s*(\d[\w]*)", re.IGNORECASE)

# Matches any canonical "Article N" reference after query normalisation.
_ARTICLE_REF_RE = re.compile(r"\bArticle\s+(\d[\w]*)\b", re.IGNORECASE)

# Matches external directive/regulation citations: "Article 10 of Directive 2014/59/EU",
# "Articles 74 and 83 of Regulation (EU) No 648/2012".  These must be stripped before
# counting CRR article references so they are not mistaken for internal lookups.
_EXTERNAL_DIRECTIVE_RE = re.compile(
    r"\bArticles?\s+\d[\w]*(?:\s*(?:,|and|or)\s*\d[\w]*)*\s+of\s+(?:Directive|Regulation)\b[^\n,;]*",
    re.IGNORECASE,
)

# Matches coordinated article-number runs: "Article 92 and 93", "Articles 92, 93 and 94".
# Presence of this pattern means multiple CRR articles are mentioned → no direct lookup.
_ARTICLE_COORD_RE = re.compile(
    r"\bArticles?\s+\d[\w]*(?:\s*(?:,|and|or)\s*\d[\w]*)+",
    re.IGNORECASE,
)

# Matches "Articles N to M" range syntax for query-time expansion.
_RANGE_RE = re.compile(r"\bArticles?\s+(\d+[a-z]*)\s+to\s+(\d+[a-z]*)\b", re.I)


def _expand_article_ranges(query: str) -> str:
    """Expand 'Articles N to M' into explicit 'Article N Article N+1 ... Article M'.

    Helps BM25/dense retrieval find individual articles in a range query.
    A sanity cap prevents expansion of unreasonably large ranges (> 20 articles).
    """
    def expand(m: re.Match) -> str:
        lo_str, hi_str = m.group(1), m.group(2)
        lo = int(re.match(r"\d+", lo_str).group())
        hi = int(re.match(r"\d+", hi_str).group())
        if hi <= lo or (hi - lo) > 20:
            return m.group(0)
        return " ".join(f"Article {n}" for n in range(lo, hi + 1))
    return _RANGE_RE.sub(expand, query)


_ROMAN_ORDER = ["I", "II", "III", "IV"]


def _ref_sort_key(a: str) -> tuple[int, str]:
    """Sort key for article-number strings: numeric part first, alpha suffix second.

    Handles plain numbers ("92"), lettered variants ("92a", "92aa"), and
    non-numeric IDs (fall to the front with key 0).
    """
    m = re.match(r"^(\d+)(.*)", a)
    return (int(m.group(1)), m.group(2)) if m else (0, a)


def _truncate_context(context_str: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Hard-truncate context to stay within the token budget.

    Cuts at the last article separator boundary (---) before the limit so we
    don't split mid-article. Falls back to a hard character cut if no separator
    is found within the window.
    """
    if len(context_str) <= max_chars:
        return context_str
    window = context_str[:max_chars]
    cut = window.rfind("\n\n---\n\n")
    if cut > 0:
        logger.warning(
            "Context truncated from %d to %d chars to stay within token budget.",
            len(context_str), cut,
        )
        return context_str[:cut]
    logger.warning(
        "Context hard-truncated from %d to %d chars (no article boundary found).",
        len(context_str), max_chars,
    )
    return window


# ---------------------------------------------------------------------------
# Synthesis completeness helpers (run_27)
# ---------------------------------------------------------------------------

# Patterns that reliably identify CRR numerical thresholds.
# Ordered from most specific to least to reduce false-positive noise.
_THRESHOLD_RE_LIST = [
    re.compile(r"\b\d[\d,.]*%"),                                         # percentages: 4.5%, 1,250%
    re.compile(r"\b\d+\s*(?:calendar\s+)?(?:days?|months?|years?|business\s+days?)", re.IGNORECASE),
    re.compile(r"\bEUR\s*[\d,.]+(?:\s*(?:million|billion))?", re.IGNORECASE),
    re.compile(r"\b\d+(?:\.\d+)?\s*basis\s+points?", re.IGNORECASE),
]
# Article header pattern as emitted by the context assembly loop.
_ART_HEADER_RE = re.compile(r"^Article\s+(\w+)", re.IGNORECASE)


def _extract_thresholds(text: str) -> list[str]:
    """Return sorted unique numerical threshold strings found in *text*."""
    found: set[str] = set()
    for pat in _THRESHOLD_RE_LIST:
        for m in pat.finditer(text):
            found.add(m.group(0).strip())
    return sorted(found)


def _build_key_facts_block(context_str: str) -> str:
    """Build a compact threshold preamble from the assembled context.

    Parses each ``Article N — Title`` section and lists its numerical
    thresholds.  The block is prepended to the context so the LLM has an
    explicit fact-sheet and is less likely to omit key values.

    Returns an empty string when no thresholds are detected (avoids adding
    noise for purely definitional queries).
    """
    lines: list[str] = []
    for section in context_str.split("\n\n---\n\n"):
        header_line = section.strip().split("\n")[0]
        thresholds = _extract_thresholds(section)
        if thresholds:
            lines.append(f"- {header_line}: {', '.join(thresholds)}")
    if not lines:
        return ""
    return (
        "KEY NUMERICAL THRESHOLDS (extracted from retrieved articles — "
        "ensure ALL relevant ones are addressed in your answer):\n"
        + "\n".join(lines)
        + "\n\n"
    )


def _append_missing_thresholds(context_str: str, answer: str) -> str:
    """Post-generation check: append any thresholds from *cited* articles
    that do not appear in the answer.

    Only checks articles referenced in the answer's ``Article References``
    section to avoid injecting noise from unreferenced context sections.
    Returns the answer unchanged when nothing is missing.
    """
    cited = set(re.findall(r"Article\s+(\w+)", answer, re.IGNORECASE))
    if not cited:
        return answer

    missing_by_art: list[str] = []
    for section in context_str.split("\n\n---\n\n"):
        header_line = section.strip().split("\n")[0]
        m = _ART_HEADER_RE.match(header_line)
        if not m or m.group(1) not in cited:
            continue
        art_thresholds = _extract_thresholds(section)
        absent = [t for t in art_thresholds if t not in answer]
        if absent:
            missing_by_art.append(f"Article {m.group(1)}: {', '.join(absent)}")

    if not missing_by_art:
        return answer

    note = (
        "\n\n> **Completeness note:** The following numerical thresholds appear in the "
        "cited articles but were not explicitly stated above: "
        + "; ".join(missing_by_art)
        + "."
    )
    return answer + note


def _expand_synonyms(query: str) -> str:
    """Expand legal paraphrases to canonical CRR terms for improved retrieval recall."""
    return _SYNONYM_RE.sub(
        lambda m: f"{m.group(0)} ({_SYNONYM_MAP[m.group(0).lower()]})",
        query,
    )


def _normalise_query(query: str) -> str:
    """Expand CRR abbreviations, synonyms, canonicalise article references, and expand ranges."""
    query = _ABBREV_RE.sub(lambda m: f"{m.group(1)} ({_ABBREV_MAP[m.group(1)]})", query)
    query = _ART_RE.sub(lambda m: f"Article {m.group(1)}", query)
    query = _expand_article_ranges(query)
    query = _expand_synonyms(query)
    return query


_DEF_QUERY_RE = re.compile(
    r"""
    (?:
        (?:what\s+(?:is|are)\s+(?:(?:a|an|the)\s+)?(?:definition\s+of\s+)?|
           what\s+does\s+|
           define\s+|
           definition\s+of\s+|
           meaning\s+of\s+)
        (['\u2018\u2019\u201c\u201d]?[a-zA-Z][\w\s\-]*?)  # group 1: term
        (?:\s+mean(?:s|ing)?|\?|$)
    |
        article\s+4\s*\(\s*(\d+[a-z]?)\s*\)  # group 2: definition number
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _detect_definition_query(query: str) -> Optional[str]:
    """Return '#N' for Article 4(N) lookups, a lowercase term string, or None."""
    m = _DEF_QUERY_RE.search(query)
    if m is None:
        return None
    if m.group(2):
        return f"#{m.group(2)}"
    term = (m.group(1) or "").strip().lower().strip("'\u2018\u2019\u201c\u201d").rstrip("?")
    if not term:
        return None
    # Reject false positives where "what is/are X" captured a long phrase
    # (real definition terms are at most 4 words)
    if len(term.split()) > 4:
        return None
    return term


def _art4_source(lang: str, entry: Optional[dict] = None) -> dict:
    return {
        "text": (entry["text"] if entry else "Article 4 — Definitions")[:500],
        "score": 1.0,
        "metadata": {"article": "4", "article_title": "Definitions", "language": lang},
        "expanded": False,
    }


def _detect_direct_article_lookup(query: str) -> Optional[str]:
    """Return article number if the query references exactly one CRR article, else None.

    Handles any phrasing that mentions a single article:
      - "What are the requirements of Article 73 of the CRR?"
      - "Explain Article 92"
      - "Does Article 428 apply to investment firms?"

    Does NOT trigger when:
      - Multiple distinct articles are mentioned ("How do Article 92 and 93 relate?")
      - Coordinated bare numbers follow a single article ref ("Article 92 and 93")
      - The article belongs to an external directive/regulation citation
        ("Article 10 of Directive 2014/59/EU")
    """
    # Remove external directive/regulation refs before counting CRR articles.
    stripped = _EXTERNAL_DIRECTIVE_RE.sub("", query)
    # Coordinated article-number runs ("Article 92 and 93") signal multi-article intent.
    if _ARTICLE_COORD_RE.search(stripped):
        return None
    matches = _ARTICLE_REF_RE.findall(stripped)
    unique = set(matches)
    return unique.pop() if len(unique) == 1 else None


def _enrich_open_ended_query(
    query: str,
    api_key: Optional[str],
    model: str = "gpt-4o-mini",
) -> str:
    """Predict relevant CRR article numbers for open-ended queries and append as search hints.

    Called for CRR_SPECIFIC queries with no explicit article reference to improve
    retrieval recall for semantically weak queries. Adds ~1-2s latency.

    Returns:
        Query enriched with article number hints, or original query on failure.
    """
    prompt = (
        "You are a CRR (EU Capital Requirements Regulation, No 575/2013) expert.\n"
        "Given the question below, list the 2-3 CRR article numbers most likely to "
        "contain the answer. Respond with ONLY the article numbers, comma-separated "
        "(e.g. '92, 93, 128'). Do not include any explanation.\n\n"
        f"Question: {query}"
    )
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            timeout=15.0,
            max_tokens=30,
        )
        hints = (response.choices[0].message.content or "").strip()
        # Validate: hints must look like article numbers (digits, letters, commas, spaces)
        if hints and re.match(r'^[\d\w,\s]+$', hints):
            logger.info("Article hint enrichment: '%s...' → hints: %s", query[:60], hints)
            return f"{query} [Relevant CRR articles: {hints}]"
    except Exception as exc:
        logger.warning("Article hint enrichment failed (%s) — using original query", exc)
    return query


def _generate_hyde_query(
    query: str,
    api_key: Optional[str],
    model: str = "gpt-4o-mini",
) -> str:
    """Combined HyDE + article-hint query for open-ended CRR retrieval.

    Single LLM call that returns:
    - A hypothetical CRR-style passage (bridges vocabulary gap for dense/BGE-M3)
    - Predicted article numbers (concrete BM25/sparse anchor tokens)

    Combined into: "{hypothesis} [Relevant CRR articles: {hints}]"

    Dense retrieval benefits from CRR legal vocabulary in the hypothesis.
    BM25/sparse benefits from explicit article number tokens even when the
    hypothesis misleads the dense vector (diluted-embedding failure mode).

    Returns:
        Combined hypothesis + article hints string, or original query on failure.
    """
    prompt = (
        "You are a CRR (EU Capital Requirements Regulation, No 575/2013) expert.\n"
        "Given the question below, respond in EXACTLY this format with no other text:\n\n"
        "PASSAGE: <3-4 sentences in formal CRR legislative style that would answer "
        "the question, using precise legal terminology. Write as the regulation text "
        "itself — do not say 'According to CRR'.>\n"
        "ARTICLES: <2-3 CRR article numbers most likely to contain the answer, "
        "comma-separated, e.g. 92, 93>\n\n"
        f"Question: {query}"
    )
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            timeout=15.0,
            max_tokens=220,
            **_EVAL_KWARGS,
        )
        raw = (response.choices[0].message.content or "").strip()

        # Parse PASSAGE and ARTICLES from structured response.
        # Accumulate continuation lines after PASSAGE: until ARTICLES: or end of input,
        # so multi-line passages are not silently truncated.
        passage_lines: list[str] = []
        hints = ""
        in_passage = False
        for line in raw.splitlines():
            if line.startswith("PASSAGE:"):
                passage_lines = [line[len("PASSAGE:"):].strip()]
                in_passage = True
            elif line.startswith("ARTICLES:"):
                hints = line[len("ARTICLES:"):].strip()
                in_passage = False
            elif in_passage:
                passage_lines.append(line.strip())
        passage = " ".join(passage_lines)

        # Validate hints look like article numbers (digits, letters, commas, spaces)
        if hints and not re.match(r'^[\d\w,\s]+$', hints):
            hints = ""

        if passage:
            retrieve_query = passage
            if hints:
                retrieve_query = f"{passage} [Relevant CRR articles: {hints}]"
            logger.info(
                "HyDE+hints for '%s...': passage='%s...' articles=%s",
                query[:50], passage[:60], hints or "none",
            )
            return retrieve_query
    except Exception as exc:
        logger.warning("HyDE generation failed (%s) — using original query", exc)
    return query


def _rewrite_query_crr_domain(
    query: str,
    api_key: Optional[str],
    model: str = "gpt-4o-mini",
) -> str:
    """Rewrite a plain-language query into CRR legal register before embedding.

    Unlike HyDE (which generates a hypothetical passage), this produces a concise
    rewritten question that replaces informal terms with the exact legal vocabulary
    used in the regulation text. Targets terminology dilution failures where the
    query shares few tokens with the article (e.g. "pledged bonds" vs "encumbered
    assets", "total risk exposure" vs "TREA", "cash pooling" vs "cash pooling
    arrangements under Article 429b").

    Returns:
        Rewritten query string, or original query on failure.
    """
    prompt = (
        "You are a CRR (EU Capital Requirements Regulation No 575/2013) legal terminology expert.\n\n"
        "Rewrite the question below using the precise legal vocabulary found in the CRR text itself.\n"
        "Replace informal or plain-language terms with the exact CRR/Basel III definitions "
        "(e.g. 'bank' → 'credit institution', 'total risk exposure' → 'total risk exposure amount (TREA)', "
        "'liquid assets' → 'high-quality liquid assets (HQLA)', 'leveraged loan' → 'exposure', "
        "'pledged bonds' → 'encumbered assets', 'profit transfer agreement' → 'Gewinnabführungsvertrag').\n\n"
        "Rules:\n"
        "- Preserve the question's intent and structure exactly\n"
        "- Add the official CRR term alongside or in place of informal terms\n"
        "- Keep it as a question, maximum 2 sentences\n"
        "- Do NOT answer the question\n"
        "- Return ONLY the rewritten question, no explanation or preamble\n\n"
        f"Question: {query}"
    )
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            timeout=10.0,
            max_tokens=120,
            **_EVAL_KWARGS,
        )
        rewritten = (response.choices[0].message.content or "").strip()
        if rewritten and rewritten != query:
            logger.info(
                "Domain rewrite: '%s...' → '%s...'",
                query[:60], rewritten[:60],
            )
            return rewritten
    except Exception as exc:
        logger.warning("Domain query rewrite failed (%s) — using original query", exc)
    return query


def _generate_sub_queries(
    query: str,
    api_key: Optional[str],
    model: str = "gpt-4o-mini",
) -> list[str]:
    """Break a complex multi-hop question into 2-3 simpler sub-queries.

    Used for questions spanning multiple articles or requiring comparison of
    different regulatory concepts. Each sub-query targets a single aspect.

    Returns:
        List of up to 3 sub-queries, or empty list on failure.
    """
    prompt = (
        "You are a CRR (EU Capital Requirements Regulation) expert.\n"
        "Break the following complex question into 2-3 simpler sub-questions that can "
        "each be answered from individual CRR articles. Return ONLY the sub-questions, "
        "one per line, with no numbering or bullets.\n\n"
        f"Question: {query}"
    )
    try:
        client = openai.OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            timeout=15.0,
            max_tokens=200,
            **_EVAL_KWARGS,
        )
        lines = (resp.choices[0].message.content or "").strip().split('\n')
        sub_queries = [q.strip().lstrip('0123456789.-) ') for q in lines if q.strip()]
        logger.info("Generated %d sub-queries for '%s...'", len(sub_queries), query[:60])
        return sub_queries[:3]
    except Exception as exc:
        logger.warning("Sub-query generation failed (%s)", exc)
        return []


@dataclass
class QueryResult:
    answer: str
    sources: list[dict]
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))


def merge_rrf(
    vector_nodes: list[NodeWithScore],
    toc_nodes: list[NodeWithScore],
    k: int = 60,
    cap: int = RERANK_TOP_N,
) -> list[NodeWithScore]:
    """Merge two ranked node lists using Reciprocal Rank Fusion.

    Combines vector retrieval results with ToC-routed retrieval results.
    Each node receives a score = sum(1 / (k + rank + 1)) across both lists,
    where rank is 0-based. Nodes appearing in both lists get a cumulative boost.

    Scores are normalised to [0, 1] after fusion so that downstream confidence
    thresholds (e.g. _LOW_CONFIDENCE_THRESHOLD) remain meaningful.

    Args:
        vector_nodes: Nodes from the primary hybrid vector retrieval (may be
                      reranked). Order reflects relevance.
        toc_nodes:    Nodes fetched for articles predicted by ToC LLM routing.
        k:            RRF smoothing constant (default 60, standard value).
        cap:          Maximum number of nodes to return.

    Returns:
        Merged, deduplicated list of NodeWithScore, capped at `cap`, with
        normalised RRF scores assigned to each node's `.score` attribute.
    """
    scores: dict[str, float] = {}
    node_map: dict[str, NodeWithScore] = {}

    for rank, node in enumerate(vector_nodes):
        nid = node.node.node_id
        scores[nid] = scores.get(nid, 0.0) + 1.0 / (k + rank + 1)
        # Keep node with highest original score for content access
        if nid not in node_map or (node.score or 0.0) > (node_map[nid].score or 0.0):
            node_map[nid] = node

    for rank, node in enumerate(toc_nodes):
        nid = node.node.node_id
        scores[nid] = scores.get(nid, 0.0) + 1.0 / (k + rank + 1)
        if nid not in node_map or (node.score or 0.0) > (node_map[nid].score or 0.0):
            node_map[nid] = node

    if not scores:
        return []

    # Normalise to [0, 1]
    max_score = max(scores.values())
    if max_score > 0:
        scores = {nid: s / max_score for nid, s in scores.items()}

    # Sort by RRF score descending, assign normalised score, cap
    sorted_ids = sorted(scores, key=lambda nid: scores[nid], reverse=True)[:cap]
    merged: list[NodeWithScore] = []
    for nid in sorted_ids:
        node = node_map[nid]
        node.score = scores[nid]
        merged.append(node)

    return merged


class QueryEngine:
    """Wraps hybrid retrieval + cross-reference expansion + GPT-4o synthesis."""

    def __init__(
        self,
        vector_store: VectorStore,
        indexer: HierarchicalIndexer,
        openai_api_key: Optional[str] = None,
        llm_model: str = LLM_MODEL,
        max_cross_ref_expansions: int = 3,
        use_reranker: Optional[bool] = None,
    ) -> None:
        self.vector_store = vector_store
        self.indexer = indexer
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.llm_model = llm_model
        self.max_cross_ref_expansions = max_cross_ref_expansions
        # If not explicitly set, read from env (USE_RERANKER=true); default False
        if use_reranker is None:
            self.use_reranker = os.getenv("USE_RERANKER", "false").lower() == "true"
        else:
            self.use_reranker = use_reranker
        self._engine: Optional[RetrieverQueryEngine] = None
        self._vector_index: Optional[VectorStoreIndex] = None
        self._engine_cache: dict[str, RetrieverQueryEngine] = {}
        self._engine_cache_lock = threading.Lock()
        self._reranker = None
        self._para_reranker = None
        self._title_booster = None
        self._tiebreaker = None
        self._defs: Optional[object] = None  # DefinitionsStore, imported lazily
        self._toc: Optional[object] = None   # TocStore, imported lazily
        self._article_graph: Optional[object] = None  # ArticleGraph, built lazily on first use
        self._graph_build_lock = threading.Lock()
        self._graph_built = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Build the retriever chain from the persisted index.

        All new objects are built into local variables before being swapped in under
        _engine_cache_lock, so concurrent retrieve() calls always see a consistent
        snapshot — never a partially-refreshed state.
        """
        # --- Build phase (no mutations to live fields) ---
        new_vector_index = self.indexer.load()  # may set Settings.llm = None as side effect
        self._configure_settings()              # restore OpenAI LLM after indexer resets it

        new_reranker = None
        if self.use_reranker:
            from llama_index.core.postprocessor import SentenceTransformerRerank
            new_reranker = SentenceTransformerRerank(top_n=RERANK_TOP_N, model=RERANKER_MODEL)
            logger.info("Reranker loaded: %s (top_n=%d)", RERANKER_MODEL, RERANK_TOP_N)

        new_title_booster = None
        if TITLE_BOOST_WEIGHT > 0:
            new_title_booster = ArticleTitleBoostPostprocessor(
                boost_weight=TITLE_BOOST_WEIGHT, top_n=RERANK_TOP_N
            )
            logger.info("ArticleTitleBoostPostprocessor enabled (weight=%.2f)", TITLE_BOOST_WEIGHT)

        new_tiebreaker = None
        if ADJACENT_TIEBREAK_DELTA > 0:
            new_tiebreaker = AdjacentArticleTiebreakerPostprocessor(delta=ADJACENT_TIEBREAK_DELTA)
            logger.info("AdjacentArticleTiebreakerPostprocessor enabled (delta=%.3f)", ADJACENT_TIEBREAK_DELTA)

        new_para_reranker = None
        if os.getenv("USE_PARAGRAPH_WINDOW_RERANKER", "false").lower() == "true":
            if USE_PARAGRAPH_CHUNKING:
                # ParagraphWindowReranker is redundant when the index already contains
                # paragraph-level chunks — each chunk IS already a paragraph window.
                # Fall back to BlendedReranker to avoid double-splitting paragraph text.
                logger.warning(
                    "USE_PARAGRAPH_WINDOW_RERANKER=true is redundant with USE_PARAGRAPH_CHUNKING=true. "
                    "Falling back to BlendedReranker."
                )
                if new_reranker is None:
                    new_reranker = BlendedReranker(
                        model=RERANKER_MODEL,
                        top_n=RERANK_TOP_N,
                        alpha=RERANK_BLEND_ALPHA,
                    )
            else:
                new_para_reranker = ParagraphWindowReranker(
                    model=RERANKER_MODEL,
                    top_n=RERANK_TOP_N,
                    alpha=RERANK_BLEND_ALPHA,
                    max_windows=PARAGRAPH_WINDOW_MAX_WINDOWS,
                )
                logger.info(
                    "ParagraphWindowReranker enabled (model=%s, top_n=%d, max_windows=%d)",
                    RERANKER_MODEL, RERANK_TOP_N, PARAGRAPH_WINDOW_MAX_WINDOWS,
                )

        new_engine = self._build_engine(new_vector_index)

        # Load Article 4 definitions fast-path store
        from src.query.definitions_store import DefinitionsStore
        new_defs = DefinitionsStore(self.vector_store)
        for lang in ("en", "it"):
            try:
                new_defs.load(lang)
            except Exception as exc:
                logger.warning(
                    "DefinitionsStore load failed for language=%s: %s", lang, exc
                )

        # Load ToC store for LLM-guided routing (opt-in via USE_TOC_ROUTING=true)
        new_toc = None
        if os.getenv("USE_TOC_ROUTING", "false").lower() == "true":
            from src.query.toc_store import TocStore
            new_toc = TocStore(self.vector_store)
            for lang in ("en", "it"):
                try:
                    new_toc.load(lang)
                except Exception as exc:
                    logger.warning(
                        "TocStore load failed for language=%s: %s", lang, exc
                    )

        # --- Atomic swap phase (brief lock hold — reference assignments only) ---
        with self._engine_cache_lock:
            self._vector_index = new_vector_index
            self._engine = new_engine
            self._engine_cache = {}
            self._reranker = new_reranker
            self._para_reranker = new_para_reranker
            self._title_booster = new_title_booster
            self._tiebreaker = new_tiebreaker
            self._defs = new_defs
            self._toc = new_toc

        logger.info("QueryEngine ready.")

    def is_loaded(self) -> bool:
        return self._engine is not None

    def _ensure_graph(self) -> Optional[object]:
        """Return the ArticleGraph, building it on first call (lazy, thread-safe).

        Returns None if USE_ARTICLE_GRAPH=false or if the Qdrant client is unavailable.
        """
        if not USE_ARTICLE_GRAPH:
            return None
        if self._graph_built:
            return self._article_graph
        with self._graph_build_lock:
            if self._graph_built:
                return self._article_graph
            client = self.vector_store._client
            if client is None:
                logger.warning("ArticleGraph: Qdrant client not available — skipping graph build.")
                self._graph_built = True
                return None
            try:
                from src.query.article_graph import ArticleGraph
                graph = ArticleGraph()
                graph.build_from_qdrant(
                    client=client,
                    collection_name=self.vector_store.collection_name,
                    language="en",
                )
                self._article_graph = graph
                logger.info(
                    "ArticleGraph ready: %d nodes, %d edges",
                    graph.node_count, graph.edge_count,
                )
            except Exception as exc:
                logger.warning("ArticleGraph build failed: %s", exc)
                self._article_graph = None
            self._graph_built = True
        return self._article_graph

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def retrieve(
        self,
        user_query: str,
        language: Optional[str] = None,
        max_cross_ref_expansions: Optional[int] = None,
        is_multi_hop: bool = False,
    ) -> tuple:
        """Run stages 1 & 2 (retrieval + cross-reference expansion).

        Args:
            is_multi_hop: When True, expansion uses deeper BFS (depth=2), wider seeding
                          (source_limit=4), and a larger budget (min 5). Targets queries
                          that explicitly compare or relate multiple CRR concepts.

        Returns:
            (all_nodes_for_synthesis, sources, trace_id, normalised_query, engine)
        """
        # Snapshot the engine references atomically so this call sees a consistent
        # view even if load() is running concurrently in a background ingestion thread.
        with self._engine_cache_lock:
            engine_snap = self._engine
            vector_index_snap = self._vector_index
            cache_snap = self._engine_cache

        if engine_snap is None:
            raise RuntimeError("Call load() before querying.")

        trace_id = str(uuid.uuid4())
        t0 = time.perf_counter()
        user_query = _normalise_query(user_query)
        logger.info("[%s] Query: %s (language=%s)", trace_id, user_query, language)

        # Task-type routing: detect direct single-article lookups
        direct_art = _detect_direct_article_lookup(user_query)
        if direct_art:
            logger.info("[%s] Direct article lookup: Article %s", trace_id, direct_art)

        # Use a cached language-filtered engine — double-checked locking for thread safety
        if language:
            if language not in cache_snap:
                with self._engine_cache_lock:
                    # Re-read cache after acquiring lock; it may have been swapped by load().
                    cache_snap = self._engine_cache
                    vector_index_snap = self._vector_index
                    if language not in cache_snap:
                        cache_snap[language] = self._build_engine(
                            vector_index_snap, language_filter=language
                        )
            engine = cache_snap[language]
        else:
            engine = engine_snap

        query_bundle = QueryBundle(query_str=user_query)

        # Stage 1: Retrieval (includes postprocessors: similarity cutoff + reranker if enabled)
        t_ret = time.perf_counter()
        if direct_art:
            source_nodes = self._direct_article_retrieve(direct_art, user_query, language)
            if not source_nodes:
                logger.warning(
                    "[%s] Direct lookup for Article %s returned no nodes — falling back to semantic retrieval",
                    trace_id, direct_art,
                )
                source_nodes = engine.retrieve(query_bundle)
        else:
            source_nodes = engine.retrieve(query_bundle)
        t_retrieval_ms = round((time.perf_counter() - t_ret) * 1000)

        # Cross-lingual fallback: if language filter produced no results, retry without filter
        if language and not source_nodes:
            logger.info("[%s] No results for language=%s — retrying without language filter", trace_id, language)
            source_nodes = engine_snap.retrieve(query_bundle)
            # Post-filter: keep only nodes matching the target language.
            # The fallback may return mixed-language results.
            source_nodes = [
                n for n in source_nodes
                if n.node.metadata.get("language") == language
            ]

        # Stage 2: Cross-reference expansion
        expansions = max_cross_ref_expansions if max_cross_ref_expansions is not None \
            else self.max_cross_ref_expansions
        # Use the article graph only when primary retrieval surfaces ≥2 distinct articles.
        # Single-article retrievals are already well-anchored; graph expansion adds noise there.
        # This signal is post-retrieval and available at runtime (no query labels needed).
        retrieved_articles = {
            n.node.metadata.get("article", "")
            for n in source_nodes
            if n.node.metadata.get("article")
        }
        use_graph = len(retrieved_articles) >= 2
        # Multi-hop queries (regex or multi-article retrieval) get wider, deeper expansion.
        exp_source_limit = 4 if (is_multi_hop or use_graph) else 2
        exp_depth        = 2 if (is_multi_hop or use_graph) else 1
        exp_budget       = max(expansions, 5) if (is_multi_hop or use_graph) else expansions
        t_exp = time.perf_counter()
        expanded_nodes = self._expand_cross_references(
            source_nodes, language=language, limit=exp_budget,
            depth=exp_depth, source_limit=exp_source_limit,
            use_graph=use_graph,
        )
        t_expand_ms = round((time.perf_counter() - t_exp) * 1000)

        # Deduplicate expanded nodes against primary results
        source_ids = {node.node.node_id for node in source_nodes}
        deduped_expanded = [n for n in expanded_nodes if n.node.node_id not in source_ids]

        # Score expanded nodes with the reranker so they enter context in evidence order,
        # not arbitrary fetch order. Skip if no reranker is loaded.
        active_reranker = getattr(self, "_para_reranker", None) or getattr(self, "_reranker", None)
        if deduped_expanded and active_reranker is not None:
            try:
                scored_expanded = active_reranker.postprocess_nodes(
                    deduped_expanded, query_bundle=QueryBundle(query_str=user_query)
                )
                # postprocess_nodes truncates to top_n — restore all by merging back any
                # that were dropped (they still belong in context, just lower priority)
                scored_ids = {n.node.node_id for n in scored_expanded}
                unscored = [n for n in deduped_expanded if n.node.node_id not in scored_ids]
                deduped_expanded = scored_expanded + unscored
            except Exception as exc:
                logger.warning("Reranking of expanded nodes failed: %s", exc)

        all_nodes_for_synthesis = source_nodes + deduped_expanded

        def _snippet(node) -> str:
            # Prefer display_text (body without hierarchy prefix) for user-facing snippets.
            raw = node.node.metadata.get("display_text") or node.node.get_content()
            return raw[:500]

        sources = [
            {
                "text": _snippet(node),
                "score": float(round(node.score or 0.0, 4)),
                "metadata": node.node.metadata,
                "expanded": False,
            }
            for node in source_nodes
        ]
        sources += [
            {
                "text": _snippet(node),
                "score": 0.0,
                "metadata": node.node.metadata,
                "expanded": True,
            }
            for node in deduped_expanded
        ]

        logger.info(
            "[%s] Retrieved %d source nodes + %d expanded (%d total); "
            "retrieval=%dms, expansion=%dms",
            trace_id,
            len(source_nodes),
            len(deduped_expanded),
            len(all_nodes_for_synthesis),
            t_retrieval_ms,
            t_expand_ms,
        )

        return all_nodes_for_synthesis, sources, trace_id, user_query, engine

    def query(
        self,
        user_query: str,
        language: Optional[str] = None,
        max_cross_ref_expansions: Optional[int] = None,
        history: Optional[list[dict]] = None,
    ) -> QueryResult:
        t0 = time.perf_counter()
        history = history or []

        # Stage 0: Query rewriting — only when history is present
        if history:
            effective_query = _rewrite_query_with_history(
                user_query, history, self.openai_api_key, self.llm_model
            )
            if effective_query != user_query:
                logger.info("Query rewritten: '%s' → '%s'", user_query, effective_query)
        else:
            effective_query = user_query

        all_nodes_for_synthesis, sources, trace_id, normalised_query, _engine = self.retrieve(
            effective_query, language, max_cross_ref_expansions
        )

        # Build article-labelled context string (mirrors stream endpoint)
        context_parts = []
        _needs_parent_fetch = USE_PARAGRAPH_CHUNKING or USE_MIXED_CHUNKING
        if _needs_parent_fetch:
            # Any PARAGRAPH chunks in results — fetch parent ARTICLE docs for full synthesis context.
            # Mixed mode: deduplicator already ensured 1 chunk per article; some may be PARAGRAPH.
            # Paragraph mode: all chunks are PARAGRAPH; always fetch parent.
            # Group by article in score order (first occurrence = best-ranked article).
            article_order: list[str] = []
            _seen_arts: set[str] = set()
            node_language = language
            for node in all_nodes_for_synthesis:
                meta = node.node.metadata
                art = meta.get("article", "")
                if not node_language:
                    node_language = meta.get("language", "")
                if art and art not in _seen_arts:
                    _seen_arts.add(art)
                    article_order.append(art)
                    # In mixed mode: if this node is already an ARTICLE chunk, use it directly
                    # rather than fetching again (optimisation tracked via chunk_type).
            for art in article_order:
                # Find the winning chunk for this article in the result set
                winning_node = next(
                    (n for n in all_nodes_for_synthesis if n.node.metadata.get("article") == art),
                    None,
                )
                winning_type = (winning_node.node.metadata.get("chunk_type", "") if winning_node else "")
                if winning_type == "ARTICLE":
                    # Already have full article text — use directly, skip network fetch
                    meta = winning_node.node.metadata  # type: ignore[union-attr]
                    art_title = meta.get("article_title", "")
                    header = f"Article {art}" + (f" — {art_title}" if art_title else "")
                    content = meta.get("display_text") or winning_node.node.get_content()  # type: ignore[union-attr]
                    context_parts.append(f"{header}\n\n{content}")
                else:
                    fetch_conds: list[tuple[str, str]] = [("article", art), ("chunk_type", "ARTICLE")]
                    if node_language:
                        fetch_conds.append(("language", node_language))
                    art_nodes = self._fetch_nodes_direct(fetch_conds, top_k=1)
                    if art_nodes:
                        art_node = art_nodes[0].node
                        art_title = art_node.metadata.get("article_title", "")
                        header = f"Article {art}" + (f" — {art_title}" if art_title else "")
                        content = art_node.metadata.get("display_text") or art_node.get_content()
                        context_parts.append(f"{header}\n\n{content}")
                    else:
                        # Fallback: use paragraph chunk text directly if ARTICLE doc not found
                        if winning_node is not None:
                            meta = winning_node.node.metadata
                            art_title = meta.get("article_title", "")
                            header = f"Article {art}" + (f" — {art_title}" if art_title else "")
                            context_parts.append(f"{header}\n\n{winning_node.node.get_content()}")
        else:
            for node in all_nodes_for_synthesis:
                meta = node.node.metadata
                art = meta.get("article", "")
                art_title = meta.get("article_title", "")
                header = f"Article {art}" + (f" — {art_title}" if art_title else "")
                context_parts.append(f"{header}\n\n{node.node.get_content()}")
        context_str = _truncate_context("\n\n---\n\n".join(context_parts))

        # Stage 3: LLM synthesis — call OpenAI directly with optional history injection
        t_syn = time.perf_counter()
        history_str = _format_history(history)

        # Part A: prepend structured threshold fact-sheet so the LLM has
        # explicit scaffolding and is less likely to omit key values.
        key_facts = _build_key_facts_block(context_str)
        synthesis_context = key_facts + context_str if key_facts else context_str

        if history_str:
            prompt = _LEGAL_QA_TEMPLATE_WITH_HISTORY.format(
                history_str=history_str, context_str=synthesis_context, query_str=normalised_query
            )
        else:
            prompt = _LEGAL_QA_TEMPLATE.format(
                context_str=synthesis_context, query_str=normalised_query
            )

        oai_client = openai.OpenAI(api_key=self.openai_api_key)
        oai_response = oai_client.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            timeout=120.0,
        )
        answer = (oai_response.choices[0].message.content or "").strip()

        # Part B: deterministic post-check — append any thresholds from cited
        # articles that the LLM omitted, without re-synthesising.
        answer = _append_missing_thresholds(context_str, answer)

        t_synthesis_ms = round((time.perf_counter() - t_syn) * 1000)

        latency_ms = round((time.perf_counter() - t0) * 1000)
        logger.info(
            "[%s] answer=%d chars; total=%dms (synthesis=%dms)",
            trace_id,
            len(answer),
            latency_ms,
            t_synthesis_ms,
        )

        return QueryResult(answer=answer, sources=sources, trace_id=trace_id)

    # ------------------------------------------------------------------
    # Definitions fast-path
    # ------------------------------------------------------------------

    def lookup_definition(
        self, query: str, language: Optional[str] = None
    ) -> Optional[QueryResult]:
        """Return a QueryResult for Article 4 definition queries, or None to fall through to RAG.

        Handles:
        - Specific term queries: "What is the definition of institution?"
        - Article 4(N) direct lookups: "Article 4(1)"
        - Generic Article 4 queries: "Explain Article 4"
        """
        if self._defs is None:
            return None
        lang = language or "en"
        norm = _normalise_query(query)
        signal = _detect_definition_query(norm)

        # Generic Article 4 query (e.g. "Explain Article 4", "What is Article 4?")
        if signal is None:
            if _detect_direct_article_lookup(norm) == "4":
                if self._defs.is_loaded(lang):
                    answer = self._defs.summary(lang)
                    return QueryResult(answer=answer, sources=[_art4_source(lang)])
            return None

        # Specific definition lookup
        entry = None
        if signal.startswith("#"):
            entry = self._defs.lookup_by_number(signal[1:], lang)
            if entry is None and lang != "en":
                entry = self._defs.lookup_by_number(signal[1:], "en")
        else:
            entry = self._defs.lookup_by_term(signal, lang)
            if entry is None and lang != "en":
                entry = self._defs.lookup_by_term(signal, "en")

        if entry is None:
            return None  # fall through to RAG

        answer = (
            f"**Article 4({entry['number']}) — Definition of '{entry['term']}'**\n\n"
            f"{entry['text']}"
        )
        return QueryResult(answer=answer, sources=[_art4_source(lang, entry)])

    # ------------------------------------------------------------------
    # Cross-reference expansion
    # ------------------------------------------------------------------

    def _expand_cross_references(
        self,
        source_nodes,
        language: Optional[str],
        limit: int,
        depth: int = 1,
        _seen: Optional[set] = None,
        _seen_annexes: Optional[set] = None,
        source_limit: int = 2,
        use_graph: bool = False,
    ) -> list:
        """Fetch articles and annexes referenced by retrieved nodes that aren't already in the result set.

        Args:
            depth: How many hops to follow (1 = single-pass, 2 = also expand refs of refs).
            _seen: Internal set of already-retrieved article numbers (used for recursion).
            _seen_annexes: Internal set of already-retrieved annex IDs (used for recursion).
            source_limit: Only collect cross-refs from the top-N primary nodes (default 2).
                Expanding from all nodes can inject low-relevance refs from weakly-scored
                articles; restricting to the strongest evidence keeps expansions targeted.
        """
        if limit <= 0 or depth <= 0:
            return []

        if _seen is None:
            _seen = {node.node.metadata.get("article", "") for node in source_nodes
                     if node.node.metadata.get("article")}

        # If no language filter was passed, infer from the majority language of the
        # source nodes so cross-ref expansions stay language-consistent.
        if language is None and source_nodes:
            from collections import Counter
            langs = [n.node.metadata.get("language") for n in source_nodes
                     if n.node.metadata.get("language")]
            if langs:
                language = Counter(langs).most_common(1)[0][0]

        # Determine candidate article numbers to fetch.
        # Use ArticleGraph BFS only for multi-hop queries (use_graph=True).
        # Single-article queries fall through to CSV fallback to avoid injecting
        # noise from tangentially cross-referenced articles.
        graph = self._ensure_graph() if use_graph else None
        if graph is not None and graph.is_built:
            seed_arts = []
            for node in source_nodes[:source_limit]:
                art = node.node.metadata.get("article", "")
                if art and art not in seed_arts:
                    seed_arts.append(art)
            candidates = graph.bfs_expand(
                seeds=seed_arts,
                max_depth=depth,
                budget=limit,
                exclude=_seen,
            )
            logger.debug("ArticleGraph BFS from %s → %d candidates", seed_arts, len(candidates))
        else:
            # Fallback: collect from referenced_articles CSV, sort deterministically.
            refs_to_fetch: set[str] = set()
            for node in source_nodes[:source_limit]:
                csv = node.node.metadata.get("referenced_articles", "")
                if csv:
                    for ref in csv.split(","):
                        ref = ref.strip()
                        if ref and ref not in _seen:
                            refs_to_fetch.add(ref)
            candidates = sorted(refs_to_fetch, key=_ref_sort_key)

        # Fetch candidate articles — in parallel for speed.
        expanded: list = []
        to_fetch = [r for r in candidates if r not in _seen and r != "4"][:limit]
        if to_fetch:
            from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed

            def _fetch_one(ref_art: str):
                conditions = [("article", ref_art)]
                if language:
                    conditions.append(("language", language))
                if USE_PARAGRAPH_CHUNKING or USE_MIXED_CHUNKING:
                    conditions.append(("chunk_type", "ARTICLE"))
                return ref_art, self._fetch_nodes_direct(conditions, top_k=1)

            max_workers = min(3, len(to_fetch))
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(_fetch_one, r): r for r in to_fetch}
                fetch_results: dict[str, list] = {}
                for fut in _as_completed(futures):
                    ref_art = futures[fut]
                    try:
                        _, nodes = fut.result()
                        fetch_results[ref_art] = nodes
                    except Exception as exc:
                        logger.warning("Cross-ref expansion failed for Article %s: %s", ref_art, exc)
                        fetch_results[ref_art] = []

            # Re-order results to match candidate priority order and apply limit
            for ref_art in to_fetch:
                if len(expanded) >= limit:
                    break
                nodes = fetch_results.get(ref_art, [])
                expanded.extend(nodes)
                _seen.add(ref_art)

        if "4" in candidates:
            _seen.add("4")

        logger.info(
            "Cross-reference expansion (depth=%d, graph=%s): fetched %d additional nodes.",
            depth, graph is not None and graph.is_built, len(expanded),
        )

        # Annex expansion
        if _seen_annexes is None:
            _seen_annexes = {
                node.node.metadata.get("annex_id", "")
                for node in source_nodes
                if node.node.metadata.get("annex_id")
            }
        annex_refs_to_fetch: set[str] = set()
        for node in source_nodes[:source_limit]:
            csv = node.node.metadata.get("referenced_annexes", "")
            for ref in csv.split(","):
                ref = ref.strip().upper()
                if ref and ref not in _seen_annexes:
                    annex_refs_to_fetch.add(ref)
        for ref_anx in [x for x in _ROMAN_ORDER if x in annex_refs_to_fetch]:
            if len(expanded) >= limit:
                break
            try:
                conditions = [("level", "ANNEX"), ("annex_id", ref_anx)]
                if language:
                    conditions.append(("language", language))
                results = self._fetch_nodes_direct(conditions, top_k=1)
                expanded.extend(results)
                _seen_annexes.add(ref_anx)
            except Exception as exc:
                logger.warning("Cross-ref expansion failed for Annex %s: %s", ref_anx, exc)

        # Recursive second-hop expansion (fallback path only — ArticleGraph handles depth natively).
        if depth > 1 and expanded and (graph is None or not graph.is_built):
            remaining = max(0, limit - len(expanded))
            second_hop = self._expand_cross_references(
                expanded, language=language, limit=remaining, depth=depth - 1,
                _seen=_seen, _seen_annexes=_seen_annexes,
            )
            expanded.extend(second_hop)

        return expanded

    def get_article(
        self, article_num: str, language: Optional[str] = None
    ) -> Optional[dict]:
        """Return full article content + metadata for the document viewer.

        Retrieves all nodes for the given article number and concatenates
        their text to produce the complete article body.

        Args:
            article_num: Article number as string (e.g. "92").
            language: Optional ISO language code to filter by (e.g. "en").

        Returns:
            Dict with article data, or None if not found / index not loaded.
        """
        if self._vector_index is None:
            return None

        filters_list: list = [
            MetadataFilter(key="article", value=article_num, operator=FilterOperator.EQ),
        ]
        if language:
            filters_list.append(
                MetadataFilter(key="language", value=language, operator=FilterOperator.EQ)
            )

        nodes = self._retrieve_with_filters(
            filters=MetadataFilters(filters=filters_list),
            query_str=f"Article {article_num}",
            top_k=20,
        )

        if not nodes:
            return None

        # Deduplicate by LlamaIndex internal node_id (guards against the same
        # Qdrant record being returned twice when HYBRID and DEFAULT modes both
        # match, or when the collection contains duplicate points from a
        # previous ingest without --reset).
        seen_ids: set[str] = set()
        unique_nodes = []
        for node in nodes:
            nid = node.node.node_id
            if nid not in seen_ids:
                seen_ids.add(nid)
                unique_nodes.append(node)
        nodes = unique_nodes

        full_text = "\n\n".join(
            node.node.get_content()
            for node in nodes
            if node.node.get_content().strip()
        )

        meta = nodes[0].node.metadata
        ref_csv: str = meta.get("referenced_articles", "") or ""
        referenced_articles = [r.strip() for r in ref_csv.split(",") if r.strip()]
        ext_csv: str = meta.get("referenced_external", "") or ""
        referenced_external = [r.strip() for r in ext_csv.split(",") if r.strip()]

        return {
            "article": article_num,
            "article_title": meta.get("article_title", ""),
            "text": full_text,
            "part": meta.get("part"),
            "title": meta.get("title"),
            "chapter": meta.get("chapter"),
            "section": meta.get("section"),
            "referenced_articles": referenced_articles,
            "referenced_external": referenced_external,
            "language": meta.get("language") or language or "en",
        }

    def get_citing_articles(
        self, article_num: str, language: Optional[str] = None
    ) -> list[dict]:
        """Return articles that reference the given article number.

        Scans all document payloads and returns those whose referenced_articles
        CSV field contains article_num as an exact token (not a substring), so
        "92" does not match "192" or "920".

        Args:
            article_num: Article number to look up (e.g. "92").
            language: Optional ISO language code to restrict results.

        Returns:
            List of citing article dicts sorted by article number, or empty list
            if the index is not loaded.
        """
        if self._vector_index is None:
            return []

        payloads = self.vector_store.scroll_payloads(language=language)

        results: list[dict] = []
        seen: set[str] = set()
        for payload in payloads:
            csv = payload.get("referenced_articles", "")
            if not csv:
                continue
            tokens = {t.strip() for t in csv.split(",") if t.strip()}
            if article_num not in tokens:
                continue
            citing_art = payload.get("article", "")
            if not citing_art or citing_art in seen:
                continue
            seen.add(citing_art)
            results.append({
                "article": citing_art,
                "article_title": payload.get("article_title", ""),
                "part": payload.get("part") or None,
                "title": payload.get("title") or None,
                "chapter": payload.get("chapter") or None,
                "section": payload.get("section") or None,
                "language": payload.get("language") or language or "en",
            })

        results.sort(key=lambda x: int(re.sub(r"[^0-9]", "", x["article"]) or "0"))
        logger.info(
            "Reverse reference lookup: Article %s cited by %d articles (language=%s)",
            article_num, len(results), language,
        )
        return results

    def _direct_article_retrieve(
        self, article_num: str, query_str: str, language: Optional[str]
    ) -> list:
        """Direct Qdrant scroll for a specific article — no embedding required.

        Uses _fetch_nodes_direct so BGE-M3 encoding is skipped entirely, which
        avoids _encode_lock contention and returns all chunks of the article
        (not just the top-k by similarity score).
        """
        conditions: list[tuple[str, str]] = [("article", article_num)]
        if language:
            conditions.append(("language", language))
        if USE_PARAGRAPH_CHUNKING and not USE_MIXED_CHUNKING:
            conditions.append(("chunk_type", "PARAGRAPH"))
        elif not USE_PARAGRAPH_CHUNKING:
            # Default and mixed modes: fetch the ARTICLE chunk for full context
            conditions.append(("chunk_type", "ARTICLE"))
        return self._fetch_nodes_direct(conditions, top_k=50)

    # ------------------------------------------------------------------
    # ToC routing support
    # ------------------------------------------------------------------

    @property
    def toc_store(self):
        """Return the TocStore instance, or None if not loaded."""
        return self._toc

    def toc_retrieve(
        self,
        article_numbers: list[str],
        query_str: str,
        language: Optional[str] = None,
    ) -> list[NodeWithScore]:
        """Fetch nodes for specific article numbers via metadata-filtered retrieval.

        Used by the orchestrator after ToC routing predicts candidate articles.
        Each article is retrieved independently (top_k=3 paragraphs) then merged
        and deduplicated by node_id.
        """
        seen: set[str] = set()
        results: list[NodeWithScore] = []

        for article_num in article_numbers:
            filter_conditions = [
                MetadataFilter(
                    key="article", value=article_num, operator=FilterOperator.EQ
                )
            ]
            if language:
                filter_conditions.append(
                    MetadataFilter(
                        key="language", value=language, operator=FilterOperator.EQ
                    )
                )
            filters = MetadataFilters(filters=filter_conditions)
            nodes = self._retrieve_with_filters(filters, query_str, top_k=3)
            for node in nodes:
                nid = node.node.node_id
                if nid not in seen:
                    seen.add(nid)
                    results.append(node)

        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fetch_nodes_direct(
        self,
        conditions: list[tuple[str, str]],
        top_k: int = 1,
    ) -> list[NodeWithScore]:
        """Fetch nodes from Qdrant using only payload filters — no embedding required.

        Used for cross-reference expansion where the article number is already known,
        so semantic search adds no value and BGE-M3 encoding would just waste CPU time
        and queue on the _encode_lock under parallel load.

        Args:
            conditions: list of (field_name, value) pairs combined with AND logic.
            top_k: maximum number of results to return.
        """
        import json as _json
        from qdrant_client.http.models import FieldCondition, Filter, MatchValue

        client = self.vector_store._client
        if client is None:
            return []

        must = [FieldCondition(key=k, match=MatchValue(value=v)) for k, v in conditions]
        try:
            records, _ = client.scroll(
                collection_name=self.vector_store.collection_name,
                scroll_filter=Filter(must=must),
                with_payload=True,
                with_vectors=False,
                limit=top_k,
            )
        except Exception as exc:
            logger.warning("_fetch_nodes_direct failed (conditions=%s): %s", conditions, exc)
            return []

        results: list[NodeWithScore] = []
        for record in records:
            payload = record.payload or {}
            text = payload.get("text", "") or ""
            if not text:
                raw = payload.get("_node_content", "")
                if raw:
                    try:
                        text = _json.loads(raw).get("text", "") or ""
                    except Exception:
                        pass
            metadata = {k: v for k, v in payload.items() if not k.startswith("_")}
            node = TextNode(id_=str(record.id), text=text, metadata=metadata)
            results.append(NodeWithScore(node=node, score=1.0))
        return results

    def _retrieve_with_filters(
        self,
        filters: MetadataFilters,
        query_str: str,
        top_k: int,
    ) -> list:
        """Retrieve nodes with metadata filters, trying HYBRID first then DEFAULT.

        Qdrant HYBRID mode can return empty results when a metadata filter is
        highly selective (e.g. exact article match) because the sparse ANN index
        finds no approximate neighbours under tight constraints.  Falling back to
        DEFAULT (dense-only) applies the filter as a post-filter and reliably
        returns the expected nodes.

        DEFAULT is tried first because metadata-filtered lookups (direct article,
        cross-reference expansion, article viewer) always have tight filters that
        cause HYBRID mode to return empty results (sparse ANN finds no neighbours
        under exact-match constraints) before falling through to DEFAULT anyway.
        Trying DEFAULT first eliminates the wasted sparse_query_fn call and the
        redundant Qdrant round-trip that HYBRID-first imposed on every such lookup.
        HYBRID is kept as a fallback so semantically-loose queries still benefit
        from sparse recall if DEFAULT unexpectedly returns nothing.
        """
        for mode in (VectorStoreQueryMode.DEFAULT, VectorStoreQueryMode.HYBRID):
            try:
                retriever = self._vector_index.as_retriever(
                    similarity_top_k=top_k,
                    vector_store_query_mode=mode,
                    filters=filters,
                    alpha=RETRIEVAL_ALPHA,
                )
                results = retriever.retrieve(query_str)
            except Exception as exc:
                logger.warning(
                    "Retrieval mode=%s failed (%s) — trying fallback", mode, exc
                )
                results = []
            if results:
                return results
        return []

    def _configure_settings(self) -> None:
        Settings.embed_model = BGEm3Embedding()
        Settings.llm = OpenAI(model=self.llm_model, api_key=self.openai_api_key, timeout=120.0)
        # Invalidate any stale PromptHelper that was cached by a prior code path
        # (e.g. the indexer sets Settings.llm = None which creates a small context window).
        # Resetting to None forces LlamaIndex to rebuild it from the current LLM metadata.
        Settings._prompt_helper = None

    def _build_engine(
        self,
        vector_index: VectorStoreIndex,
        language_filter: Optional[str] = None,
    ) -> RetrieverQueryEngine:
        # Optional metadata filters: language + chunk_type
        # Mixed mode: no chunk_type filter — ARTICLE and PARAGRAPH chunks compete freely.
        # Paragraph mode: filter to PARAGRAPH only; article text is fetched at synthesis time.
        # Default mode: filter to ARTICLE only; paragraph chunks are ignored.
        filter_list = []
        if language_filter:
            filter_list.append(
                MetadataFilter(key="language", value=language_filter, operator=FilterOperator.EQ)
            )
        if USE_PARAGRAPH_CHUNKING and not USE_MIXED_CHUNKING:
            filter_list.append(
                MetadataFilter(key="chunk_type", value="PARAGRAPH", operator=FilterOperator.EQ)
            )
        elif not USE_PARAGRAPH_CHUNKING and not USE_MIXED_CHUNKING:
            filter_list.append(
                MetadataFilter(key="chunk_type", value="ARTICLE", operator=FilterOperator.EQ)
            )
        # USE_MIXED_CHUNKING=true: no chunk_type filter added
        filters = MetadataFilters(filters=filter_list) if filter_list else None

        # Articles are self-contained units — no AutoMergingRetriever needed
        vector_retriever = vector_index.as_retriever(
            similarity_top_k=RETRIEVAL_TOP_K,
            vector_store_query_mode=VectorStoreQueryMode.HYBRID,
            filters=filters,
            alpha=RETRIEVAL_ALPHA,
        )

        synthesizer = get_response_synthesizer(
            text_qa_template=_LEGAL_QA_TEMPLATE,
            verbose=False,
        )

        # Postprocessors: similarity filter → [deduplicator] → reranker → title boost → tiebreaker
        # ArticleDeduplicatorPostprocessor is inserted in mixed-chunking mode to ensure at most
        # one chunk per article enters the reranker, preventing flooding of the top-k.
        # ParagraphWindowReranker takes precedence over the standard reranker when both are enabled.
        # When title boost is active and no paragraph reranker, the reranker is widened to
        # pass_through (top_n=RETRIEVAL_TOP_K) so that title boost makes the final truncation.
        postprocessors = [SimilarityPostprocessor(similarity_cutoff=SIMILARITY_CUTOFF)]
        if USE_MIXED_CHUNKING:
            postprocessors.append(ArticleDeduplicatorPostprocessor())
        para_reranker = getattr(self, "_para_reranker", None)
        title_booster = getattr(self, "_title_booster", None)
        tiebreaker = getattr(self, "_tiebreaker", None)
        if para_reranker is not None:
            # Paragraph-window reranker handles blending and truncation itself
            postprocessors.append(para_reranker)
            if title_booster is not None:
                postprocessors.append(title_booster)
        elif self._reranker is not None:
            if title_booster is not None:
                # Widen reranker window so title boost can reshuffle all scored nodes
                from llama_index.core.postprocessor import SentenceTransformerRerank
                wide_reranker = SentenceTransformerRerank(
                    top_n=RETRIEVAL_TOP_K, model=RERANKER_MODEL
                )
                postprocessors.append(wide_reranker)
                postprocessors.append(title_booster)
            else:
                postprocessors.append(self._reranker)
        elif title_booster is not None:
            postprocessors.append(title_booster)
        if tiebreaker is not None:
            postprocessors.append(tiebreaker)

        return RetrieverQueryEngine.from_args(
            retriever=vector_retriever,
            response_synthesizer=synthesizer,
            node_postprocessors=postprocessors,
            verbose=False,
        )
