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

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.schema import QueryBundle
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

RETRIEVAL_TOP_K = 12       # first-stage candidates passed to reranker
RERANK_TOP_N = 6           # final results after reranking
SIMILARITY_CUTOFF = 0.3
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
LLM_MODEL = "gpt-4o"

_LEGAL_QA_TEMPLATE = PromptTemplate(
    "You are a regulatory compliance expert specialising in EU prudential banking regulation "
    "(CRR – Regulation (EU) No 575/2013).\n\n"
    "Use ONLY the context below to answer the question. "
    "Do not speculate or introduce information not present in the context. "
    "If the context does not contain enough information to answer, respond with ONLY the "
    "following sentence: 'The provided context does not contain sufficient information to "
    "answer this question.'\n\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n\n"
    "Question: {query_str}\n\n"
    "Respond using the following structure. Omit any section that is not applicable.\n\n"
    "**Direct Answer**\n"
    "A concise answer to the question in 1–3 sentences.\n\n"
    "**Key Provisions**\n"
    "Bullet-point summary of the relevant rules, thresholds, or definitions drawn from the context. "
    "Each bullet must reference the Article it comes from (e.g. '- Article 92(1)(a): …').\n\n"
    "**Conditions, Exceptions & Definitions**\n"
    "Any qualifications, carve-outs, special treatments, or defined terms that affect the answer.\n\n"
    "**Article References**\n"
    "Comma-separated list of every Article cited above (e.g. 'Article 92, Article 93, Article 128').\n\n"
    "Answer:"
)


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
}
_ABBREV_RE = re.compile(r"\b(" + "|".join(re.escape(k) for k in _ABBREV_MAP) + r")\b")

# Normalise shorthand article references to the canonical "Article N" form used in metadata.
# Handles: "art. 92", "Art 92", "article92", "§92" → "Article 92"
_ART_RE = re.compile(r"\b(?:art(?:icle|\.)?)\s*(\d[\w]*)", re.IGNORECASE)

# Matches any canonical "Article N" reference after query normalisation.
_ARTICLE_REF_RE = re.compile(r"\bArticle\s+(\d[\w]*)\b", re.IGNORECASE)


def _normalise_query(query: str) -> str:
    """Expand CRR abbreviations and canonicalise article shorthand references."""
    query = _ABBREV_RE.sub(lambda m: f"{m.group(1)} ({_ABBREV_MAP[m.group(1)]})", query)
    return _ART_RE.sub(lambda m: f"Article {m.group(1)}", query)


def _detect_direct_article_lookup(query: str) -> Optional[str]:
    """Return article number if the query references exactly one article, else None.

    Handles any phrasing that mentions a single article:
      - "What are the requirements of Article 73 of the CRR?"
      - "Explain Article 92"
      - "Does Article 428 apply to investment firms?"
    Does NOT trigger when multiple distinct articles are mentioned, so that
    cross-article questions ("How do Article 92 and 93 relate?") use normal retrieval.
    """
    matches = _ARTICLE_REF_RE.findall(query)
    unique = set(matches)
    return unique.pop() if len(unique) == 1 else None


@dataclass
class QueryResult:
    answer: str
    sources: list[dict]
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))


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
        self._token_counter: Optional[TokenCountingHandler] = None
        self._reranker = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Build the retriever chain from the persisted index."""
        self._vector_index = self.indexer.load()  # may set Settings.llm = None as side effect
        self._configure_settings()                 # restore OpenAI LLM after indexer resets it
        if self.use_reranker:
            from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
            self._reranker = FlagEmbeddingReranker(top_n=RERANK_TOP_N, model=RERANKER_MODEL)
            logger.info("Reranker loaded: %s (top_n=%d)", RERANKER_MODEL, RERANK_TOP_N)
        self._engine = self._build_engine(self._vector_index)
        with self._engine_cache_lock:
            self._engine_cache = {}
        logger.info("QueryEngine ready.")

    def is_loaded(self) -> bool:
        return self._engine is not None

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        user_query: str,
        language: Optional[str] = None,
        max_cross_ref_expansions: Optional[int] = None,
    ) -> QueryResult:
        if self._engine is None:
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
            if language not in self._engine_cache:
                with self._engine_cache_lock:
                    if language not in self._engine_cache:
                        self._engine_cache[language] = self._build_engine(
                            self._vector_index, language_filter=language
                        )
            engine = self._engine_cache[language]
        else:
            engine = self._engine

        # Per-request token counter (avoids mixing counts across concurrent requests)
        token_counter = TokenCountingHandler(verbose=False)
        Settings.callback_manager = CallbackManager([token_counter])

        query_bundle = QueryBundle(query_str=user_query)

        # Stage 1: Retrieval (includes postprocessors: similarity cutoff + reranker if enabled)
        t_ret = time.perf_counter()
        if direct_art:
            source_nodes = self._direct_article_retrieve(direct_art, user_query, language)
            if not source_nodes:
                logger.warning("[%s] Direct lookup for Article %s returned no nodes — falling back to semantic retrieval", trace_id, direct_art)
                source_nodes = engine.retrieve(query_bundle)
        else:
            source_nodes = engine.retrieve(query_bundle)
        t_retrieval_ms = round((time.perf_counter() - t_ret) * 1000)

        # Cross-lingual fallback: if language filter produced no results, retry without filter
        if language and not source_nodes:
            logger.info("[%s] No results for language=%s — retrying without language filter", trace_id, language)
            source_nodes = self._engine.retrieve(query_bundle)

        # Stage 2: Cross-reference expansion
        expansions = max_cross_ref_expansions if max_cross_ref_expansions is not None \
            else self.max_cross_ref_expansions
        t_exp = time.perf_counter()
        expanded_nodes = self._expand_cross_references(source_nodes, language=language, limit=expansions)
        t_expand_ms = round((time.perf_counter() - t_exp) * 1000)

        # Stage 3: LLM synthesis
        t_syn = time.perf_counter()
        response = engine.synthesize(query_bundle, source_nodes)
        t_synthesis_ms = round((time.perf_counter() - t_syn) * 1000)

        sources = [
            {
                "text": node.node.get_content()[:500],
                "score": round(node.score or 0.0, 4),
                "metadata": node.node.metadata,
                "expanded": False,
            }
            for node in source_nodes
        ]
        sources += [
            {
                "text": node.node.get_content()[:500],
                "score": 0.0,
                "metadata": node.node.metadata,
                "expanded": True,
            }
            for node in expanded_nodes
        ]

        latency_ms = round((time.perf_counter() - t0) * 1000)
        prompt_tokens = token_counter.prompt_llm_token_count
        completion_tokens = token_counter.completion_llm_token_count

        logger.info(
            "[%s] Retrieved %d source nodes + %d expanded; "
            "answer=%d chars; latency=%dms "
            "(retrieval=%dms, expansion=%dms, synthesis=%dms); "
            "tokens=prompt:%d,completion:%d",
            trace_id,
            len(source_nodes),
            len(expanded_nodes),
            len(str(response)),
            latency_ms,
            t_retrieval_ms,
            t_expand_ms,
            t_synthesis_ms,
            prompt_tokens,
            completion_tokens,
        )

        return QueryResult(answer=str(response), sources=sources, trace_id=trace_id)

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
    ) -> list:
        """Fetch articles referenced by retrieved nodes that aren't already in the result set.

        Args:
            depth: How many hops to follow (1 = single-pass, 2 = also expand refs of refs).
            _seen: Internal set of already-retrieved article numbers (used for recursion).
        """
        if limit <= 0 or depth <= 0:
            return []

        if _seen is None:
            _seen = {node.node.metadata.get("article", "") for node in source_nodes
                     if node.node.metadata.get("article")}

        # Collect referenced articles from all source node metadata
        refs_to_fetch: set[str] = set()
        for node in source_nodes:
            csv = node.node.metadata.get("referenced_articles", "")
            if csv:
                for ref in csv.split(","):
                    ref = ref.strip()
                    if ref and ref not in _seen:
                        refs_to_fetch.add(ref)

        if not refs_to_fetch:
            return []

        expanded: list = []
        for ref_art in list(refs_to_fetch)[:limit]:
            try:
                filters_list = [
                    MetadataFilter(key="article", value=ref_art, operator=FilterOperator.EQ),
                ]
                if language:
                    filters_list.append(
                        MetadataFilter(key="language", value=language, operator=FilterOperator.EQ)
                    )
                retriever = self._vector_index.as_retriever(
                    similarity_top_k=1,
                    vector_store_query_mode=VectorStoreQueryMode.HYBRID,
                    filters=MetadataFilters(filters=filters_list),
                )
                results = retriever.retrieve(f"Article {ref_art}")
                expanded.extend(results)
                _seen.add(ref_art)
            except Exception as exc:
                logger.warning("Cross-ref expansion failed for Article %s: %s", ref_art, exc)

        logger.info(
            "Cross-reference expansion (depth=%d): fetched %d additional nodes.",
            depth, len(expanded),
        )

        # Recursive second-hop expansion
        if depth > 1 and expanded:
            remaining = max(0, limit - len(expanded))
            second_hop = self._expand_cross_references(
                expanded, language=language, limit=remaining, depth=depth - 1, _seen=_seen
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

        # Use DEFAULT (dense-only) — HYBRID with exact metadata filters returns zero
        # results for selective filters (same issue fixed in _direct_article_retrieve).
        retriever = self._vector_index.as_retriever(
            similarity_top_k=20,
            vector_store_query_mode=VectorStoreQueryMode.DEFAULT,
            filters=MetadataFilters(filters=filters_list),
        )
        nodes = retriever.retrieve(f"Article {article_num}")

        if not nodes:
            return None

        full_text = "\n\n".join(
            node.node.get_content()
            for node in nodes
            if node.node.get_content().strip()
        )

        meta = nodes[0].node.metadata
        ref_csv: str = meta.get("referenced_articles", "") or ""
        referenced_articles = [r.strip() for r in ref_csv.split(",") if r.strip()]

        return {
            "article": article_num,
            "article_title": meta.get("article_title", ""),
            "text": full_text,
            "part": meta.get("part"),
            "title": meta.get("title"),
            "chapter": meta.get("chapter"),
            "section": meta.get("section"),
            "referenced_articles": referenced_articles,
            "language": meta.get("language") or language or "en",
        }

    def _direct_article_retrieve(
        self, article_num: str, query_str: str, language: Optional[str]
    ) -> list:
        """Metadata-filtered retrieval for a specific article number, bypassing vector ranking.

        Uses DEFAULT (dense-only) mode because we have an exact metadata filter — hybrid
        sparse retrieval can return zero results when the filter is very selective.
        """
        filters_list = [
            MetadataFilter(key="article", value=article_num, operator=FilterOperator.EQ),
        ]
        if language:
            filters_list.append(
                MetadataFilter(key="language", value=language, operator=FilterOperator.EQ)
            )
        retriever = self._vector_index.as_retriever(
            similarity_top_k=10,
            vector_store_query_mode=VectorStoreQueryMode.DEFAULT,
            filters=MetadataFilters(filters=filters_list),
        )
        return retriever.retrieve(query_str)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _configure_settings(self) -> None:
        Settings.embed_model = BGEm3Embedding()
        Settings.llm = OpenAI(model=self.llm_model, api_key=self.openai_api_key)

    def _build_engine(
        self,
        vector_index: VectorStoreIndex,
        language_filter: Optional[str] = None,
    ) -> RetrieverQueryEngine:
        # Optional metadata filter for language-scoped retrieval
        filters = (
            MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="language",
                        value=language_filter,
                        operator=FilterOperator.EQ,
                    )
                ]
            )
            if language_filter
            else None
        )

        # Articles are self-contained units — no AutoMergingRetriever needed
        vector_retriever = vector_index.as_retriever(
            similarity_top_k=RETRIEVAL_TOP_K,
            vector_store_query_mode=VectorStoreQueryMode.HYBRID,
            filters=filters,
        )

        synthesizer = get_response_synthesizer(
            text_qa_template=_LEGAL_QA_TEMPLATE,
            verbose=False,
        )

        # Postprocessors: similarity filter first, then reranker (if enabled)
        postprocessors = [SimilarityPostprocessor(similarity_cutoff=SIMILARITY_CUTOFF)]
        if self._reranker is not None:
            postprocessors.append(self._reranker)

        return RetrieverQueryEngine.from_args(
            retriever=vector_retriever,
            response_synthesizer=synthesizer,
            node_postprocessors=postprocessors,
            verbose=False,
        )
