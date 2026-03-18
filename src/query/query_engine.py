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
RETRIEVAL_ALPHA: float = float(os.getenv("RETRIEVAL_ALPHA", "0.5"))

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


def _normalise_query(query: str) -> str:
    """Expand CRR abbreviations, canonicalise article shorthand references, and expand ranges."""
    query = _ABBREV_RE.sub(lambda m: f"{m.group(1)} ({_ABBREV_MAP[m.group(1)]})", query)
    query = _ART_RE.sub(lambda m: f"Article {m.group(1)}", query)
    query = _expand_article_ranges(query)
    return query


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

        # Deduplicate expanded nodes against primary results so the LLM doesn't
        # see the same article twice, then merge for synthesis.
        source_ids = {node.node.node_id for node in source_nodes}
        deduped_expanded = [n for n in expanded_nodes if n.node.node_id not in source_ids]
        all_nodes_for_synthesis = source_nodes + deduped_expanded

        # Stage 3: LLM synthesis over primary + expanded context
        t_syn = time.perf_counter()
        response = engine.synthesize(query_bundle, all_nodes_for_synthesis)
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
            for node in deduped_expanded
        ]

        latency_ms = round((time.perf_counter() - t0) * 1000)

        logger.info(
            "[%s] Retrieved %d source nodes + %d expanded (%d synthesized); "
            "answer=%d chars; latency=%dms "
            "(retrieval=%dms, expansion=%dms, synthesis=%dms)",
            trace_id,
            len(source_nodes),
            len(deduped_expanded),
            len(all_nodes_for_synthesis),
            len(str(response)),
            latency_ms,
            t_retrieval_ms,
            t_expand_ms,
            t_synthesis_ms,
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
        _seen_annexes: Optional[set] = None,
    ) -> list:
        """Fetch articles and annexes referenced by retrieved nodes that aren't already in the result set.

        Args:
            depth: How many hops to follow (1 = single-pass, 2 = also expand refs of refs).
            _seen: Internal set of already-retrieved article numbers (used for recursion).
            _seen_annexes: Internal set of already-retrieved annex IDs (used for recursion).
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

        # Collect referenced articles from all source node metadata
        refs_to_fetch: set[str] = set()
        for node in source_nodes:
            csv = node.node.metadata.get("referenced_articles", "")
            if csv:
                for ref in csv.split(","):
                    ref = ref.strip()
                    if ref and ref not in _seen:
                        refs_to_fetch.add(ref)

        # Sort deterministically (numeric part first, alpha suffix second) so the
        # chosen subset is stable across runs and independent of Python hash order.
        candidates = sorted(refs_to_fetch, key=_ref_sort_key)
        expanded: list = []
        for ref_art in candidates:
            if len(expanded) >= limit:
                break
            try:
                filters_list = [
                    MetadataFilter(key="article", value=ref_art, operator=FilterOperator.EQ),
                ]
                if language:
                    filters_list.append(
                        MetadataFilter(key="language", value=language, operator=FilterOperator.EQ)
                    )
                results = self._retrieve_with_filters(
                    filters=MetadataFilters(filters=filters_list),
                    query_str=f"Article {ref_art}",
                    top_k=1,
                )
                expanded.extend(results)
                _seen.add(ref_art)
            except Exception as exc:
                # A failed fetch does NOT consume a cap slot — continue to next candidate.
                logger.warning("Cross-ref expansion failed for Article %s: %s", ref_art, exc)

        logger.info(
            "Cross-reference expansion (depth=%d): fetched %d additional nodes.",
            depth, len(expanded),
        )

        # Annex expansion
        if _seen_annexes is None:
            _seen_annexes = {
                node.node.metadata.get("annex_id", "")
                for node in source_nodes
                if node.node.metadata.get("annex_id")
            }
        annex_refs_to_fetch: set[str] = set()
        for node in source_nodes:
            csv = node.node.metadata.get("referenced_annexes", "")
            for ref in csv.split(","):
                ref = ref.strip().upper()
                if ref and ref not in _seen_annexes:
                    annex_refs_to_fetch.add(ref)
        for ref_anx in [x for x in _ROMAN_ORDER if x in annex_refs_to_fetch]:
            if len(expanded) >= limit:
                break
            try:
                filters_list = [
                    MetadataFilter(key="level", value="ANNEX", operator=FilterOperator.EQ),
                    MetadataFilter(key="annex_id", value=ref_anx, operator=FilterOperator.EQ),
                ]
                if language:
                    filters_list.append(
                        MetadataFilter(key="language", value=language, operator=FilterOperator.EQ)
                    )
                results = self._retrieve_with_filters(
                    filters=MetadataFilters(filters=filters_list),
                    query_str=f"Annex {ref_anx}",
                    top_k=1,
                )
                expanded.extend(results)
                _seen_annexes.add(ref_anx)
            except Exception as exc:
                logger.warning("Cross-ref expansion failed for Annex %s: %s", ref_anx, exc)

        # Recursive second-hop expansion
        if depth > 1 and expanded:
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
        """Metadata-filtered retrieval for a specific article number, bypassing vector ranking."""
        filters_list = [
            MetadataFilter(key="article", value=article_num, operator=FilterOperator.EQ),
        ]
        if language:
            filters_list.append(
                MetadataFilter(key="language", value=language, operator=FilterOperator.EQ)
            )
        return self._retrieve_with_filters(
            filters=MetadataFilters(filters=filters_list),
            query_str=query_str,
            top_k=10,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

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

        Using HYBRID as the first attempt means queries that do work in HYBRID
        mode (e.g. cross-reference expansion with looser filters) benefit from
        sparse recall without any code changes.
        """
        for mode in (VectorStoreQueryMode.HYBRID, VectorStoreQueryMode.DEFAULT):
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
            alpha=RETRIEVAL_ALPHA,
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
