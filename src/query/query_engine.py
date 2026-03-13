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
from llama_index.core.vector_stores.types import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQueryMode,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI

from src.indexing.index_builder import HierarchicalIndexer
from src.indexing.vector_store import VectorStore

logger = logging.getLogger(__name__)

RETRIEVAL_TOP_K = 12       # first-stage candidates passed to reranker
RERANK_TOP_N = 6           # final results after reranking
SIMILARITY_CUTOFF = 0.3
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
EMBED_MODEL = "BAAI/bge-m3"
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


# Normalise shorthand article references to the canonical "Article N" form used in metadata.
# Handles: "art. 92", "Art 92", "article92", "§92" → "Article 92"
_ART_RE = re.compile(r"\b(?:art(?:icle|\.)?)\s*(\d[\w]*)", re.IGNORECASE)


def _normalise_query(query: str) -> str:
    return _ART_RE.sub(lambda m: f"Article {m.group(1)}", query)


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

        response = engine.query(user_query)
        source_nodes = response.source_nodes or []

        # Cross-lingual fallback: if language filter produced no results, retry without filter
        if language and not source_nodes:
            logger.info("[%s] No results for language=%s — retrying without language filter", trace_id, language)
            response = self._engine.query(user_query)
            source_nodes = response.source_nodes or []

        # Cross-reference expansion
        expansions = max_cross_ref_expansions if max_cross_ref_expansions is not None \
            else self.max_cross_ref_expansions
        expanded_nodes = self._expand_cross_references(source_nodes, language=language, limit=expansions)

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
            "answer=%d chars; latency=%dms; tokens=prompt:%d,completion:%d",
            trace_id,
            len(source_nodes),
            len(expanded_nodes),
            len(str(response)),
            latency_ms,
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
    ) -> list:
        """Fetch articles referenced by retrieved nodes that aren't already in the result set."""
        if limit <= 0:
            return []

        # Collect already-retrieved article numbers
        retrieved_articles: set[str] = set()
        for node in source_nodes:
            art = node.node.metadata.get("article", "")
            if art:
                retrieved_articles.add(art)

        # Collect referenced articles from all source node metadata
        refs_to_fetch: set[str] = set()
        for node in source_nodes:
            csv = node.node.metadata.get("referenced_articles", "")
            if csv:
                for ref in csv.split(","):
                    ref = ref.strip()
                    if ref and ref not in retrieved_articles:
                        refs_to_fetch.add(ref)

        if not refs_to_fetch:
            return []

        expanded: list = []
        retriever = self._vector_index.as_retriever(
            similarity_top_k=1,
            vector_store_query_mode=VectorStoreQueryMode.HYBRID,
        )

        for ref_art in list(refs_to_fetch)[:limit]:
            try:
                filters = MetadataFilters(filters=[
                    MetadataFilter(key="article", value=ref_art, operator=FilterOperator.EQ),
                ])
                if language:
                    filters.filters.append(
                        MetadataFilter(key="language", value=language, operator=FilterOperator.EQ)
                    )
                lang_retriever = self._vector_index.as_retriever(
                    similarity_top_k=1,
                    vector_store_query_mode=VectorStoreQueryMode.HYBRID,
                    filters=filters,
                )
                results = lang_retriever.retrieve(f"Article {ref_art}")
                expanded.extend(results)
            except Exception as exc:
                logger.warning("Cross-ref expansion failed for Article %s: %s", ref_art, exc)

        logger.info("Cross-reference expansion: fetched %d additional nodes.", len(expanded))
        return expanded

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _configure_settings(self) -> None:
        Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
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
