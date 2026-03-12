"""
QueryEngine: hybrid retrieval (AutoMergingRetriever + BM25 via QueryFusionRetriever)
followed by GPT-4o synthesis.
"""
from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import Optional

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import AutoMergingRetriever, QueryFusionRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.retrievers.bm25 import BM25Retriever

from src.indexing.index_builder import HierarchicalIndexer
from src.indexing.vector_store import VectorStore

logger = logging.getLogger(__name__)

SIMILARITY_TOP_K = 12
FUSION_TOP_K = 5
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "gpt-4o"


@dataclass
class QueryResult:
    answer: str
    sources: list[dict]
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))


class QueryEngine:
    """Wraps hybrid retrieval + GPT-4o synthesis into a single query interface."""

    def __init__(
        self,
        vector_store: VectorStore,
        indexer: HierarchicalIndexer,
        openai_api_key: Optional[str] = None,
        llm_model: str = LLM_MODEL,
    ) -> None:
        self.vector_store = vector_store
        self.indexer = indexer
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.llm_model = llm_model
        self._engine: Optional[RetrieverQueryEngine] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Build the retriever chain from persisted indices."""
        vector_index = self.indexer.load()   # may set Settings.llm = None as side effect
        self._configure_settings()           # restore OpenAI LLM after indexer resets it
        self._engine = self._build_engine(vector_index)
        logger.info("QueryEngine ready.")

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, user_query: str) -> QueryResult:
        if self._engine is None:
            raise RuntimeError("Call load() before querying.")

        trace_id = str(uuid.uuid4())
        logger.info("[%s] Query: %s", trace_id, user_query)

        response = self._engine.query(user_query)

        sources = [
            {
                "text": node.node.get_content()[:500],
                "score": round(node.score or 0.0, 4),
                "metadata": node.node.metadata,
            }
            for node in (response.source_nodes or [])
        ]

        logger.info(
            "[%s] Retrieved %d source nodes; answer length=%d chars.",
            trace_id, len(sources), len(str(response)),
        )

        return QueryResult(answer=str(response), sources=sources, trace_id=trace_id)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _configure_settings(self) -> None:
        Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
        Settings.llm = OpenAI(model=self.llm_model, api_key=self.openai_api_key)

    def _build_engine(self, vector_index: VectorStoreIndex) -> RetrieverQueryEngine:
        storage_context = vector_index.storage_context

        # Vector leg: AutoMergingRetriever climbs the node tree to return full parent context
        vector_retriever = vector_index.as_retriever(similarity_top_k=SIMILARITY_TOP_K)
        auto_merging_retriever = AutoMergingRetriever(
            vector_retriever,
            storage_context,
            verbose=False,
        )

        # BM25 leg: keyword-based retrieval on leaf nodes
        bm25_retriever = self.indexer.load_bm25_retriever(similarity_top_k=SIMILARITY_TOP_K)

        # Fusion: Reciprocal Rank Fusion over both legs
        fusion_retriever = QueryFusionRetriever(
            retrievers=[auto_merging_retriever, bm25_retriever],
            similarity_top_k=FUSION_TOP_K,
            num_queries=1,  # no query augmentation; use the original query only
            mode="reciprocal_rerank",
            verbose=False,
        )

        return RetrieverQueryEngine.from_args(
            retriever=fusion_retriever,
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.0)],
            verbose=False,
        )
