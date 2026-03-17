"""
HierarchicalIndexer: indexes article-level Documents into a Qdrant hybrid vector index.

Each Document corresponds to one Article or Annex section — no sub-article chunking.
"""
from __future__ import annotations

import logging
from typing import Optional

from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex

from src.indexing.bge_m3_sparse import BGEm3Embedding
from src.indexing.vector_store import VectorStore

logger = logging.getLogger(__name__)


class HierarchicalIndexer:
    """Builds and persists the vector index from a list of Documents."""

    def __init__(
        self,
        vector_store: VectorStore,
        reset_store: bool = False,
    ) -> None:
        self.vector_store = vector_store
        self.reset_store = reset_store
        self._vector_index: Optional[VectorStoreIndex] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, documents: list[Document]) -> VectorStoreIndex:
        """Index article-level documents. Returns the vector index."""
        self._configure_settings()

        if self.reset_store:
            self.vector_store.reset()
        else:
            self.vector_store.connect()

        logger.info("Indexing %d documents (article-level).", len(documents))
        storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store.as_llama_vector_store()
        )
        self._vector_index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True,
            transformations=[],  # belt-and-suspenders; Settings.transformations=[] is the real guard ([] is falsy so this alone wouldn't work)
        )
        logger.info("Vector index built with %d items.", self.vector_store.item_count)
        return self._vector_index

    def load(self) -> VectorStoreIndex:
        """Re-open an existing persisted index (no re-ingestion)."""
        self._configure_settings()
        self.vector_store.connect()
        self._vector_index = VectorStoreIndex.from_vector_store(
            self.vector_store.as_llama_vector_store()
        )
        logger.info("Loaded existing vector index (%d items).", self.vector_store.item_count)
        return self._vector_index

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _configure_settings(self) -> None:
        Settings.embed_model = BGEm3Embedding()
        Settings.llm = None  # LLM is set per-query in QueryEngine
        # Prevent LlamaIndex from chunking article-level documents.
        # IMPORTANT: `transformations=[]` passed to from_documents() is falsy in Python,
        # so `transformations or Settings.transformations` falls through to the default
        # pipeline which contains SentenceSplitter. Setting Settings.transformations=[]
        # here ensures the fallback is also empty. Embedding still happens in
        # _add_nodes_to_index → _get_node_with_embedding (not in the transformations pipeline).
        Settings.transformations = []
        Settings.chunk_size = 8192
        Settings.chunk_overlap = 0
