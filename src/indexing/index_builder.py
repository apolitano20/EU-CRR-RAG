"""
HierarchicalIndexer: indexes article-level Documents into a Qdrant hybrid vector index.

Each Document corresponds to one Article or Annex section — no sub-article chunking.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Optional

from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex

from src.indexing.bge_m3_sparse import BGEm3Embedding
from src.indexing.vector_store import VectorStore

logger = logging.getLogger(__name__)


@contextmanager
def _settings_scope():
    """Snapshot and restore LlamaIndex global Settings on exit.

    Prevents indexer-specific values (embed_model, llm=None, transformations=[],
    chunk_size, chunk_overlap) from leaking into other components that share the
    same process — e.g. QueryEngine, which needs Settings.llm = OpenAI(...).
    """
    # embed_model / llm / transformations have lazy resolvers that try to import
    # llama-index-embeddings-openai (not installed in Colab) — bypass via getattr.
    # chunk_size / chunk_overlap: safe in most versions, but some LlamaIndex builds
    # delegate these to node_parser which in turn calls embed_model. Guard with a
    # sentinel so the finally block skips restoration when the read itself fails.
    _UNSET = object()
    prev_embed = getattr(Settings, "_embed_model", None)
    prev_llm = getattr(Settings, "_llm", None)
    prev_transforms = getattr(Settings, "_transformations", None)
    try:
        prev_chunk_size: object = Settings.chunk_size
        prev_chunk_overlap: object = Settings.chunk_overlap
    except Exception:
        prev_chunk_size = _UNSET
        prev_chunk_overlap = _UNSET
    try:
        yield
    finally:
        Settings._embed_model = prev_embed
        Settings._llm = prev_llm
        Settings._transformations = prev_transforms
        if prev_chunk_size is not _UNSET:
            Settings.chunk_size = prev_chunk_size  # type: ignore[assignment]
        if prev_chunk_overlap is not _UNSET:
            Settings.chunk_overlap = prev_chunk_overlap  # type: ignore[assignment]


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
        with _settings_scope():
            self._configure_settings()

            if self.reset_store:
                self.vector_store.reset()
            else:
                self.vector_store.connect()

            logger.info("Indexing %d documents (article-level).", len(documents))
            storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store.as_llama_vector_store()
            )
            # Use the constructor directly instead of from_documents() to bypass the
            # transformation pipeline entirely. from_documents() has a falsy-empty-list
            # trap: `transformations = transformations or Settings.transformations` means
            # passing `transformations=[]` falls through to Settings.transformations which
            # defaults to [SentenceSplitter()]. The constructor path goes straight to
            # _build_index_from_nodes → _add_nodes_to_index → embed + upsert, with no
            # chunking at all. Document is a BaseNode subclass so this is safe.
            self._vector_index = VectorStoreIndex(
                nodes=documents,  # type: ignore[arg-type]  Document is a BaseNode subclass
                storage_context=storage_context,
                show_progress=True,
            )
        logger.info("Vector index built with %d items.", self.vector_store.item_count)
        return self._vector_index

    def load(self) -> VectorStoreIndex:
        """Re-open an existing persisted index (no re-ingestion)."""
        with _settings_scope():
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
