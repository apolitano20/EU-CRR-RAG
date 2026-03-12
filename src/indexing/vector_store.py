"""
VectorStore: thin wrapper around ChromaDB for storing and retrieving embeddings.
"""
from __future__ import annotations

import logging
from pathlib import Path

import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore as LlamaChromaVectorStore

logger = logging.getLogger(__name__)

DEFAULT_PERSIST_DIR = "./chroma_db"
DEFAULT_COLLECTION = "eu_crr"


class VectorStore:
    """Manages the Chroma vector store used by HierarchicalIndexer and QueryEngine."""

    def __init__(
        self,
        persist_dir: str = DEFAULT_PERSIST_DIR,
        collection_name: str = DEFAULT_COLLECTION,
    ) -> None:
        self.persist_dir = str(Path(persist_dir).resolve())
        self.collection_name = collection_name
        self._client: chromadb.ClientAPI | None = None
        self._collection: chromadb.Collection | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Initialise or re-open the persistent Chroma client."""
        logger.info("Connecting to Chroma at %s (collection=%s)", self.persist_dir, self.collection_name)
        self._client = chromadb.PersistentClient(path=self.persist_dir)
        self._collection = self._client.get_or_create_collection(self.collection_name)
        logger.info("Chroma collection has %d items.", self._collection.count())

    def reset(self) -> None:
        """Drop and recreate the collection (use before a fresh ingest)."""
        if self._client is None:
            self.connect()
        logger.warning("Resetting Chroma collection '%s'.", self.collection_name)
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(self.collection_name)

    # ------------------------------------------------------------------
    # LlamaIndex adapter
    # ------------------------------------------------------------------

    def as_llama_vector_store(self) -> LlamaChromaVectorStore:
        """Return a LlamaIndex-compatible ChromaVectorStore instance."""
        if self._collection is None:
            self.connect()
        return LlamaChromaVectorStore(chroma_collection=self._collection)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def item_count(self) -> int:
        if self._collection is None:
            return 0
        return self._collection.count()
