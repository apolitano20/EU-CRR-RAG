"""
VectorStore: thin wrapper around Qdrant Cloud for storing and retrieving embeddings.

Supports BGE-M3 native hybrid search (dense + sparse) via a single Qdrant collection.
"""
from __future__ import annotations

import logging
import os

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    PayloadSchemaType,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)

from src.indexing.bge_m3_sparse import sparse_doc_fn, sparse_query_fn

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION = "eu_crr"
_DENSE_DIM = 1024  # BGE-M3 dense output dimension


class VectorStore:
    """Manages the Qdrant Cloud vector store used by HierarchicalIndexer and QueryEngine."""

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION,
        qdrant_url: str | None = None,
        qdrant_api_key: str | None = None,
    ) -> None:
        self.collection_name = collection_name
        self._qdrant_url = qdrant_url or os.getenv("QDRANT_URL")
        self._qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")
        self._client: QdrantClient | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Initialise Qdrant client and ensure the collection exists."""
        logger.info(
            "Connecting to Qdrant at %s (collection=%s)",
            self._qdrant_url,
            self.collection_name,
        )
        self._client = QdrantClient(url=self._qdrant_url, api_key=self._qdrant_api_key)
        self._ensure_collection()
        self._ensure_payload_indexes()
        logger.info("Qdrant collection has %d items.", self.item_count)

    def reset(self) -> None:
        """Drop and recreate the collection (use before a fresh ingest)."""
        if self._client is None:
            self.connect()
        if self._client is None:
            raise RuntimeError("Failed to connect to Qdrant.")
        logger.warning("Resetting Qdrant collection '%s'.", self.collection_name)
        self._client.delete_collection(self.collection_name)
        self._ensure_collection()
        self._ensure_payload_indexes()

    # ------------------------------------------------------------------
    # LlamaIndex adapter
    # ------------------------------------------------------------------

    def as_llama_vector_store(self):
        """Return a LlamaIndex-compatible QdrantVectorStore with hybrid search enabled."""
        from llama_index.vector_stores.qdrant import QdrantVectorStore

        if self._client is None:
            self.connect()
        return QdrantVectorStore(
            client=self._client,
            collection_name=self.collection_name,
            enable_hybrid=True,
            sparse_doc_fn=sparse_doc_fn,
            sparse_query_fn=sparse_query_fn,
            sparse_vector_name="sparse",
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def item_count(self) -> int:
        if self._client is None:
            return 0
        try:
            return self._client.count(self.collection_name).count
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_collection(self) -> None:
        """Create the collection if it does not already exist."""
        existing = {c.name for c in self._client.get_collections().collections}
        if self.collection_name not in existing:
            logger.info("Creating Qdrant collection '%s'.", self.collection_name)
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=_DENSE_DIM, distance=Distance.COSINE),
                sparse_vectors_config={
                    "sparse": SparseVectorParams(
                        index=SparseIndexParams(on_disk=False)
                    )
                },
            )

    def _ensure_payload_indexes(self) -> None:
        """Create keyword payload indexes required for metadata filtering.

        Qdrant requires an index on any field used in a filter. This method is
        idempotent — calling it on a collection that already has the indexes is safe.
        """
        for field in ("language", "article", "level"):
            self._client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )
