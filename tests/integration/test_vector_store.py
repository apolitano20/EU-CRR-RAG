"""
Integration tests for VectorStore (live Qdrant Cloud).

Requires: QDRANT_URL, QDRANT_API_KEY in .env
Uses a dedicated test collection (eu_crr_test) to avoid touching production data.
"""
from __future__ import annotations

import pytest

from src.indexing.vector_store import VectorStore


@pytest.fixture(scope="module")
def store(test_collection_name, qdrant_url, qdrant_api_key):
    """Fresh test collection, torn down after the module."""
    vs = VectorStore(
        collection_name=test_collection_name,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
    )
    vs.reset()  # ensure clean state
    yield vs
    # Teardown: drop test collection
    vs._client.delete_collection(test_collection_name)


@pytest.mark.integration
class TestVectorStoreConnect:
    def test_connect_succeeds(self, store):
        """connect() should not raise and client should be set."""
        assert store._client is not None

    def test_collection_exists_after_connect(self, store):
        existing = {c.name for c in store._client.get_collections().collections}
        assert store.collection_name in existing


@pytest.mark.integration
class TestVectorStoreItemCount:
    def test_item_count_returns_int(self, store):
        count = store.item_count
        assert isinstance(count, int)

    def test_item_count_non_negative(self, store):
        assert store.item_count >= 0

    def test_fresh_collection_is_empty(self, store):
        assert store.item_count == 0


@pytest.mark.integration
class TestVectorStoreReset:
    def test_reset_drops_recreates_and_clears(self, store):
        """Single test: reset must both recreate the collection and leave it empty."""
        store.reset()
        existing = {c.name for c in store._client.get_collections().collections}
        assert store.collection_name in existing
        assert store.item_count == 0


@pytest.mark.integration
class TestAsLlamaVectorStore:
    def test_returns_qdrant_vector_store(self, store):
        from llama_index.vector_stores.qdrant import QdrantVectorStore
        lv = store.as_llama_vector_store()
        assert isinstance(lv, QdrantVectorStore)

    def test_hybrid_enabled(self, store):
        lv = store.as_llama_vector_store()
        # Attribute was renamed from _enable_hybrid to enable_hybrid in newer adapter versions
        enabled = getattr(lv, "enable_hybrid", None) or getattr(lv, "_enable_hybrid", None)
        assert enabled is True
