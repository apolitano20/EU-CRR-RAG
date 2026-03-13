"""
Integration tests for the ingestion pipeline (live Qdrant, no OpenAI needed).

Uses a small synthetic document set so the test runs in seconds rather than
the 20+ minutes a full CRR ingest would take.

Requires: QDRANT_URL, QDRANT_API_KEY in .env
"""
from __future__ import annotations

import pytest

from llama_index.core import Document

from src.indexing.index_builder import HierarchicalIndexer
from src.indexing.vector_store import VectorStore


MINI_DOCS = [
    Document(
        text=(
            "Article 92 sets out own funds requirements. "
            "Institutions shall at all times satisfy the following own funds requirements: "
            "(a) a Common Equity Tier 1 capital ratio of 4,5 %; "
            "(b) a Tier 1 capital ratio of 6 %; "
            "(c) a total capital ratio of 8 %."
        ),
        metadata={
            "node_id": "art_92_en",
            "level": "ARTICLE",
            "part": "THREE",
            "title": "I",
            "chapter": "1",
            "section": "",
            "article": "92",
            "article_title": "Own funds requirements",
            "annex_id": "",
            "annex_title": "",
            "referenced_articles": "26,36",
            "referenced_external": "",
            "has_table": False,
            "has_formula": False,
            "language": "en",
        },
    ),
    Document(
        text=(
            "Article 26 defines Common Equity Tier 1 items. "
            "Common Equity Tier 1 items of institutions consist of the following: "
            "(a) capital instruments, provided that the conditions laid down in Article 28 are met."
        ),
        metadata={
            "node_id": "art_26_en",
            "level": "ARTICLE",
            "part": "TWO",
            "title": "I",
            "chapter": "1",
            "section": "",
            "article": "26",
            "article_title": "CET1 items",
            "annex_id": "",
            "annex_title": "",
            "referenced_articles": "28",
            "referenced_external": "",
            "has_table": False,
            "has_formula": False,
            "language": "en",
        },
    ),
    Document(
        text=(
            "Articolo 92 stabilisce i requisiti in materia di fondi propri. "
            "Gli enti soddisfano in ogni momento i seguenti requisiti in materia di fondi propri."
        ),
        metadata={
            "node_id": "art_92_it",
            "level": "ARTICLE",
            "part": "THREE",
            "title": "I",
            "chapter": "1",
            "section": "",
            "article": "92",
            "article_title": "Requisiti in materia di fondi propri",
            "annex_id": "",
            "annex_title": "",
            "referenced_articles": "26,36",
            "referenced_external": "",
            "has_table": False,
            "has_formula": False,
            "language": "it",
        },
    ),
]


@pytest.fixture(scope="module")
def store(test_collection_name, qdrant_url, qdrant_api_key):
    vs = VectorStore(
        collection_name=test_collection_name,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
    )
    vs.reset()
    yield vs
    vs._client.delete_collection(test_collection_name)


@pytest.fixture(scope="module")
def built_indexer(store):
    indexer = HierarchicalIndexer(
        vector_store=store,
        reset_store=False,
    )
    indexer.build(MINI_DOCS)
    return indexer


@pytest.mark.integration
class TestIngestPipelineSmallSlice:
    def test_item_count_increases_after_build(self, built_indexer, store):
        assert store.item_count > 0

    def test_item_count_matches_document_count(self, built_indexer, store):
        """Article-level indexing: one vector per document."""
        assert store.item_count == len(MINI_DOCS)

    def test_vector_index_returned(self, built_indexer):
        from llama_index.core import VectorStoreIndex
        assert isinstance(built_indexer._vector_index, VectorStoreIndex)


@pytest.mark.integration
class TestLanguageMetadataStored:
    def test_en_nodes_have_language_metadata(self, built_indexer, store):
        """Scroll through stored points and verify language metadata is present."""
        points, _ = store._client.scroll(
            collection_name=store.collection_name,
            limit=50,
            with_payload=True,
        )
        lang_values = {
            p.payload.get("language")
            for p in points
            if p.payload and "language" in p.payload
        }
        assert "en" in lang_values

    def test_it_nodes_have_language_metadata(self, built_indexer, store):
        points, _ = store._client.scroll(
            collection_name=store.collection_name,
            limit=50,
            with_payload=True,
        )
        lang_values = {
            p.payload.get("language")
            for p in points
            if p.payload and "language" in p.payload
        }
        assert "it" in lang_values

    def test_new_metadata_fields_stored(self, built_indexer, store):
        """Verify the new schema fields are persisted in Qdrant."""
        points, _ = store._client.scroll(
            collection_name=store.collection_name,
            limit=50,
            with_payload=True,
        )
        for point in points:
            payload = point.payload or {}
            assert "article_title" in payload
            assert "referenced_articles" in payload
            assert "has_table" in payload
            assert "has_formula" in payload


@pytest.mark.integration
class TestIndexReload:
    def test_load_succeeds_after_build(self, built_indexer, store):
        """Re-open the persisted index without re-ingesting."""
        reload_indexer = HierarchicalIndexer(vector_store=store)
        idx = reload_indexer.load()
        from llama_index.core import VectorStoreIndex
        assert isinstance(idx, VectorStoreIndex)
