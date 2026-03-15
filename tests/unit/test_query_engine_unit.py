"""
Unit tests for QueryEngine internals that can be exercised without Qdrant or OpenAI.

Tests cover:
- _retrieve_with_filters: HYBRID→DEFAULT fallback logic
- synthesis node merging: expanded nodes are deduped and included
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch, call
import pytest

from llama_index.core.vector_stores.types import VectorStoreQueryMode


# ---------------------------------------------------------------------------
# Helpers to build lightweight mock nodes
# ---------------------------------------------------------------------------

def _make_node(node_id: str, article: str = "", score: float = 0.8) -> MagicMock:
    node = MagicMock()
    node.node.node_id = node_id
    node.node.metadata = {"article": article}
    node.node.get_content.return_value = f"content of {node_id}"
    node.score = score
    return node


# ---------------------------------------------------------------------------
# _retrieve_with_filters
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRetrieveWithFilters:
    """_retrieve_with_filters tries HYBRID first; falls back to DEFAULT on empty."""

    def _engine(self, hybrid_results, default_results=None):
        """Build a minimal QueryEngine with a mocked _vector_index."""
        from src.query.query_engine import QueryEngine
        from src.indexing.index_builder import HierarchicalIndexer
        from src.indexing.vector_store import VectorStore

        vs = MagicMock(spec=VectorStore)
        idx = MagicMock(spec=HierarchicalIndexer)
        qe = QueryEngine.__new__(QueryEngine)
        qe.vector_store = vs
        qe.indexer = idx
        qe._vector_index = MagicMock()

        # Configure as_retriever to return different results per mode
        def _as_retriever(similarity_top_k, vector_store_query_mode, filters):
            retriever = MagicMock()
            if vector_store_query_mode == VectorStoreQueryMode.HYBRID:
                retriever.retrieve.return_value = hybrid_results
            else:
                retriever.retrieve.return_value = (
                    default_results if default_results is not None else []
                )
            return retriever

        qe._vector_index.as_retriever.side_effect = _as_retriever
        return qe

    def test_returns_hybrid_results_when_non_empty(self):
        nodes = [_make_node("art_92_en", "92")]
        qe = self._engine(hybrid_results=nodes)
        filters = MagicMock()
        result = qe._retrieve_with_filters(filters=filters, query_str="Article 92", top_k=5)
        assert result == nodes

    def test_falls_back_to_default_when_hybrid_empty(self):
        default_nodes = [_make_node("art_92_en", "92")]
        qe = self._engine(hybrid_results=[], default_results=default_nodes)
        filters = MagicMock()
        result = qe._retrieve_with_filters(filters=filters, query_str="Article 92", top_k=5)
        assert result == default_nodes

    def test_returns_empty_when_both_modes_empty(self):
        qe = self._engine(hybrid_results=[], default_results=[])
        filters = MagicMock()
        result = qe._retrieve_with_filters(filters=filters, query_str="Article 92", top_k=5)
        assert result == []

    def test_falls_back_to_default_when_hybrid_raises(self):
        default_nodes = [_make_node("art_92_en", "92")]
        from src.query.query_engine import QueryEngine
        from src.indexing.vector_store import VectorStore
        from src.indexing.index_builder import HierarchicalIndexer

        qe = QueryEngine.__new__(QueryEngine)
        qe._vector_index = MagicMock()

        call_count = {"n": 0}

        def _as_retriever(similarity_top_k, vector_store_query_mode, filters):
            retriever = MagicMock()
            if vector_store_query_mode == VectorStoreQueryMode.HYBRID:
                retriever.retrieve.side_effect = RuntimeError("Qdrant sparse error")
            else:
                retriever.retrieve.return_value = default_nodes
            return retriever

        qe._vector_index.as_retriever.side_effect = _as_retriever
        result = qe._retrieve_with_filters(filters=MagicMock(), query_str="q", top_k=1)
        assert result == default_nodes

    def test_does_not_call_default_when_hybrid_succeeds(self):
        nodes = [_make_node("art_4_en", "4")]
        qe = self._engine(hybrid_results=nodes, default_results=[_make_node("art_4_en", "4")])
        filters = MagicMock()
        qe._retrieve_with_filters(filters=filters, query_str="Article 4", top_k=1)
        # Only one call to as_retriever — HYBRID was enough
        assert qe._vector_index.as_retriever.call_count == 1
        call_kwargs = qe._vector_index.as_retriever.call_args[1]
        assert call_kwargs["vector_store_query_mode"] == VectorStoreQueryMode.HYBRID


# ---------------------------------------------------------------------------
# Synthesis node merging (expanded nodes included + deduplication)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSynthesisNodeMerging:
    """Verify that query() synthesizes over source_nodes + deduped expanded_nodes."""

    def _build_query_engine(self):
        """Return a QueryEngine with all external dependencies mocked."""
        from src.query.query_engine import QueryEngine
        from src.indexing.vector_store import VectorStore
        from src.indexing.index_builder import HierarchicalIndexer

        vs = MagicMock(spec=VectorStore)
        idx = MagicMock(spec=HierarchicalIndexer)
        qe = QueryEngine.__new__(QueryEngine)
        qe.vector_store = vs
        qe.indexer = idx
        qe.openai_api_key = "test"
        qe.llm_model = "gpt-4o"
        qe.max_cross_ref_expansions = 3
        qe.use_reranker = False
        qe._reranker = None
        qe._vector_index = MagicMock()
        qe._engine_cache = {}
        qe._engine_cache_lock = __import__("threading").Lock()
        return qe

    def test_expanded_nodes_included_in_synthesis(self):
        """Synthesis call receives source + deduped expanded nodes."""
        qe = self._build_query_engine()

        source = _make_node("art_92_en", "92")
        expanded = _make_node("art_26_en", "26")

        mock_engine = MagicMock()
        mock_engine.retrieve.return_value = [source]
        mock_engine.synthesize.return_value = MagicMock(__str__=lambda s: "answer")
        qe._engine = mock_engine

        qe._expand_cross_references = MagicMock(return_value=[expanded])

        with patch("src.query.query_engine._normalise_query", side_effect=lambda q: q), \
             patch("src.query.query_engine._detect_direct_article_lookup", return_value=None):
            result = qe.query("What are own funds requirements?")

        # synthesize must have been called with both nodes
        call_args = mock_engine.synthesize.call_args
        nodes_passed = call_args[0][1]  # second positional arg
        node_ids = {n.node.node_id for n in nodes_passed}
        assert "art_92_en" in node_ids
        assert "art_26_en" in node_ids

    def test_duplicate_expanded_nodes_are_deduped(self):
        """A node already in source_nodes must not appear twice in synthesis."""
        qe = self._build_query_engine()

        source = _make_node("art_92_en", "92")
        # Same node_id as source — should be dropped from expanded
        duplicate = _make_node("art_92_en", "92")

        mock_engine = MagicMock()
        mock_engine.retrieve.return_value = [source]
        mock_engine.synthesize.return_value = MagicMock(__str__=lambda s: "answer")
        qe._engine = mock_engine

        qe._expand_cross_references = MagicMock(return_value=[duplicate])

        with patch("src.query.query_engine._normalise_query", side_effect=lambda q: q), \
             patch("src.query.query_engine._detect_direct_article_lookup", return_value=None):
            result = qe.query("Article 92 requirements")

        call_args = mock_engine.synthesize.call_args
        nodes_passed = call_args[0][1]
        # Only one node — duplicate was deduped
        assert len(nodes_passed) == 1
        assert nodes_passed[0].node.node_id == "art_92_en"

    def test_get_article_deduplicates_by_internal_node_id(self):
        """get_article() must not concatenate the same Qdrant record twice.

        LlamaIndex may return the same node twice when both HYBRID and DEFAULT
        retrieval modes are tried, or if a collection has duplicate Qdrant points
        left over from a previous ingest without --reset.  Both scenarios produce
        duplicate text in the returned article body.
        """
        from src.query.query_engine import QueryEngine
        from src.indexing.vector_store import VectorStore
        from src.indexing.index_builder import HierarchicalIndexer

        qe = QueryEngine.__new__(QueryEngine)
        qe.vector_store = MagicMock(spec=VectorStore)
        qe.indexer = MagicMock(spec=HierarchicalIndexer)
        qe._vector_index = MagicMock()

        # Two nodes with the SAME internal node_id (same Qdrant record returned twice)
        node_a = _make_node("uuid-abc-123", article="94")
        node_a.node.get_content.return_value = "1. Paragraph one. 2. Paragraph two."
        node_b = _make_node("uuid-abc-123", article="94")  # same UUID
        node_b.node.metadata = {"article": "94", "language": "en",
                                "article_title": "", "part": "III",
                                "title": "I", "chapter": "1", "section": "1",
                                "referenced_articles": ""}
        node_b.node.get_content.return_value = "1. Paragraph one. 2. Paragraph two."

        qe._retrieve_with_filters = MagicMock(return_value=[node_a, node_b])

        result = qe.get_article("94", language="en")
        assert result is not None
        # Full text must contain paragraph content exactly once, not duplicated
        assert result["text"].count("Paragraph one") == 1, (
            "Duplicate node with same node_id should be deduplicated"
        )

    def test_expanded_sources_flagged_correctly(self):
        """Sources list: primary nodes have expanded=False, expansion nodes expanded=True."""
        qe = self._build_query_engine()

        source = _make_node("art_92_en", "92")
        expanded = _make_node("art_26_en", "26")

        mock_engine = MagicMock()
        mock_engine.retrieve.return_value = [source]
        mock_engine.synthesize.return_value = MagicMock(__str__=lambda s: "answer")
        qe._engine = mock_engine

        qe._expand_cross_references = MagicMock(return_value=[expanded])

        with patch("src.query.query_engine._normalise_query", side_effect=lambda q: q), \
             patch("src.query.query_engine._detect_direct_article_lookup", return_value=None):
            result = qe.query("What are own funds requirements?")

        expanded_flags = {s["metadata"]["article"]: s["expanded"] for s in result.sources}
        assert expanded_flags["92"] is False
        assert expanded_flags["26"] is True
