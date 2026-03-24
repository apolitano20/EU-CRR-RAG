"""Unit tests for ToC routing: merge_rrf() and orchestrator _toc_route() integration.

All tests are offline: no Qdrant, no OpenAI API calls.
"""
from __future__ import annotations

import json
import uuid
from unittest.mock import MagicMock, patch

import pytest

from src.query.orchestrator import QueryOrchestrator, QueryType
from src.query.query_engine import RERANK_TOP_N, QueryResult, merge_rrf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_node(node_id: str, article: str = "92", score: float = 0.8):
    """Create a minimal mock NodeWithScore for testing."""
    from llama_index.core.schema import NodeWithScore, TextNode
    node = TextNode(text=f"Content for article {article}", id_=node_id)
    node.metadata = {"article": article, "language": "en"}
    return NodeWithScore(node=node, score=score)


def _make_orchestrator(toc_store=None):
    """Create a QueryOrchestrator with a mocked QueryEngine."""
    engine = MagicMock()
    engine.is_loaded.return_value = True
    engine.toc_store = toc_store
    orch = QueryOrchestrator(query_engine=engine, openai_api_key="test-key")
    return orch, engine


# ---------------------------------------------------------------------------
# Group 1: merge_rrf
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMergeRrf:
    def test_basic_no_overlap(self):
        """Two disjoint lists: all nodes present, sorted by RRF score."""
        v_nodes = [_make_node("v1", "92", 0.9), _make_node("v2", "93", 0.8)]
        t_nodes = [_make_node("t1", "411", 0.7), _make_node("t2", "412", 0.6)]
        result = merge_rrf(v_nodes, t_nodes, k=60)
        assert len(result) == 4
        ids = [n.node.node_id for n in result]
        # v1 and t1 both rank first in their respective lists → should be top
        assert ids[0] == "v1"
        assert ids[1] == "t1"

    def test_overlap_boosts_shared_node(self):
        """A node in both lists gets a cumulative RRF score boost."""
        shared = _make_node("shared", "92", 0.9)
        v_nodes = [shared, _make_node("v2", "93", 0.8)]
        # shared also appears first in toc list
        t_nodes = [_make_node("shared", "92", 0.7), _make_node("t2", "411", 0.6)]
        result = merge_rrf(v_nodes, t_nodes, k=60)
        # shared gets contributions from both lists → should rank first
        ids = [n.node.node_id for n in result]
        assert ids[0] == "shared"

    def test_dedup_single_entry_per_node_id(self):
        """Same node_id in both lists → only one entry in output."""
        shared_node = _make_node("shared", "92", 0.9)
        v_nodes = [shared_node]
        t_nodes = [_make_node("shared", "92", 0.5)]
        result = merge_rrf(v_nodes, t_nodes)
        assert len(result) == 1
        assert result[0].node.node_id == "shared"

    def test_cap_at_rerank_top_n(self):
        """Output is capped at RERANK_TOP_N even with many inputs."""
        v_nodes = [_make_node(f"v{i}", str(i), 0.9 - i * 0.05) for i in range(10)]
        t_nodes = [_make_node(f"t{i}", str(i + 100), 0.8 - i * 0.05) for i in range(10)]
        result = merge_rrf(v_nodes, t_nodes, cap=RERANK_TOP_N)
        assert len(result) == RERANK_TOP_N

    def test_custom_cap(self):
        v_nodes = [_make_node(f"v{i}", str(i)) for i in range(5)]
        t_nodes = []
        result = merge_rrf(v_nodes, t_nodes, cap=3)
        assert len(result) == 3

    def test_empty_toc_nodes_returns_vector_nodes_normalised(self):
        """Empty toc_nodes: vector nodes returned with normalised scores."""
        v_nodes = [
            _make_node("v1", "92", 0.9),
            _make_node("v2", "93", 0.8),
        ]
        result = merge_rrf(v_nodes, [], k=60, cap=RERANK_TOP_N)
        assert len(result) == 2
        # Top node should have score = 1.0 (normalised)
        assert result[0].score == pytest.approx(1.0)
        assert result[0].node.node_id == "v1"

    def test_empty_vector_nodes_returns_toc_nodes_normalised(self):
        """Empty vector_nodes: toc nodes returned with normalised scores."""
        t_nodes = [_make_node("t1", "411", 0.7)]
        result = merge_rrf([], t_nodes)
        assert len(result) == 1
        assert result[0].score == pytest.approx(1.0)

    def test_both_empty_returns_empty(self):
        result = merge_rrf([], [])
        assert result == []

    def test_scores_are_normalised_to_0_1(self):
        """All output scores must be in [0, 1]."""
        v_nodes = [_make_node(f"v{i}", str(i)) for i in range(5)]
        t_nodes = [_make_node(f"t{i}", str(i + 10)) for i in range(5)]
        result = merge_rrf(v_nodes, t_nodes)
        for node in result:
            assert 0.0 <= node.score <= 1.0, f"Score out of range: {node.score}"

    def test_node_content_preserved(self):
        """The merged node should preserve the original node content."""
        v_nodes = [_make_node("n1", "92", 0.9)]
        result = merge_rrf(v_nodes, [])
        assert result[0].node.get_content() == "Content for article 92"


# ---------------------------------------------------------------------------
# Group 2: _toc_route
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTocRoute:
    def _make_toc_store(self, loaded: bool = True, language: str = "en"):
        store = MagicMock()
        store.is_loaded.return_value = loaded
        store.format_for_prompt.return_value = "PART THREE\n  Art. 92 — Own funds"
        return store

    def test_returns_article_list_on_success(self):
        store = self._make_toc_store()
        orch, engine = _make_orchestrator(toc_store=store)

        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps(
            {"reasoning": "TREA is defined in Art 92", "articles": ["92", "93"]}
        )

        with patch("openai.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.create.return_value = mock_response
            result = orch._toc_route("What is TREA?", "en")

        assert result == ["92", "93"]

    def test_returns_empty_when_toc_store_none(self):
        orch, _ = _make_orchestrator(toc_store=None)
        result = orch._toc_route("What is TREA?", "en")
        assert result == []

    def test_returns_empty_when_language_not_loaded(self):
        store = self._make_toc_store(loaded=False)
        orch, _ = _make_orchestrator(toc_store=store)
        result = orch._toc_route("What is TREA?", "en")
        assert result == []

    def test_returns_empty_on_api_exception(self):
        store = self._make_toc_store()
        orch, _ = _make_orchestrator(toc_store=store)

        with patch("openai.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.create.side_effect = Exception(
                "API error"
            )
            result = orch._toc_route("What is TREA?", "en")

        assert result == []

    def test_regex_fallback_on_invalid_json(self):
        """If JSON parse fails, regex extracts article numbers from raw text."""
        store = self._make_toc_store()
        orch, _ = _make_orchestrator(toc_store=store)

        mock_response = MagicMock()
        # Non-JSON response containing article numbers
        mock_response.choices[0].message.content = (
            "The relevant articles are 92 and 93."
        )

        with patch("openai.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.create.return_value = mock_response
            result = orch._toc_route("What is TREA?", "en")

        # Regex fallback should extract "92" and "93"
        assert "92" in result or "93" in result

    def test_caps_at_6_articles(self):
        store = self._make_toc_store()
        orch, _ = _make_orchestrator(toc_store=store)

        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps(
            {"reasoning": "Many articles", "articles": ["1", "2", "3", "4", "5", "6", "7", "8"]}
        )

        with patch("openai.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.create.return_value = mock_response
            result = orch._toc_route("complex query", "en")

        assert len(result) <= 6

    def test_filters_non_article_numbers(self):
        """Non-numeric strings (e.g. Roman numerals for annexes) are filtered out."""
        store = self._make_toc_store()
        orch, _ = _make_orchestrator(toc_store=store)

        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps(
            {"reasoning": "...", "articles": ["92", "ANNEX_I", "93", "invalid"]}
        )

        with patch("openai.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.create.return_value = mock_response
            result = orch._toc_route("query", "en")

        assert "ANNEX_I" not in result
        assert "invalid" not in result
        assert "92" in result
        assert "93" in result


# ---------------------------------------------------------------------------
# Group 3: Orchestrator parallel execution
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestOrchestratorWithToc:
    def _mock_retrieve_result(self, articles: list[str]):
        """Create a mock retrieve() return value."""
        nodes = [_make_node(f"n{i}", art, 0.8) for i, art in enumerate(articles)]
        sources = [
            {"text": f"Art {art}", "score": 0.8, "metadata": {"article": art}, "expanded": False}
            for art in articles
        ]
        trace_id = str(uuid.uuid4())
        return (nodes, sources, trace_id, "normalised query", None)

    def test_toc_routing_merges_results(self):
        """When USE_TOC_ROUTING=true and toc returns articles, results are merged."""
        store = MagicMock()
        store.is_loaded.return_value = True
        store.format_for_prompt.return_value = "Art. 92 — Own funds"

        orch, engine = _make_orchestrator(toc_store=store)

        # Vector retrieval returns articles 93, 94
        engine.retrieve.return_value = self._mock_retrieve_result(["93", "94"])
        # ToC retrieval returns node for article 92
        toc_node = _make_node("toc_92", "92", 0.5)
        engine.toc_retrieve.return_value = [toc_node]

        # Patch _toc_route to return ["92"]
        with (
            patch.object(orch, "_toc_route", return_value=["92"]),
            patch("openai.OpenAI") as MockOpenAI,
            patch.dict("os.environ", {"USE_TOC_ROUTING": "true"}),
        ):
            mock_resp = MagicMock()
            mock_resp.choices[0].message.content = "The own funds requirement is..."
            MockOpenAI.return_value.chat.completions.create.return_value = mock_resp

            result = orch.query("What is TREA?", language="en")

        assert isinstance(result, QueryResult)
        # toc_retrieve should have been called with ["92"]
        engine.toc_retrieve.assert_called_once()
        call_args = engine.toc_retrieve.call_args
        assert "92" in call_args[0][0]  # first positional arg is article_numbers

    def test_toc_routing_skipped_when_env_false(self):
        """When USE_TOC_ROUTING=false, _toc_route is never called."""
        store = MagicMock()
        store.is_loaded.return_value = True

        orch, engine = _make_orchestrator(toc_store=store)
        engine.retrieve.return_value = self._mock_retrieve_result(["92"])

        with (
            patch.object(orch, "_toc_route") as mock_toc_route,
            patch("openai.OpenAI") as MockOpenAI,
            patch.dict("os.environ", {"USE_TOC_ROUTING": "false"}),
        ):
            mock_resp = MagicMock()
            mock_resp.choices[0].message.content = "Answer here."
            MockOpenAI.return_value.chat.completions.create.return_value = mock_resp

            orch.query("What is TREA?", language="en")

        mock_toc_route.assert_not_called()

    def test_toc_routing_skipped_for_direct_article(self):
        """DIRECT_ARTICLE queries skip ToC routing even if USE_TOC_ROUTING=true."""
        store = MagicMock()
        store.is_loaded.return_value = True

        orch, engine = _make_orchestrator(toc_store=store)
        engine.retrieve.return_value = self._mock_retrieve_result(["92"])

        with (
            patch.object(orch, "_toc_route") as mock_toc_route,
            patch("openai.OpenAI") as MockOpenAI,
            patch.dict("os.environ", {"USE_TOC_ROUTING": "true"}),
        ):
            mock_resp = MagicMock()
            mock_resp.choices[0].message.content = "Answer here."
            MockOpenAI.return_value.chat.completions.create.return_value = mock_resp

            # Query that directly references an article → DIRECT_ARTICLE type
            orch.query("What does Article 92 say?", language="en")

        mock_toc_route.assert_not_called()

    def test_graceful_fallback_when_toc_route_returns_empty(self):
        """Empty _toc_route result → vector results used unchanged."""
        store = MagicMock()
        store.is_loaded.return_value = True

        orch, engine = _make_orchestrator(toc_store=store)
        vector_result = self._mock_retrieve_result(["92", "93"])
        engine.retrieve.return_value = vector_result

        with (
            patch.object(orch, "_toc_route", return_value=[]),
            patch("openai.OpenAI") as MockOpenAI,
            patch.dict("os.environ", {"USE_TOC_ROUTING": "true"}),
        ):
            mock_resp = MagicMock()
            mock_resp.choices[0].message.content = "Answer."
            MockOpenAI.return_value.chat.completions.create.return_value = mock_resp

            result = orch.query("What is TREA?", language="en")

        assert isinstance(result, QueryResult)
        # toc_retrieve should NOT be called when _toc_route returns []
        engine.toc_retrieve.assert_not_called()
