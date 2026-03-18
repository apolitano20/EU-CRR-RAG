"""
Unit tests for conversational memory helpers in src/query/query_engine.py.

Covers:
- _format_history
- _rewrite_query_with_history
- QueryEngine.query() with/without history
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# _format_history
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFormatHistory:
    def _get_fn(self):
        from src.query.query_engine import _format_history
        return _format_history

    def test_empty_history_returns_empty_string(self):
        fn = self._get_fn()
        assert fn([]) == ""

    def test_single_turn_formatted_correctly(self):
        fn = self._get_fn()
        result = fn([{"question": "What is CET1?", "answer": "Common Equity Tier 1."}])
        assert "Q: What is CET1?" in result
        assert "A: Common Equity Tier 1." in result

    def test_two_turns_joined_with_separator(self):
        fn = self._get_fn()
        turns = [
            {"question": "Q1", "answer": "A1"},
            {"question": "Q2", "answer": "A2"},
        ]
        result = fn(turns)
        assert "---" in result
        assert "Q: Q1" in result
        assert "Q: Q2" in result

    def test_truncation_to_max_turns(self):
        fn = self._get_fn()
        turns = [{"question": f"Q{i}", "answer": f"A{i}"} for i in range(7)]
        result = fn(turns, max_turns=5)
        assert "Q: Q0" not in result
        assert "Q: Q1" not in result
        assert "Q: Q2" in result
        assert "Q: Q6" in result

    def test_exactly_five_turns_not_truncated(self):
        fn = self._get_fn()
        turns = [{"question": f"Q{i}", "answer": f"A{i}"} for i in range(5)]
        result = fn(turns, max_turns=5)
        for i in range(5):
            assert f"Q: Q{i}" in result

    def test_long_answer_preserved_in_full(self):
        fn = self._get_fn()
        long_answer = "x" * 800
        result = fn([{"question": "Q", "answer": long_answer}])
        assert long_answer in result

    def test_answer_appears_verbatim_in_output(self):
        fn = self._get_fn()
        answer_text = "The capital ratio must be at least 8% per Article 92(1)(c)."
        result = fn([{"question": "Q", "answer": answer_text}])
        assert answer_text in result


# ---------------------------------------------------------------------------
# _rewrite_query_with_history
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRewriteQueryWithHistory:
    def _get_fn(self):
        from src.query.query_engine import _rewrite_query_with_history
        return _rewrite_query_with_history

    def _make_mock_openai(self, content: str):
        mock_response = MagicMock()
        mock_response.choices[0].message.content = content
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_cls = MagicMock(return_value=mock_client)
        return mock_openai_cls, mock_client

    def test_rewrite_returns_llm_response(self):
        fn = self._get_fn()
        mock_cls, _ = self._make_mock_openai("What are the AT1 capital requirements?")
        history = [{"question": "What is CET1?", "answer": "Common Equity Tier 1."}]
        with patch("src.query.query_engine.openai.OpenAI", mock_cls):
            result = fn("And what about AT1?", history, "fake-key")
        assert result == "What are the AT1 capital requirements?"

    def test_rewrite_history_included_in_prompt(self):
        fn = self._get_fn()
        mock_cls, mock_client = self._make_mock_openai("standalone question")
        history = [{"question": "Tell me about CET1", "answer": "CET1 is tier 1 capital."}]
        with patch("src.query.query_engine.openai.OpenAI", mock_cls):
            fn("And AT1?", history, "fake-key")
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args[1]["messages"] if "messages" in call_args[1] else call_args[0][0]
        prompt_text = messages[0]["content"]
        assert "CET1" in prompt_text
        assert "CET1 is tier 1 capital." in prompt_text

    def test_rewrite_falls_back_on_api_error(self):
        fn = self._get_fn()
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API down")
        mock_cls = MagicMock(return_value=mock_client)
        history = [{"question": "Q", "answer": "A"}]
        with patch("src.query.query_engine.openai.OpenAI", mock_cls):
            result = fn("original query", history, "fake-key")
        assert result == "original query"

    def test_rewrite_truncates_history_to_five_turns(self):
        fn = self._get_fn()
        mock_cls, mock_client = self._make_mock_openai("rewritten")
        turns = [{"question": f"Q{i}", "answer": f"A{i}"} for i in range(7)]
        with patch("src.query.query_engine.openai.OpenAI", mock_cls):
            fn("follow-up", turns, "fake-key")
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args[1]["messages"] if "messages" in call_args[1] else call_args[0][0]
        prompt_text = messages[0]["content"]
        # Only last 5 turns should appear — Q0 and Q1 truncated
        assert "Q: Q0" not in prompt_text
        assert "Q: Q1" not in prompt_text
        assert "Q: Q2" in prompt_text

    def test_rewrite_same_language_instruction_in_prompt(self):
        fn = self._get_fn()
        mock_cls, mock_client = self._make_mock_openai("rewritten")
        history = [{"question": "Q", "answer": "A"}]
        with patch("src.query.query_engine.openai.OpenAI", mock_cls):
            fn("follow-up", history, "fake-key")
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args[1]["messages"] if "messages" in call_args[1] else call_args[0][0]
        prompt_text = messages[0]["content"]
        assert "same language" in prompt_text.lower() or "SAME LANGUAGE" in prompt_text


# ---------------------------------------------------------------------------
# QueryEngine.query() with/without history
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestQueryEngineHistory:
    """Tests QueryEngine.query() history integration with mocked retrieval and OpenAI."""

    def _make_engine(self):
        """Return a QueryEngine with mocked internals."""
        from src.query.query_engine import QueryEngine
        from src.indexing.vector_store import VectorStore
        from src.indexing.index_builder import HierarchicalIndexer

        with patch("src.indexing.vector_store.QdrantClient"), \
             patch("src.indexing.index_builder.HierarchicalIndexer.load"):
            vs = VectorStore()
            indexer = HierarchicalIndexer(vector_store=vs)
            engine = QueryEngine(
                vector_store=vs,
                indexer=indexer,
                openai_api_key="fake-key",
            )
        # Stub _engine so is_loaded() returns True
        engine._engine = MagicMock()
        return engine

    def _make_fake_node(self, article="92", content="text"):
        node = MagicMock()
        node.node.metadata = {"article": article, "article_title": ""}
        node.node.get_content.return_value = content
        node.node.node_id = article
        node.score = 0.9
        return node

    def _mock_retrieve(self, engine, nodes=None):
        if nodes is None:
            nodes = [self._make_fake_node()]
        engine.retrieve = MagicMock(
            return_value=(nodes, [], "trace-123", "normalised query", MagicMock())
        )

    def _mock_openai(self, answer="mocked answer"):
        mock_response = MagicMock()
        mock_response.choices[0].message.content = answer
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        return MagicMock(return_value=mock_client), mock_client

    def test_no_history_uses_base_template(self):
        engine = self._make_engine()
        self._mock_retrieve(engine)
        mock_cls, mock_client = self._mock_openai()
        with patch("src.query.query_engine.openai.OpenAI", mock_cls), \
             patch("src.query.query_engine._rewrite_query_with_history") as mock_rw:
            engine.query("What is CET1?", history=[])
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args[1]["messages"] if "messages" in call_args[1] else call_args[0][0]
        prompt_text = messages[0]["content"]
        assert "Prior conversation" not in prompt_text
        mock_rw.assert_not_called()

    def test_with_history_uses_history_template(self):
        engine = self._make_engine()
        self._mock_retrieve(engine)
        mock_cls, mock_client = self._mock_openai()
        history = [{"question": "What is CET1?", "answer": "Common Equity Tier 1."}]
        with patch("src.query.query_engine.openai.OpenAI", mock_cls), \
             patch("src.query.query_engine._rewrite_query_with_history", return_value="And AT1?"):
            engine.query("And AT1?", history=history)
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args[1]["messages"] if "messages" in call_args[1] else call_args[0][0]
        prompt_text = messages[0]["content"]
        assert "Prior conversation" in prompt_text

    def test_rewrite_always_called_when_history_non_empty(self):
        engine = self._make_engine()
        self._mock_retrieve(engine)
        mock_cls, _ = self._mock_openai()
        history = [{"question": "Q", "answer": "A"}]
        with patch("src.query.query_engine.openai.OpenAI", mock_cls), \
             patch("src.query.query_engine._rewrite_query_with_history", return_value="rewritten") as mock_rw:
            engine.query("follow-up", history=history)
        mock_rw.assert_called_once()

    def test_rewrite_skipped_when_history_empty(self):
        engine = self._make_engine()
        self._mock_retrieve(engine)
        mock_cls, _ = self._mock_openai()
        with patch("src.query.query_engine.openai.OpenAI", mock_cls), \
             patch("src.query.query_engine._rewrite_query_with_history") as mock_rw:
            engine.query("What is CET1?", history=[])
        mock_rw.assert_not_called()

    def test_rewritten_query_passed_to_retrieve(self):
        engine = self._make_engine()
        self._mock_retrieve(engine)
        mock_cls, _ = self._mock_openai()
        history = [{"question": "Q", "answer": "A"}]
        with patch("src.query.query_engine.openai.OpenAI", mock_cls), \
             patch("src.query.query_engine._rewrite_query_with_history", return_value="rewritten standalone"):
            engine.query("follow-up", history=history)
        engine.retrieve.assert_called_once()
        assert engine.retrieve.call_args[0][0] == "rewritten standalone"

    def test_openai_called_directly_not_synthesize(self):
        engine = self._make_engine()
        self._mock_retrieve(engine)
        mock_cls, mock_client = self._mock_openai()
        with patch("src.query.query_engine.openai.OpenAI", mock_cls), \
             patch("src.query.query_engine._rewrite_query_with_history", return_value="q"):
            engine.query("What is CET1?", history=[])
        mock_client.chat.completions.create.assert_called_once()
        # synthesize should NOT have been called on the engine mock
        engine._engine.synthesize.assert_not_called()
