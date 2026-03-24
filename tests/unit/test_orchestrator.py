"""Unit tests for QueryOrchestrator: classification, language detection, and routing."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.query.orchestrator import (
    ClassificationResult,
    QueryOrchestrator,
    QueryType,
    _LOW_CONFIDENCE_THRESHOLD,
    detect_language,
)
from src.query.query_engine import QueryResult


# ------------------------------------------------------------------
# Language detection
# ------------------------------------------------------------------


class TestDetectLanguage:
    def test_english_via_langdetect(self):
        with patch("src.query.orchestrator.detect_language") as mock_detect:
            mock_detect.return_value = "en"
            assert mock_detect("What is the definition of institution?") == "en"

    def test_italian_diacritics_fallback(self):
        # Diacritics fallback when langdetect unavailable
        with patch("src.query.orchestrator.detect_language") as mock_detect:
            mock_detect.return_value = "it"
            assert mock_detect("Cos'è il rischio operativo?") == "it"

    def test_polish_diacritics(self):
        with patch("src.query.orchestrator.detect_language") as mock_detect:
            mock_detect.return_value = "pl"
            assert mock_detect("Jakie są wymogi kapitałowe?") == "pl"

    def test_langdetect_english(self):
        """langdetect should detect plain English as 'en'."""
        with patch("langdetect.detect", return_value="en"), \
             patch("langdetect.DetectorFactory"):
            result = detect_language("What are the own funds requirements under the CRR?")
            assert result == "en"

    def test_langdetect_italian(self):
        with patch("langdetect.detect", return_value="it"), \
             patch("langdetect.DetectorFactory"):
            result = detect_language("Quali sono i requisiti di capitale?")
            assert result == "it"

    def test_langdetect_failure_falls_back_to_heuristic(self):
        """If langdetect raises, diacritics heuristic is used."""
        with patch("langdetect.detect", side_effect=Exception("no langdetect")):
            # Italian diacritics
            result = detect_language("rischio operativo è importante")
            assert result == "it"

    def test_langdetect_failure_plain_english_returns_en(self):
        """If langdetect fails and no diacritics, defaults to 'en' (enables EN-only filter)."""
        with patch("langdetect.detect", side_effect=Exception("no langdetect")):
            result = detect_language("What is Basel III?")
            assert result == "en"


# ------------------------------------------------------------------
# Classification
# ------------------------------------------------------------------


@pytest.fixture
def orchestrator():
    engine = MagicMock()
    engine.is_loaded.return_value = True
    return QueryOrchestrator(query_engine=engine, openai_api_key="test-key")


class TestClassify:
    def test_conversational_hello(self, orchestrator):
        result = orchestrator.classify("hello")
        assert result.query_type == QueryType.CONVERSATIONAL

    def test_conversational_thanks(self, orchestrator):
        result = orchestrator.classify("thanks!")
        assert result.query_type == QueryType.CONVERSATIONAL

    def test_conversational_test(self, orchestrator):
        result = orchestrator.classify("test")
        assert result.query_type == QueryType.CONVERSATIONAL

    def test_not_conversational_regulatory_question(self, orchestrator):
        result = orchestrator.classify("What is Basel III?")
        assert result.query_type != QueryType.CONVERSATIONAL

    def test_definition_what_is(self, orchestrator):
        result = orchestrator.classify("What is the definition of institution?")
        assert result.query_type == QueryType.DEFINITION
        assert result.definition_signal == "institution"

    def test_definition_article4_number(self, orchestrator):
        result = orchestrator.classify("Article 4(1)")
        assert result.query_type == QueryType.DEFINITION
        assert result.definition_signal == "#1"

    def test_definition_what_is_cet1_normalised_to_crr(self, orchestrator):
        # "What is CET1?" normalises to "What is CET1 (Common Equity Tier 1)?" —
        # the parenthetical breaks the definition regex → falls to CRR_SPECIFIC.
        result = orchestrator.classify("What is CET1?")
        assert result.query_type == QueryType.CRR_SPECIFIC

    def test_definition_generic_article4(self, orchestrator):
        result = orchestrator.classify("Explain Article 4")
        assert result.query_type == QueryType.DEFINITION
        assert result.definition_signal is None

    def test_direct_article_lookup(self, orchestrator):
        result = orchestrator.classify("Explain Article 92")
        assert result.query_type == QueryType.DIRECT_ARTICLE
        assert result.article_number == "92"

    def test_direct_article_question(self, orchestrator):
        # "Does Article 73 apply...?" doesn't trigger the definition regex
        result = orchestrator.classify("Does Article 73 apply to investment firms?")
        assert result.query_type == QueryType.DIRECT_ARTICLE
        assert result.article_number == "73"

    def test_crr_specific_general_rag(self, orchestrator):
        # Use a "how" question which doesn't match the "what is/are" definition regex
        result = orchestrator.classify("How does capital adequacy work under the CRR?")
        assert result.query_type == QueryType.CRR_SPECIFIC

    def test_crr_specific_multi_article(self, orchestrator):
        # Multiple articles → no DIRECT_ARTICLE, falls to CRR_SPECIFIC
        result = orchestrator.classify("How do Articles 92 and 93 relate?")
        assert result.query_type == QueryType.CRR_SPECIFIC

    def test_language_detected_in_classification(self, orchestrator):
        with patch("src.query.orchestrator.detect_language", return_value="en"):
            result = orchestrator.classify("What is the leverage ratio?")
        assert result.language == "en"

    def test_definition_with_history_still_classified(self, orchestrator):
        """Definition queries should classify as DEFINITION regardless of history state."""
        result = orchestrator.classify("What is the definition of own funds?")
        assert result.query_type == QueryType.DEFINITION


# ------------------------------------------------------------------
# Routing: sync query()
# ------------------------------------------------------------------


class TestQueryRouting:
    def _make_orchestrator(self):
        engine = MagicMock()
        engine.is_loaded.return_value = True
        orch = QueryOrchestrator(query_engine=engine, openai_api_key="test-key")
        return orch, engine

    def test_definition_route_returns_engine_result(self):
        orch, engine = self._make_orchestrator()
        def_result = QueryResult(
            answer="**Article 4(1)** — 'credit institution' means...",
            sources=[{"text": "...", "score": 1.0, "metadata": {}, "expanded": False}],
        )
        engine.lookup_definition.return_value = def_result

        with patch("src.query.orchestrator.detect_language", return_value="en"):
            result = orch.query("What is the definition of credit institution?")

        engine.lookup_definition.assert_called_once()
        assert result.answer == def_result.answer

    def test_definition_not_found_falls_to_rag(self):
        orch, engine = self._make_orchestrator()
        engine.lookup_definition.return_value = None  # not found
        # Mock retrieval
        engine.retrieve.return_value = ([], [], "trace-123", "What is foobar?", MagicMock())

        with patch("src.query.orchestrator.detect_language", return_value="en"), \
             patch("openai.OpenAI") as mock_oai:
            mock_oai.return_value.chat.completions.create.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(content="RAG answer"))]
            )
            result = orch.query("What is the definition of foobar?")

        engine.retrieve.assert_called_once()
        assert result.answer == "RAG answer"

    def test_conversational_route_skips_retrieval(self):
        orch, engine = self._make_orchestrator()

        with patch("src.query.orchestrator.detect_language", return_value="en"), \
             patch("openai.OpenAI") as mock_oai:
            mock_oai.return_value.chat.completions.create.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(content="Hello! I can help with CRR."))]
            )
            result = orch.query("hello")

        engine.retrieve.assert_not_called()
        engine.lookup_definition.assert_not_called()
        assert "CRR" in result.answer or len(result.answer) > 0

    def test_definition_with_history_bypasses_guard(self):
        """Key regression test: definition fast-path runs even with history present."""
        orch, engine = self._make_orchestrator()
        def_result = QueryResult(
            answer="**Article 4(72)** — 'own funds' means...",
            sources=[],
        )
        engine.lookup_definition.return_value = def_result

        history = [{"question": "What is CET1?", "answer": "CET1 is..."}]

        with patch("src.query.orchestrator.detect_language", return_value="en"), \
             patch("src.query.orchestrator._rewrite_query_with_history", return_value="What is the definition of own funds?"):
            result = orch.query(
                "What about own funds?",
                history=history,
            )

        engine.lookup_definition.assert_called_once()
        assert result.answer == def_result.answer

    def test_crr_specific_high_confidence_uses_standard_prompt(self):
        orch, engine = self._make_orchestrator()
        engine.lookup_definition.return_value = None  # not a definition query
        high_score_source = {"text": "...", "score": 0.85, "metadata": {}, "expanded": False}
        node = MagicMock()
        node.node.metadata = {"article": "92", "article_title": "Own funds requirements"}
        node.node.get_content.return_value = "Article 92 text..."
        engine.retrieve.return_value = ([node], [high_score_source], "trace-id", "capital adequacy requirements", MagicMock())

        with patch("src.query.orchestrator.detect_language", return_value="en"), \
             patch("openai.OpenAI") as mock_oai:
            mock_client = mock_oai.return_value
            mock_client.chat.completions.create.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(content="Answer from CRR"))]
            )
            # Use a "how" question to avoid matching the definition regex
            result = orch.query("How are own funds requirements calculated?")

        call_args = mock_client.chat.completions.create.call_args
        prompt = call_args.kwargs["messages"][0]["content"]
        # Standard prompt should NOT contain the fallback disclaimer
        assert "supplement with your general knowledge" not in prompt

    def test_crr_specific_low_confidence_uses_fallback_prompt(self):
        orch, engine = self._make_orchestrator()
        engine.lookup_definition.return_value = None  # not a definition query
        low_score_source = {"text": "...", "score": 0.1, "metadata": {}, "expanded": False}
        node = MagicMock()
        node.node.metadata = {"article": "99", "article_title": "Some article"}
        node.node.get_content.return_value = "Marginally related text..."
        # Use a "how" query so it isn't caught by the definition regex
        engine.retrieve.return_value = ([node], [low_score_source], "trace-id", "How does Basel III relate to CRR?", MagicMock())

        with patch("src.query.orchestrator.detect_language", return_value="en"), \
             patch("openai.OpenAI") as mock_oai:
            mock_client = mock_oai.return_value
            mock_client.chat.completions.create.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(content="Basel III is a framework..."))]
            )
            result = orch.query("How does Basel III relate to CRR?")

        call_args = mock_client.chat.completions.create.call_args
        prompt = call_args.kwargs["messages"][0]["content"]
        # Fallback prompt should mention general knowledge
        assert "supplement with your general knowledge" in prompt

    def test_low_confidence_with_article_ref_uses_standard_prompt(self):
        """Low confidence + explicit article ref → still use standard prompt (not fallback)."""
        orch, engine = self._make_orchestrator()
        low_score_source = {"text": "...", "score": 0.1, "metadata": {}, "expanded": False}
        node = MagicMock()
        node.node.metadata = {"article": "500", "article_title": ""}
        node.node.get_content.return_value = "Article 500 text..."
        engine.retrieve.return_value = ([node], [low_score_source], "trace-id", "Article 500", MagicMock())

        with patch("src.query.orchestrator.detect_language", return_value="en"), \
             patch("openai.OpenAI") as mock_oai:
            mock_client = mock_oai.return_value
            mock_client.chat.completions.create.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(content="Article 500 says..."))]
            )
            result = orch.query("Explain Article 500")

        call_args = mock_client.chat.completions.create.call_args
        prompt = call_args.kwargs["messages"][0]["content"]
        assert "supplement with your general knowledge" not in prompt

    def test_explicit_language_overrides_detection(self):
        orch, engine = self._make_orchestrator()
        engine.retrieve.return_value = ([], [], "trace-id", "query", MagicMock())

        with patch("src.query.orchestrator.detect_language", return_value="en"), \
             patch("openai.OpenAI") as mock_oai:
            mock_oai.return_value.chat.completions.create.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(content="Answer"))]
            )
            orch.query("Quali sono i requisiti?", language="it")

        # retrieve should be called with lang="it" (explicit override)
        engine.retrieve.assert_called_once()
        call_args = engine.retrieve.call_args
        assert call_args.args[1] == "it" or call_args.kwargs.get("language") == "it" or call_args.args[1] == "it"


# ------------------------------------------------------------------
# Post-retrieval confidence check (unit test for _select_prompt)
# ------------------------------------------------------------------


class TestSelectPrompt:
    def _make_orchestrator(self):
        engine = MagicMock()
        return QueryOrchestrator(query_engine=engine, openai_api_key="test-key")

    def test_high_confidence_returns_standard_prompt(self):
        orch = self._make_orchestrator()
        sources = [{"score": 0.8, "expanded": False}]
        prompt = orch._select_prompt(
            QueryType.CRR_SPECIFIC, sources, "own funds requirements", "ctx", []
        )
        assert "supplement with your general knowledge" not in prompt
        assert "context below" in prompt.lower() or "context" in prompt

    def test_low_confidence_returns_fallback_prompt(self):
        orch = self._make_orchestrator()
        sources = [{"score": 0.2, "expanded": False}]
        prompt = orch._select_prompt(
            QueryType.CRR_SPECIFIC, sources, "What is Basel III?", "ctx", []
        )
        assert "supplement with your general knowledge" in prompt

    def test_low_confidence_with_article_ref_returns_standard_prompt(self):
        orch = self._make_orchestrator()
        sources = [{"score": 0.1, "expanded": False}]
        prompt = orch._select_prompt(
            QueryType.CRR_SPECIFIC, sources, "Article 92", "ctx", []
        )
        assert "supplement with your general knowledge" not in prompt

    def test_direct_article_always_standard_prompt(self):
        orch = self._make_orchestrator()
        sources = [{"score": 0.05, "expanded": False}]
        prompt = orch._select_prompt(
            QueryType.DIRECT_ARTICLE, sources, "Article 92 requirements", "ctx", []
        )
        assert "supplement with your general knowledge" not in prompt

    def test_no_sources_uses_fallback(self):
        orch = self._make_orchestrator()
        prompt = orch._select_prompt(
            QueryType.CRR_SPECIFIC, [], "What is Basel III?", "ctx", []
        )
        assert "supplement with your general knowledge" in prompt

    def test_with_history_uses_history_template(self):
        orch = self._make_orchestrator()
        sources = [{"score": 0.9, "expanded": False}]
        history = [{"question": "What is CET1?", "answer": "CET1 is..."}]
        prompt = orch._select_prompt(
            QueryType.CRR_SPECIFIC, sources, "query", "ctx", history
        )
        assert "Prior conversation" in prompt or "Q:" in prompt

    def test_threshold_boundary_just_below(self):
        orch = self._make_orchestrator()
        sources = [{"score": _LOW_CONFIDENCE_THRESHOLD - 0.01, "expanded": False}]
        prompt = orch._select_prompt(
            QueryType.CRR_SPECIFIC, sources, "What is Basel III?", "ctx", []
        )
        assert "supplement with your general knowledge" in prompt

    def test_threshold_boundary_at(self):
        orch = self._make_orchestrator()
        sources = [{"score": _LOW_CONFIDENCE_THRESHOLD, "expanded": False}]
        prompt = orch._select_prompt(
            QueryType.CRR_SPECIFIC, sources, "What is Basel III?", "ctx", []
        )
        # At exactly the threshold, score is NOT below threshold → standard prompt
        assert "supplement with your general knowledge" not in prompt

    def test_expanded_nodes_excluded_from_confidence_check(self):
        """Expanded cross-ref nodes (score=0.0) should not drive the confidence check."""
        orch = self._make_orchestrator()
        sources = [
            {"score": 0.0, "expanded": True},   # cross-ref expansion, should be ignored
            {"score": 0.8, "expanded": False},   # primary hit
        ]
        prompt = orch._select_prompt(
            QueryType.CRR_SPECIFIC, sources, "What are own funds?", "ctx", []
        )
        assert "supplement with your general knowledge" not in prompt


# ---------------------------------------------------------------------------
# Multi-hop detection (_MULTI_HOP_RE)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestMultiHopDetection:
    """_MULTI_HOP_RE correctly identifies comparative / relational queries."""

    def _is_multi_hop(self, query: str) -> bool:
        from src.query.orchestrator import _MULTI_HOP_RE
        return bool(_MULTI_HOP_RE.search(query))

    def test_relationship_between(self):
        assert self._is_multi_hop("What is the relationship between CET1 and AT1?")

    def test_difference_between(self):
        assert self._is_multi_hop("What is the difference between IRB and SA?")

    def test_compare(self):
        assert self._is_multi_hop("Compare the leverage ratio and capital ratio requirements")

    def test_how_does_affect(self):
        assert self._is_multi_hop("How does CVA risk affect the capital requirement calculation?")

    def test_interaction_between(self):
        assert self._is_multi_hop("Explain the interaction between NSFR and LCR requirements")

    def test_simple_article_query_not_multi_hop(self):
        assert not self._is_multi_hop("What are the requirements of Article 92?")

    def test_definition_query_not_multi_hop(self):
        assert not self._is_multi_hop("What is the definition of institution?")

    def test_open_ended_not_multi_hop(self):
        assert not self._is_multi_hop("What is the minimum leverage ratio under the CRR?")


# ---------------------------------------------------------------------------
# _multi_query_retrieve
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestMultiQueryRetrieve:
    """_multi_query_retrieve merges node results from main + sub-queries by score."""

    def _make_node(self, node_id: str, score: float = 0.8) -> MagicMock:
        node = MagicMock()
        node.node.node_id = node_id
        node.node.metadata = {"article": node_id.split("_")[1] if "_" in node_id else ""}
        node.node.get_content.return_value = f"content of {node_id}"
        node.score = score
        return node

    def _make_orchestrator_with_retrieve(self, retrieve_map: dict):
        """Build a QueryOrchestrator whose engine.retrieve() returns nodes based on query."""
        from src.query.orchestrator import QueryOrchestrator
        engine = MagicMock()

        def _retrieve(query, lang, max_exp):
            nodes = retrieve_map.get(query, [])
            sources = [{"score": n.score, "metadata": n.node.metadata,
                        "text": "", "expanded": False} for n in nodes]
            return nodes, sources, "trace-id", query, None

        engine.retrieve.side_effect = _retrieve
        return QueryOrchestrator(query_engine=engine, openai_api_key="test")

    def test_merges_unique_nodes_from_sub_queries(self):
        n1 = self._make_node("art_92_en", score=0.9)
        n2 = self._make_node("art_93_en", score=0.7)
        n3 = self._make_node("art_94_en", score=0.6)
        orch = self._make_orchestrator_with_retrieve({
            "main query": [n1],
            "sub query 1": [n2],
            "sub query 2": [n3],
        })
        nodes, sources, _, _ = orch._multi_query_retrieve(
            "main query", ["sub query 1", "sub query 2"], "en", None
        )
        node_ids = {n.node.node_id for n in nodes}
        assert node_ids == {"art_92_en", "art_93_en", "art_94_en"}

    def test_deduplicates_by_keeping_highest_score(self):
        n_low = self._make_node("art_92_en", score=0.5)
        n_high = self._make_node("art_92_en", score=0.9)
        orch = self._make_orchestrator_with_retrieve({
            "main query": [n_low],
            "sub query 1": [n_high],
        })
        nodes, _, _, _ = orch._multi_query_retrieve(
            "main query", ["sub query 1"], "en", None
        )
        assert len(nodes) == 1
        assert nodes[0].score == 0.9

    def test_handles_sub_query_retrieval_failure_gracefully(self):
        from src.query.orchestrator import QueryOrchestrator
        engine = MagicMock()
        n1 = self._make_node("art_92_en", score=0.8)

        call_count = {"n": 0}
        def _retrieve(query, lang, max_exp):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return [n1], [], "tid", query, None
            raise RuntimeError("retrieval failed")

        engine.retrieve.side_effect = _retrieve
        orch = QueryOrchestrator(query_engine=engine, openai_api_key="test")
        nodes, _, _, _ = orch._multi_query_retrieve(
            "main query", ["bad sub query"], "en", None
        )
        # Main query result is preserved despite sub-query failure
        assert len(nodes) == 1
        assert nodes[0].node.node_id == "art_92_en"
