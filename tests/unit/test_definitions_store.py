"""Unit tests for DefinitionsStore and related query_engine helpers.

All tests are offline: no Qdrant, no OpenAI.
"""
from __future__ import annotations

import pytest

from src.query.definitions_store import DefinitionsStore
from src.query.query_engine import (
    QueryResult,
    _detect_definition_query,
    _detect_direct_article_lookup,
    _normalise_query,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

SAMPLE_TEXT = (
    "Some preamble text. "
    "(1) \u2018institution\u2019 means a credit institution or an investment firm; "
    "(2) \u2018credit institution\u2019 means an undertaking whose business is to take "
    "deposits or other repayable funds from the public and to grant credits for its own "
    "account; "
    "(3) \u2018investment firm\u2019 means a person as defined in Directive 2014/65/EU, "
    "subject to the exceptions set out therein; "
    "(4) \u2018financial institution\u2019 means an undertaking other than an institution, "
    "the principal activity of which is to acquire holdings."
)

SAMPLE_TEXT_IT = (
    "Testo preambolo. "
    "(1) \u00abente\u00bb indica un ente creditizio o un\u2019impresa di investimento; "
    "(2) \u00abente creditizio\u00bb indica un'impresa la cui attivit\u00e0 consiste "
    "nel raccogliere depositi."
)


def _make_store_with_data(lang: str, definitions: list[dict]) -> DefinitionsStore:
    """Create a DefinitionsStore with in-memory data (no Qdrant needed)."""
    store = object.__new__(DefinitionsStore)
    store.vector_store = None
    store._definitions = {}
    store._term_index = {}
    store._definitions[lang] = {d["number"]: d for d in definitions}
    store._term_index[lang] = {
        d["term"].lower(): d for d in definitions if d.get("term")
    }
    return store


# ---------------------------------------------------------------------------
# Group 1: _parse() static method
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestParse:
    def test_count(self):
        defs = DefinitionsStore._parse(SAMPLE_TEXT)
        assert len(defs) == 4

    def test_number_extraction(self):
        defs = DefinitionsStore._parse(SAMPLE_TEXT)
        numbers = [d["number"] for d in defs]
        assert numbers == ["1", "2", "3", "4"]

    def test_term_extraction(self):
        defs = DefinitionsStore._parse(SAMPLE_TEXT)
        assert defs[0]["term"] == "institution"
        assert defs[1]["term"] == "credit institution"
        assert defs[2]["term"] == "investment firm"
        assert defs[3]["term"] == "financial institution"

    def test_multi_line_collapsed(self):
        text = "(1) 'foo'\nmeans\na bar; (2) 'baz' means qux"
        defs = DefinitionsStore._parse(text)
        assert len(defs) == 2
        assert "\n" not in defs[0]["text"]

    def test_sub_items_not_split(self):
        # (a), (b) inside a definition should NOT create new entries
        text = "(1) 'institution' means any of the following: (a) a bank; (b) a firm."
        defs = DefinitionsStore._parse(text)
        assert len(defs) == 1
        assert defs[0]["number"] == "1"
        assert "(a)" in defs[0]["text"]

    def test_empty_text(self):
        defs = DefinitionsStore._parse("")
        assert defs == []

    def test_no_numbered_items(self):
        defs = DefinitionsStore._parse("No definitions here.")
        assert defs == []

    def test_italian_guillemets(self):
        defs = DefinitionsStore._parse(SAMPLE_TEXT_IT)
        assert len(defs) == 2
        assert defs[0]["term"] == "ente"
        assert defs[1]["term"] == "ente creditizio"

    def test_term_defaults_to_empty_when_no_quote(self):
        text = "(1) institution means a credit institution or investment firm."
        defs = DefinitionsStore._parse(text)
        assert defs[0]["term"] == ""
        assert defs[0]["number"] == "1"


# ---------------------------------------------------------------------------
# Group 2: lookup methods
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLookup:
    @pytest.fixture
    def store(self):
        defs = DefinitionsStore._parse(SAMPLE_TEXT)
        return _make_store_with_data("en", defs)

    def test_lookup_by_term_exact(self, store):
        entry = store.lookup_by_term("institution", "en")
        assert entry is not None
        assert entry["number"] == "1"

    def test_lookup_by_term_case_insensitive(self, store):
        entry = store.lookup_by_term("Institution", "en")
        assert entry is not None
        assert entry["number"] == "1"

    def test_lookup_by_term_multi_word(self, store):
        entry = store.lookup_by_term("credit institution", "en")
        assert entry is not None
        assert entry["number"] == "2"

    def test_lookup_by_term_not_found(self, store):
        assert store.lookup_by_term("nonexistent term", "en") is None

    def test_lookup_by_number_exact(self, store):
        entry = store.lookup_by_number("3", "en")
        assert entry is not None
        assert entry["term"] == "investment firm"

    def test_lookup_by_number_int_coercion(self, store):
        entry = store.lookup_by_number(3, "en")
        assert entry is not None
        assert entry["number"] == "3"

    def test_lookup_by_number_not_found(self, store):
        assert store.lookup_by_number("999", "en") is None

    def test_unloaded_language_returns_none(self, store):
        assert store.lookup_by_term("institution", "fr") is None
        assert store.lookup_by_number("1", "fr") is None

    def test_is_loaded(self, store):
        assert store.is_loaded("en") is True
        assert store.is_loaded("it") is False


# ---------------------------------------------------------------------------
# Group 3: _detect_definition_query()
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDetectDefinitionQuery:
    @pytest.mark.parametrize(
        "query, expected",
        [
            ("what is institution?", "institution"),
            ("What is an institution?", "institution"),
            ("What is the definition of institution?", "institution"),
            ("define institution", "institution"),
            ("definition of institution", "institution"),
            ("meaning of institution", "institution"),
            ("what does institution mean?", "institution"),
            ("what does institution mean", "institution"),
            # Article 4(N) pattern
            ("Article 4(1)", "#1"),
            ("article 4 (42)", "#42"),
            # Multi-word terms
            ("what is credit institution?", "credit institution"),
        ],
    )
    def test_detection(self, query, expected):
        result = _detect_definition_query(_normalise_query(query))
        assert result == expected

    def test_generic_article_4_not_matched(self):
        # "Explain Article 4" — should not trigger _detect_definition_query
        result = _detect_definition_query(_normalise_query("Explain Article 4"))
        assert result is None

    def test_non_definition_query_returns_none(self):
        assert _detect_definition_query("What are the CET1 requirements under Article 92?") is None
        assert _detect_definition_query("How is leverage ratio calculated?") is None


# ---------------------------------------------------------------------------
# Group 4: QueryEngine.lookup_definition() — tested with a mock engine
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLookupDefinitionMethod:
    """Test QueryEngine.lookup_definition() without needing a full engine."""

    def _make_engine_with_store(self, definitions: list[dict]) -> object:
        """Build a minimal QueryEngine-like object with an injected DefinitionsStore."""
        from src.query.query_engine import QueryEngine
        from src.indexing.index_builder import HierarchicalIndexer

        # Build a hollow QueryEngine (no Qdrant, no index)
        engine = object.__new__(QueryEngine)
        engine.vector_store = None
        engine.indexer = None
        engine.openai_api_key = None
        engine.llm_model = "gpt-4o-mini"
        engine.max_cross_ref_expansions = 3
        engine.use_reranker = False
        engine._engine = None
        engine._vector_index = None
        engine._engine_cache = {}
        import threading
        engine._engine_cache_lock = threading.Lock()
        engine._reranker = None

        store = _make_store_with_data("en", definitions)
        engine._defs = store
        return engine

    def test_known_term_returns_query_result(self):
        defs = DefinitionsStore._parse(SAMPLE_TEXT)
        engine = self._make_engine_with_store(defs)
        result = engine.lookup_definition("what is institution?", "en")
        assert result is not None
        assert isinstance(result, QueryResult)
        assert "institution" in result.answer.lower()
        assert result.sources[0]["metadata"]["article"] == "4"

    def test_unknown_term_returns_none(self):
        defs = DefinitionsStore._parse(SAMPLE_TEXT)
        engine = self._make_engine_with_store(defs)
        result = engine.lookup_definition("what is leverage ratio?", "en")
        assert result is None

    def test_defs_none_returns_none(self):
        from src.query.query_engine import QueryEngine
        engine = object.__new__(QueryEngine)
        engine._defs = None
        result = engine.lookup_definition("what is institution?", "en")
        assert result is None

    def test_generic_article_4_returns_summary(self):
        defs = DefinitionsStore._parse(SAMPLE_TEXT)
        engine = self._make_engine_with_store(defs)
        result = engine.lookup_definition("Explain Article 4", "en")
        assert result is not None
        assert "definitions" in result.answer.lower()
        assert result.sources[0]["metadata"]["article"] == "4"

    def test_article_4_number_lookup(self):
        defs = DefinitionsStore._parse(SAMPLE_TEXT)
        engine = self._make_engine_with_store(defs)
        result = engine.lookup_definition("Article 4(2)", "en")
        assert result is not None
        assert "credit institution" in result.answer.lower()
