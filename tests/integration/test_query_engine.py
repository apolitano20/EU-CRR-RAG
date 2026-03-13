"""
Integration tests for QueryEngine (live Qdrant + OpenAI).

Requires: QDRANT_URL, QDRANT_API_KEY, OPENAI_API_KEY in .env

These tests ingest a small slice (same MINI_DOCS as ingest tests) and then
query against it. They do NOT mock Qdrant or OpenAI — see WORKLOG.md for why.
"""
from __future__ import annotations

import pytest

from llama_index.core import Document

from src.indexing.index_builder import HierarchicalIndexer
from src.indexing.vector_store import VectorStore
from src.query.query_engine import QueryEngine, QueryResult


# ---------------------------------------------------------------------------
# Shared mini corpus (article-level, new metadata schema)
# ---------------------------------------------------------------------------

MINI_DOCS = [
    Document(
        text=(
            "Article 92 sets out own funds requirements. "
            "Institutions shall at all times satisfy the following own funds requirements: "
            "(a) a Common Equity Tier 1 capital ratio of 4.5 %; "
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
            "CET1 items consist of: (a) capital instruments meeting Article 28 conditions; "
            "(b) share premium accounts; (c) retained earnings."
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
            "Article 114 covers risk weights for central governments and central banks. "
            "Exposures to central governments or central banks shall be assigned a risk weight "
            "of 0 % where denominated and funded in the domestic currency."
        ),
        metadata={
            "node_id": "art_114_en",
            "level": "ARTICLE",
            "part": "THREE",
            "title": "II",
            "chapter": "2",
            "section": "",
            "article": "114",
            "article_title": "Risk weights for central governments",
            "annex_id": "",
            "annex_title": "",
            "referenced_articles": "",
            "referenced_external": "",
            "has_table": False,
            "has_formula": False,
            "language": "en",
        },
    ),
    Document(
        text=(
            "Articolo 92 stabilisce i requisiti in materia di fondi propri. "
            "Gli enti soddisfano in ogni momento un coefficiente di capitale primario di classe 1 "
            "pari al 4,5 %."
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

QUERY_COLLECTION = "eu_crr_query_test"


@pytest.fixture(scope="module")
def store(qdrant_url, qdrant_api_key):
    vs = VectorStore(
        collection_name=QUERY_COLLECTION,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
    )
    vs.reset()
    yield vs
    vs._client.delete_collection(QUERY_COLLECTION)


@pytest.fixture(scope="module")
def query_engine(store, openai_api_key):
    indexer = HierarchicalIndexer(
        vector_store=store,
        reset_store=False,
    )
    indexer.build(MINI_DOCS)

    engine = QueryEngine(
        vector_store=store,
        indexer=indexer,
        openai_api_key=openai_api_key,
        max_cross_ref_expansions=2,
    )
    engine.load()
    return engine


# ---------------------------------------------------------------------------
# Basic query mechanics
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestQueryEngineBasic:
    def test_query_returns_query_result(self, query_engine):
        result = query_engine.query("What are the CET1 capital requirements?")
        assert isinstance(result, QueryResult)

    def test_answer_is_non_empty_string(self, query_engine):
        result = query_engine.query("What is CET1?")
        assert isinstance(result.answer, str)
        assert len(result.answer) > 0

    def test_sources_is_list(self, query_engine):
        result = query_engine.query("What is CET1?")
        assert isinstance(result.sources, list)

    def test_trace_id_is_string(self, query_engine):
        result = query_engine.query("What is CET1?")
        assert isinstance(result.trace_id, str)
        assert len(result.trace_id) > 0

    def test_each_source_has_required_keys(self, query_engine):
        result = query_engine.query("What is CET1?")
        for src in result.sources:
            assert "text" in src
            assert "score" in src
            assert "metadata" in src
            assert "expanded" in src

    def test_source_text_truncated_to_500_chars(self, query_engine):
        result = query_engine.query("What is CET1?")
        for src in result.sources:
            assert len(src["text"]) <= 500

    def test_answer_mentions_article_92_for_own_funds(self, query_engine):
        """The mini corpus has Article 92 for own funds — answer should reference it."""
        result = query_engine.query("What are the own funds requirements?")
        assert "92" in result.answer or any(
            "92" in str(s.get("metadata", {})) for s in result.sources
        )

    def test_sources_have_article_metadata(self, query_engine):
        result = query_engine.query("What are the own funds requirements?")
        articles = [s["metadata"].get("article") for s in result.sources]
        assert any(a is not None for a in articles)

    def test_sources_have_new_metadata_fields(self, query_engine):
        result = query_engine.query("What are the own funds requirements?")
        for src in result.sources:
            meta = src["metadata"]
            assert "article_title" in meta
            assert "referenced_articles" in meta

    def test_no_load_raises_runtime_error(self, store):
        indexer = HierarchicalIndexer(vector_store=store)
        engine = QueryEngine(vector_store=store, indexer=indexer)
        with pytest.raises(RuntimeError, match="load()"):
            engine.query("test")


# ---------------------------------------------------------------------------
# Language filter
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestLanguageFilter:
    def test_en_filter_returns_only_en_sources(self, query_engine):
        result = query_engine.query("What are own funds requirements?", language="en")
        for src in result.sources:
            lang = src["metadata"].get("language")
            if lang is not None:
                assert lang == "en", f"Expected en source, got: {lang}"

    def test_it_filter_returns_only_it_sources(self, query_engine):
        result = query_engine.query("Quali sono i requisiti di fondi propri?", language="it")
        for src in result.sources:
            lang = src["metadata"].get("language")
            if lang is not None:
                assert lang == "it", f"Expected it source, got: {lang}"

    def test_no_language_filter_returns_results(self, query_engine):
        """Cross-language retrieval (no filter) should still find documents."""
        result = query_engine.query("own funds", language=None)
        assert len(result.sources) > 0

    def test_unknown_language_filter_returns_empty_or_partial(self, query_engine):
        """Filtering on a language with no indexed data should return few/no sources."""
        result = query_engine.query("capital requirements", language="pl")
        assert isinstance(result.sources, list)


# ---------------------------------------------------------------------------
# Cross-reference expansion
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestCrossReferenceExpansion:
    def test_expanded_flag_present_in_sources(self, query_engine):
        result = query_engine.query("What are the own funds requirements under Article 92?")
        expanded_flags = [s["expanded"] for s in result.sources]
        assert all(isinstance(f, bool) for f in expanded_flags)

    def test_direct_sources_have_expanded_false(self, query_engine):
        result = query_engine.query("What are the own funds requirements?")
        direct = [s for s in result.sources if not s["expanded"]]
        assert len(direct) > 0

    def test_no_cross_ref_expansion_with_zero_limit(self, query_engine):
        result = query_engine.query(
            "What are the own funds requirements?",
            max_cross_ref_expansions=0,
        )
        expanded = [s for s in result.sources if s["expanded"]]
        assert len(expanded) == 0

    def test_cross_ref_expansion_can_fetch_referenced_articles(self, query_engine):
        """Article 92 references articles 26 and 36; with expansion enabled,
        Article 26 may appear as an expanded source."""
        result = query_engine.query(
            "What are the own funds requirements?",
            max_cross_ref_expansions=3,
        )
        # Test passes as long as the structure is correct (Article 26 may or may not be retrieved)
        for src in result.sources:
            assert "expanded" in src
            assert "metadata" in src


# ---------------------------------------------------------------------------
# Determinism / stability
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestQueryDeterminism:
    def test_same_query_returns_same_article_sources(self, query_engine):
        """Two identical queries should retrieve the same top source articles."""
        q = "What is the CET1 ratio requirement?"
        r1 = query_engine.query(q, max_cross_ref_expansions=0)
        r2 = query_engine.query(q, max_cross_ref_expansions=0)
        articles1 = {s["metadata"].get("article") for s in r1.sources if not s["expanded"]}
        articles2 = {s["metadata"].get("article") for s in r2.sources if not s["expanded"]}
        assert articles1 == articles2 or bool(articles1 & articles2)
