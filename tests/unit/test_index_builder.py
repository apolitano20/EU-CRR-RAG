"""
Unit tests for HierarchicalIndexer._configure_settings() and the no-chunking guarantee.

Regression suite for the bug where `transformations=[]` passed to
`VectorStoreIndex.from_documents()` was silently ignored because an empty list is
falsy in Python:

    transformations = transformations or Settings.transformations
    # [] or [SentenceSplitter()] → [SentenceSplitter()]

The fix: set `Settings.transformations = []` in `_configure_settings()` so the
fallback is also empty.  Embedding is unaffected — it happens in
`_add_nodes_to_index → _get_node_with_embedding`, not in the transformations pipeline.

Tests:
  TestConfigureSettings      — Settings values after _configure_settings()
  TestNoChunkingRegression   — run_transformations() node count == input doc count
  TestParserIndexerParity    — parser output count == transformations output count
                               for multi-article fixtures including long articles
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from llama_index.core import Settings
from llama_index.core.ingestion import run_transformations

from src.indexing.index_builder import HierarchicalIndexer
from src.indexing.vector_store import VectorStore
from src.ingestion.eurlex_ingest import EurLexIngester


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_indexer() -> HierarchicalIndexer:
    """Return an indexer backed by a mock VectorStore (no Qdrant needed)."""
    vs = MagicMock(spec=VectorStore)
    return HierarchicalIndexer(vector_store=vs)


def _parse(html: str, language: str = "en") -> list:
    """Parse HTML with EurLexIngester and return the document list."""
    return EurLexIngester(language=language, local_file="dummy.html")._parse_with_beautifulsoup(html)


# ---------------------------------------------------------------------------
# TestConfigureSettings
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestConfigureSettings:
    """_configure_settings() must set the three guards that prevent LlamaIndex chunking."""

    def setup_method(self):
        self.indexer = _make_indexer()
        self.indexer._configure_settings()

    def test_transformations_is_empty_list(self):
        """Settings.transformations must be [] — the key regression guard.

        If this is not [], LlamaIndex's `from_documents()` fallback
        (`transformations or Settings.transformations`) will use the default
        SentenceSplitter and silently chunk articles.
        """
        assert Settings.transformations == [], (
            "Settings.transformations must be [] to prevent LlamaIndex from chunking "
            "article-level documents. The falsy-empty-list regression: "
            "`[] or Settings.transformations` evaluates to Settings.transformations."
        )

    def test_chunk_size_is_8192(self):
        """Belt-and-suspenders: chunk_size larger than any CRR article."""
        assert Settings.chunk_size == 8192

    def test_chunk_overlap_is_zero(self):
        assert Settings.chunk_overlap == 0

    def test_transformations_not_none(self):
        """[] is truthy-empty, not None — both must be guarded against."""
        assert Settings.transformations is not None

    def test_transformations_contains_no_sentence_splitter(self):
        from llama_index.core.node_parser import SentenceSplitter
        for t in Settings.transformations:
            assert not isinstance(t, SentenceSplitter), (
                "SentenceSplitter found in Settings.transformations — "
                "articles would be chunked on ingest."
            )


# ---------------------------------------------------------------------------
# TestNoChunkingRegression
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestNoChunkingRegression:
    """run_transformations(docs, Settings.transformations) must not multiply documents.

    This directly reproduces the failure mode: before the fix, passing `transformations=[]`
    to `from_documents()` fell through to `Settings.transformations = [SentenceSplitter()]`,
    which split any article exceeding 1024 tokens into multiple nodes.
    """

    def setup_method(self):
        self.indexer = _make_indexer()
        self.indexer._configure_settings()

    def test_short_article_produces_one_node(self, eurlex_html_en):
        docs = _parse(eurlex_html_en)
        assert len(docs) == 3
        result = run_transformations(docs, Settings.transformations)
        assert len(result) == 3

    def test_long_article_not_split(self, eurlex_html_long_article):
        """An article with >4000 chars (>1024 old-default tokens) must remain one node.

        With the old bug this article was split into 2+ chunks by SentenceSplitter.
        """
        docs = _parse(eurlex_html_long_article)
        assert len(docs) == 1, "Parser should produce exactly 1 document for 1 article"
        result = run_transformations(docs, Settings.transformations)
        assert len(result) == 1, (
            f"Expected 1 node after transformations but got {len(result)}. "
            "This means Settings.transformations is not empty and the article is being "
            "chunked. Check that _configure_settings() sets Settings.transformations = []."
        )

    def test_multi_article_count_preserved(self, eurlex_html_part_three_title_i):
        """5 articles (2 long, 3 short) must produce exactly 5 nodes."""
        docs = _parse(eurlex_html_part_three_title_i)
        assert len(docs) == 5, f"Parser should produce 5 docs, got {len(docs)}"
        result = run_transformations(docs, Settings.transformations)
        assert len(result) == 5, (
            f"Expected 5 nodes but got {len(result)} — long articles are being "
            "split by SentenceSplitter. Fix: Settings.transformations = [] in "
            "_configure_settings()."
        )

    def test_node_count_equals_document_count_invariant(self, eurlex_html_part_three_title_i):
        """Invariant: len(run_transformations(docs, Settings.transformations)) == len(docs)."""
        docs = _parse(eurlex_html_part_three_title_i)
        result = run_transformations(docs, Settings.transformations)
        assert len(result) == len(docs), (
            f"Node count ({len(result)}) != document count ({len(docs)}). "
            "Articles are being chunked."
        )


# ---------------------------------------------------------------------------
# TestParserIndexerParity
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestParserIndexerParity:
    """Parser document count must equal what the transformations pipeline produces.

    This catches the class of bugs where the indexer introduces extra nodes that
    the parser did not intend — the root cause of the 2000+ item count regression.
    """

    def setup_method(self):
        self.indexer = _make_indexer()
        self.indexer._configure_settings()

    def _check_parity(self, html: str, language: str = "en") -> tuple[int, int]:
        """Return (parser_count, transformation_count) for the given HTML."""
        docs = _parse(html, language)
        nodes = run_transformations(docs, Settings.transformations)
        return len(docs), len(nodes)

    def test_parity_en_basic(self, eurlex_html_en):
        parser_count, node_count = self._check_parity(eurlex_html_en)
        assert parser_count == node_count

    def test_parity_en_long_article(self, eurlex_html_long_article):
        parser_count, node_count = self._check_parity(eurlex_html_long_article)
        assert parser_count == node_count

    def test_parity_part_three_five_articles(self, eurlex_html_part_three_title_i):
        parser_count, node_count = self._check_parity(eurlex_html_part_three_title_i)
        assert parser_count == node_count, (
            f"Parser produced {parser_count} docs but transformations produced "
            f"{node_count} nodes. The difference ({node_count - parser_count}) "
            "represents articles that were chunked."
        )

    def test_parity_it_basic(self, eurlex_html_it):
        parser_count, node_count = self._check_parity(eurlex_html_it, language="it")
        assert parser_count == node_count

    def test_parity_with_table(self, eurlex_html_with_table):
        parser_count, node_count = self._check_parity(eurlex_html_with_table)
        assert parser_count == node_count

    def test_parity_with_amendment_blocks(self, eurlex_html_with_amendment_blocks):
        """The Article 94 amendment-block fixture must also preserve parity."""
        parser_count, node_count = self._check_parity(eurlex_html_with_amendment_blocks)
        assert parser_count == node_count


# ---------------------------------------------------------------------------
# TestSettingsScope
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSettingsScope:
    """_settings_scope() must restore all mutated Settings values on exit.

    Without this guard, HierarchicalIndexer._configure_settings() sets
    Settings.llm = None and Settings.transformations = [] permanently.
    Any component that runs afterwards (e.g. QueryEngine) would inherit these
    values unless it calls its own _configure_settings().
    """

    def test_settings_restored_after_scope(self):
        """Values set inside _settings_scope() are reverted on exit.

        Note: Settings.embed_model has a property setter that validates BaseEmbedding,
        so we test only the attributes that accept arbitrary values (llm=None,
        transformations, chunk_size, chunk_overlap) or use _configure_settings()
        to drive mutation inside the scope.
        """
        from src.indexing.index_builder import _settings_scope

        # Capture values before entering the scope
        orig_llm = Settings.llm
        orig_chunk_size = Settings.chunk_size
        orig_chunk_overlap = Settings.chunk_overlap
        orig_transformations = list(Settings.transformations)

        with _settings_scope():
            # Simulate what _configure_settings() does
            Settings.llm = None
            Settings.transformations = []
            Settings.chunk_size = 8192
            Settings.chunk_overlap = 0

        assert Settings.llm is orig_llm
        assert Settings.transformations == orig_transformations
        assert Settings.chunk_size == orig_chunk_size
        assert Settings.chunk_overlap == orig_chunk_overlap

    def test_settings_restored_after_configure_settings(self):
        """After build()/load(), the indexer's Settings mutations are unwound.

        This is the real-world scenario: _configure_settings() sets llm=None,
        which must not leak to the QueryEngine that runs afterwards.
        """
        from src.indexing.index_builder import _settings_scope

        orig_llm = Settings.llm

        indexer = _make_indexer()
        with _settings_scope():
            indexer._configure_settings()
            # Inside scope: LLM was mutated by _configure_settings() (Settings.llm = None
            # → LlamaIndex substitutes MockLLM in test env; either way it changed).
            assert Settings.llm is not orig_llm

        # After scope: original llm restored
        assert Settings.llm is orig_llm, (
            "Settings.llm was not restored after _settings_scope() exited. "
            "Indexer mutations must not leak to other components."
        )

    def test_settings_restored_even_on_exception(self):
        """Settings are restored even if the body of the scope raises."""
        from src.indexing.index_builder import _settings_scope

        orig_llm = Settings.llm
        orig_chunk_size = Settings.chunk_size

        with pytest.raises(RuntimeError):
            with _settings_scope():
                Settings.llm = None
                Settings.chunk_size = 1
                raise RuntimeError("boom")

        assert Settings.llm is orig_llm, (
            "Settings.llm must be restored after exception inside _settings_scope()"
        )
        assert Settings.chunk_size == orig_chunk_size
