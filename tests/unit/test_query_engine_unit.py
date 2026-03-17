"""
Unit tests for QueryEngine internals that can be exercised without Qdrant or OpenAI.

Tests cover:
- _retrieve_with_filters: HYBRID→DEFAULT fallback logic
- synthesis node merging: expanded nodes are deduped and included
- _ref_sort_key: deterministic article-number sort order
- _expand_cross_references: deterministic ordering and cap not consumed by failed fetches
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


def _make_annex_node(node_id: str, annex_id: str, referenced_annexes: str = "") -> MagicMock:
    node = MagicMock()
    node.node.node_id = node_id
    node.node.metadata = {
        "article": "",
        "annex_id": annex_id,
        "level": "ANNEX",
        "referenced_annexes": referenced_annexes,
    }
    node.node.get_content.return_value = f"content of annex {annex_id}"
    node.score = 0.8
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
        def _as_retriever(similarity_top_k, vector_store_query_mode, filters, **kwargs):
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

        def _as_retriever(similarity_top_k, vector_store_query_mode, filters, **kwargs):
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

    def test_configure_settings_resets_prompt_helper(self):
        """_configure_settings() must assign None to Settings._prompt_helper.

        If a prior code path (e.g. the indexer setting llm=None) left a cached
        PromptHelper with a smaller context window, synthesis would repack to the
        wrong token budget.  Resetting to None forces LlamaIndex to rebuild it
        from the current LLM metadata (GPT-4o, 128k context).

        Patch the entire Settings object to avoid LlamaIndex property-setter
        validation (embed_model and llm setters reject non-BaseEmbedding / non-LLM
        values, which would raise before we reach the _prompt_helper reset line).
        """
        from src.query.query_engine import QueryEngine

        qe = QueryEngine.__new__(QueryEngine)
        qe.llm_model = "gpt-4o"
        qe.openai_api_key = "test"

        with patch("src.query.query_engine.Settings") as mock_settings:
            qe._configure_settings()
            assert mock_settings._prompt_helper is None, (
                "_configure_settings() must assign None to Settings._prompt_helper "
                "so LlamaIndex rebuilds it with the current LLM's context window."
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


# ---------------------------------------------------------------------------
# _ref_sort_key
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRefSortKey:
    """_ref_sort_key must order article numbers numerically with alpha suffix tie-break."""

    def _key(self, a: str):
        from src.query.query_engine import _ref_sort_key
        return _ref_sort_key(a)

    def test_plain_numbers_ordered_numerically(self):
        articles = ["114", "26", "92"]
        assert sorted(articles, key=self._key) == ["26", "92", "114"]

    def test_alpha_suffix_after_numeric(self):
        """92a < 92b, and 92 < 92a."""
        assert sorted(["92b", "92", "92a"], key=self._key) == ["92", "92a", "92b"]

    def test_multi_char_suffix(self):
        """92aa sorts after 92a."""
        assert sorted(["92aa", "92a", "92"], key=self._key) == ["92", "92a", "92aa"]

    def test_non_numeric_id_falls_to_front(self):
        """Non-numeric IDs get key (0, id) and sort before numeric ones."""
        result = sorted(["92", "anx_I", "26"], key=self._key)
        assert result[0] == "anx_I"

    def test_mixed_set_stable(self):
        """A realistic set of refs sorts deterministically."""
        refs = {"114", "92", "26", "92a", "4"}
        result = sorted(refs, key=self._key)
        assert result == ["4", "26", "92", "92a", "114"]


# ---------------------------------------------------------------------------
# _expand_cross_references: determinism + cap behaviour
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestExpandCrossReferences:
    """_expand_cross_references must be deterministic and not waste cap slots on failures."""

    def _engine_with_refs(self, refs_csv: str, failing_articles: set[str] | None = None):
        """Build a minimal QueryEngine where one source node has the given refs_csv."""
        from src.query.query_engine import QueryEngine
        from src.indexing.vector_store import VectorStore
        from src.indexing.index_builder import HierarchicalIndexer

        failing_articles = failing_articles or set()

        qe = QueryEngine.__new__(QueryEngine)
        qe.max_cross_ref_expansions = 3
        qe._vector_index = MagicMock()

        # Source node with referenced_articles CSV
        source = _make_node("art_92_en", "92")
        source.node.metadata = {"article": "92", "referenced_articles": refs_csv}

        def fake_retrieve(filters, query_str, top_k):
            # Extract article number from query_str ("Article N")
            art = query_str.split()[-1]
            if art in failing_articles:
                raise RuntimeError(f"Simulated fetch failure for Article {art}")
            return [_make_node(f"art_{art}_en", art)]

        qe._retrieve_with_filters = MagicMock(side_effect=fake_retrieve)
        return qe, [source]

    def test_refs_fetched_in_numeric_order(self):
        """References must be fetched in ascending numeric order, not hash order."""
        qe, source_nodes = self._engine_with_refs("114,26,4")
        qe._expand_cross_references(source_nodes, language=None, limit=3)

        call_args = [call[1]["query_str"] for call in qe._retrieve_with_filters.call_args_list]
        fetched_articles = [q.split()[-1] for q in call_args]
        assert fetched_articles == ["4", "26", "114"], (
            f"Expected numeric order [4, 26, 114] but got {fetched_articles}"
        )

    def test_cap_not_consumed_by_failed_fetch(self):
        """A failed fetch must not count against the expansion cap.

        With limit=2 and refs [26, 93, 114] where article 26 fails:
        both 93 and 114 must be returned (cap still has 2 slots after the failure).

        Note: the source node article ("92") is excluded from refs_to_fetch via
        _seen, so refs must not include the source article number.
        """
        qe, source_nodes = self._engine_with_refs("26,93,114", failing_articles={"26"})
        expanded = qe._expand_cross_references(source_nodes, language=None, limit=2)
        fetched = {n.node.metadata["article"] for n in expanded}
        assert fetched == {"93", "114"}, (
            f"Expected {{'93', '114'}} but got {fetched}. "
            "Failed fetch for article 26 should not consume a cap slot."
        )

    def test_cap_stops_at_limit(self):
        """Exactly `limit` nodes are returned when all fetches succeed."""
        qe, source_nodes = self._engine_with_refs("4,26,92,114")
        expanded = qe._expand_cross_references(source_nodes, language=None, limit=2)
        assert len(expanded) == 2

    def test_empty_refs_returns_empty(self):
        qe, source_nodes = self._engine_with_refs("")
        expanded = qe._expand_cross_references(source_nodes, language=None, limit=3)
        assert expanded == []


# ---------------------------------------------------------------------------
# _expand_cross_references: annex expansion
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestExpandCrossReferencesAnnex:
    """Annex refs in referenced_annexes metadata trigger annex node fetches."""

    def _engine_with_annex_refs(
        self, referenced_annexes: str, failing_annexes: set[str] | None = None
    ):
        """Build a minimal QueryEngine where source node has the given referenced_annexes."""
        from src.query.query_engine import QueryEngine

        failing_annexes = failing_annexes or set()

        qe = QueryEngine.__new__(QueryEngine)
        qe.max_cross_ref_expansions = 5
        qe._vector_index = MagicMock()

        source = _make_node("art_92_en", "92")
        source.node.metadata = {
            "article": "92",
            "referenced_articles": "",
            "referenced_annexes": referenced_annexes,
        }

        def fake_retrieve(filters, query_str, top_k):
            # Annex fetches use "Annex X" as query_str
            parts = query_str.split()
            if parts[0] == "Annex":
                anx = parts[1]
                if anx in failing_annexes:
                    raise RuntimeError(f"Simulated fetch failure for Annex {anx}")
                return [_make_annex_node(f"anx_{anx}_en", anx)]
            # Article fetches
            art = query_str.split()[-1]
            return [_make_node(f"art_{art}_en", art)]

        qe._retrieve_with_filters = MagicMock(side_effect=fake_retrieve)
        return qe, [source]

    def test_annex_refs_fetched(self):
        """Source node with referenced_annexes='I,III' triggers two annex fetches."""
        qe, source_nodes = self._engine_with_annex_refs("I,III")
        expanded = qe._expand_cross_references(source_nodes, language=None, limit=5)
        annex_ids = {n.node.metadata.get("annex_id") for n in expanded}
        assert "I" in annex_ids
        assert "III" in annex_ids

    def test_annex_ref_failure_does_not_consume_cap(self):
        """Failed Annex I fetch doesn't block II and III from being fetched."""
        qe, source_nodes = self._engine_with_annex_refs("I,II,III", failing_annexes={"I"})
        expanded = qe._expand_cross_references(source_nodes, language=None, limit=2)
        annex_ids = {n.node.metadata.get("annex_id") for n in expanded}
        assert "II" in annex_ids
        assert "III" in annex_ids

    def test_annex_refs_fetched_in_roman_order(self):
        """Fetch order is I, II, IV regardless of CSV order."""
        qe, source_nodes = self._engine_with_annex_refs("IV,I,II")
        qe._expand_cross_references(source_nodes, language=None, limit=5)
        annex_calls = [
            call[1]["query_str"]
            for call in qe._retrieve_with_filters.call_args_list
            if call[1]["query_str"].startswith("Annex")
        ]
        fetched_order = [q.split()[1] for q in annex_calls]
        assert fetched_order == ["I", "II", "IV"]


# ---------------------------------------------------------------------------
# Task 3 — get_article returns referenced_external
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGetArticleReferencedExternal:
    def _engine_with_metadata(self, metadata: dict):
        """Build a QueryEngine whose _retrieve_with_filters returns one node with given metadata."""
        from src.query.query_engine import QueryEngine
        from src.indexing.index_builder import HierarchicalIndexer
        from src.indexing.vector_store import VectorStore

        vs = MagicMock(spec=VectorStore)
        idx = MagicMock(spec=HierarchicalIndexer)
        qe = QueryEngine.__new__(QueryEngine)
        qe.vector_store = vs
        qe.indexer = idx
        qe._vector_index = MagicMock()

        node = _make_node("art_92_en", "92")
        node.node.metadata = metadata
        node.node.get_content.return_value = "content"

        qe._retrieve_with_filters = MagicMock(return_value=[node])
        return qe

    def test_get_article_returns_referenced_external_list(self):
        """CSV 'Directive 2013/36/EU,Regulation (EU) No 648/2012' → list of 2 strings."""
        meta = {
            "article": "92",
            "article_title": "Own funds requirements",
            "referenced_articles": "",
            "referenced_external": "Directive 2013/36/EU,Regulation (EU) No 648/2012",
            "language": "en",
        }
        qe = self._engine_with_metadata(meta)
        result = qe.get_article("92")
        assert result is not None
        assert result["referenced_external"] == [
            "Directive 2013/36/EU",
            "Regulation (EU) No 648/2012",
        ]

    def test_get_article_referenced_external_empty_when_absent(self):
        """Missing referenced_external key → empty list."""
        meta = {
            "article": "92",
            "article_title": "Own funds requirements",
            "referenced_articles": "",
            "language": "en",
        }
        qe = self._engine_with_metadata(meta)
        result = qe.get_article("92")
        assert result is not None
        assert result["referenced_external"] == []


# ---------------------------------------------------------------------------
# Task 4 — RETRIEVAL_ALPHA is passed as kwarg to as_retriever
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRetrievalAlpha:
    def _minimal_engine(self):
        from src.query.query_engine import QueryEngine
        from src.indexing.index_builder import HierarchicalIndexer
        from src.indexing.vector_store import VectorStore

        vs = MagicMock(spec=VectorStore)
        idx = MagicMock(spec=HierarchicalIndexer)
        qe = QueryEngine.__new__(QueryEngine)
        qe.vector_store = vs
        qe.indexer = idx
        qe._reranker = None
        qe._vector_index = MagicMock()
        return qe

    def test_alpha_kwarg_passed_to_as_retriever_in_build_engine(self, monkeypatch):
        """_build_engine passes alpha=RETRIEVAL_ALPHA to as_retriever."""
        import src.query.query_engine as qe_module
        monkeypatch.setattr(qe_module, "RETRIEVAL_ALPHA", 0.7)

        qe = self._minimal_engine()
        qe._build_engine(qe._vector_index)

        call_kwargs = qe._vector_index.as_retriever.call_args[1]
        assert call_kwargs.get("alpha") == 0.7

    def test_alpha_kwarg_passed_in_retrieve_with_filters(self, monkeypatch):
        """_retrieve_with_filters passes alpha=RETRIEVAL_ALPHA to as_retriever."""
        import src.query.query_engine as qe_module
        monkeypatch.setattr(qe_module, "RETRIEVAL_ALPHA", 0.3)

        qe = self._minimal_engine()
        node = _make_node("art_4_en", "4")
        qe._vector_index.as_retriever.return_value.retrieve.return_value = [node]

        from llama_index.core.vector_stores.types import MetadataFilters
        qe._retrieve_with_filters(
            filters=MetadataFilters(filters=[]),
            query_str="Article 4",
            top_k=5,
        )

        call_kwargs = qe._vector_index.as_retriever.call_args[1]
        assert call_kwargs.get("alpha") == 0.3
