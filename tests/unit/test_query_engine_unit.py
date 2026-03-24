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

    def test_does_not_call_hybrid_when_default_succeeds(self):
        # DEFAULT is tried first (faster for metadata-filtered lookups); if it returns
        # results HYBRID is never called. Renamed from the old HYBRID-first test when
        # the retrieval order was inverted (DEFAULT first, HYBRID fallback).
        default_nodes = [_make_node("art_4_en", "4")]
        qe = self._engine(hybrid_results=default_nodes, default_results=default_nodes)
        filters = MagicMock()
        qe._retrieve_with_filters(filters=filters, query_str="Article 4", top_k=1)
        # Only one call to as_retriever — DEFAULT was enough
        assert qe._vector_index.as_retriever.call_count == 1
        call_kwargs = qe._vector_index.as_retriever.call_args[1]
        assert call_kwargs["vector_store_query_mode"] == VectorStoreQueryMode.DEFAULT


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
        qe._defs = None  # definitions fast-path disabled in unit tests
        return qe

    def _mock_openai(self, answer="answer"):
        mock_response = MagicMock()
        mock_response.choices[0].message.content = answer
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        return MagicMock(return_value=mock_client), mock_client

    def test_expanded_nodes_included_in_synthesis(self):
        """Query prompt receives content from source + deduped expanded nodes."""
        qe = self._build_query_engine()

        source = _make_node("art_92_en", "92")
        source.node.get_content.return_value = "source content 92"
        expanded = _make_node("art_26_en", "26")
        expanded.node.get_content.return_value = "expanded content 26"

        mock_engine = MagicMock()
        mock_engine.retrieve.return_value = [source]
        qe._engine = mock_engine

        qe._expand_cross_references = MagicMock(return_value=[expanded])
        mock_cls, mock_client = self._mock_openai()

        with patch("src.query.query_engine._normalise_query", side_effect=lambda q: q), \
             patch("src.query.query_engine._detect_direct_article_lookup", return_value=None), \
             patch("src.query.query_engine.openai.OpenAI", mock_cls):
            result = qe.query("What are own funds requirements?")

        # Both nodes' content must appear in the prompt passed to OpenAI
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args[1]["messages"] if "messages" in call_args[1] else call_args[0][0]
        prompt_text = messages[0]["content"]
        assert "source content 92" in prompt_text
        assert "expanded content 26" in prompt_text

    def test_duplicate_expanded_nodes_are_deduped(self):
        """A node already in source_nodes must not appear twice in the prompt."""
        qe = self._build_query_engine()

        source = _make_node("art_92_en", "92")
        source.node.get_content.return_value = "unique content 92"
        # Same node_id as source — should be dropped from expanded
        duplicate = _make_node("art_92_en", "92")
        duplicate.node.get_content.return_value = "unique content 92"

        mock_engine = MagicMock()
        mock_engine.retrieve.return_value = [source]
        qe._engine = mock_engine

        qe._expand_cross_references = MagicMock(return_value=[duplicate])
        mock_cls, mock_client = self._mock_openai()

        with patch("src.query.query_engine._normalise_query", side_effect=lambda q: q), \
             patch("src.query.query_engine._detect_direct_article_lookup", return_value=None), \
             patch("src.query.query_engine.openai.OpenAI", mock_cls):
            result = qe.query("Article 92 requirements")

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args[1]["messages"] if "messages" in call_args[1] else call_args[0][0]
        prompt_text = messages[0]["content"]
        # Content should appear only once (duplicate was deduped)
        assert prompt_text.count("unique content 92") == 1

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
        qe._engine = mock_engine

        qe._expand_cross_references = MagicMock(return_value=[expanded])
        mock_cls, _ = self._mock_openai()

        with patch("src.query.query_engine._normalise_query", side_effect=lambda q: q), \
             patch("src.query.query_engine._detect_direct_article_lookup", return_value=None), \
             patch("src.query.query_engine.openai.OpenAI", mock_cls):
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

        def fake_fetch(conditions, top_k=1):
            # Extract article number from conditions: [("article", "N"), ...]
            art = next(v for k, v in conditions if k == "article")
            if art in failing_articles:
                raise RuntimeError(f"Simulated fetch failure for Article {art}")
            return [_make_node(f"art_{art}_en", art)]

        qe._fetch_nodes_direct = MagicMock(side_effect=fake_fetch)
        return qe, [source]

    def test_refs_fetched_in_numeric_order(self):
        """References must be fetched in ascending numeric order; Article 4 is skipped."""
        qe, source_nodes = self._engine_with_refs("114,26,4")
        qe._expand_cross_references(source_nodes, language=None, limit=3)

        call_args = [call[0][0] for call in qe._fetch_nodes_direct.call_args_list]
        fetched_articles = [next(v for k, v in conds if k == "article") for conds in call_args]
        # Article 4 is skipped by the definitions fast-path guard
        assert fetched_articles == ["26", "114"], (
            f"Expected numeric order [26, 114] (Article 4 skipped) but got {fetched_articles}"
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

        def fake_fetch(conditions, top_k=1):
            # Distinguish article vs annex by which key is present
            cond_dict = dict(conditions)
            if "annex_id" in cond_dict:
                anx = cond_dict["annex_id"]
                if anx in failing_annexes:
                    raise RuntimeError(f"Simulated fetch failure for Annex {anx}")
                return [_make_annex_node(f"anx_{anx}_en", anx)]
            art = cond_dict.get("article", "")
            return [_make_node(f"art_{art}_en", art)]

        qe._fetch_nodes_direct = MagicMock(side_effect=fake_fetch)
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
            dict(call[0][0])
            for call in qe._fetch_nodes_direct.call_args_list
            if "annex_id" in dict(call[0][0])
        ]
        fetched_order = [c["annex_id"] for c in annex_calls]
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


# ---------------------------------------------------------------------------
# _expand_synonyms
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestExpandSynonyms:
    """_expand_synonyms inlines canonical CRR terms next to user paraphrases."""

    def _expand(self, text: str) -> str:
        from src.query.query_engine import _expand_synonyms
        return _expand_synonyms(text)

    def test_expands_preference_shares(self):
        result = self._expand("What are preference shares requirements?")
        assert "preference shares" in result
        assert "Additional Tier 1 instruments" in result

    def test_expands_subordinated_debt(self):
        result = self._expand("Can subordinated debt count as capital?")
        assert "subordinated debt" in result
        assert "Tier 2 instruments" in result

    def test_no_change_when_no_synonym(self):
        q = "What is the minimum CET1 ratio under Article 92?"
        assert self._expand(q) == q

    def test_case_insensitive(self):
        result = self._expand("Are Preference Shares eligible?")
        assert "Additional Tier 1 instruments" in result

    def test_normalise_query_calls_expand_synonyms(self):
        """_normalise_query pipeline includes synonym expansion."""
        from src.query.query_engine import _normalise_query
        result = _normalise_query("What counts as preference shares?")
        assert "Additional Tier 1 instruments" in result


# ---------------------------------------------------------------------------
# _enrich_open_ended_query
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestEnrichOpenEndedQuery:
    """_enrich_open_ended_query appends predicted article numbers to open-ended queries."""

    def _enrich(self, query: str, hints: str) -> str:
        from src.query.query_engine import _enrich_open_ended_query
        with patch("openai.OpenAI") as mock_openai:
            mock_resp = MagicMock()
            mock_resp.choices[0].message.content = hints
            mock_openai.return_value.chat.completions.create.return_value = mock_resp
            return _enrich_open_ended_query(query, api_key="test-key")

    def test_appends_article_hints(self):
        result = self._enrich("What is the minimum leverage ratio?", "429, 429a")
        assert "What is the minimum leverage ratio?" in result
        assert "429" in result

    def test_returns_original_on_invalid_hints(self):
        """Hints that don't look like article numbers (e.g. a full sentence) are ignored."""
        query = "What is the minimum leverage ratio?"
        result = self._enrich(query, "I'm not sure, maybe article 92 or so?")
        assert result == query

    def test_returns_original_on_empty_hints(self):
        query = "What are own funds requirements?"
        result = self._enrich(query, "")
        assert result == query

    def test_returns_original_on_api_failure(self):
        from src.query.query_engine import _enrich_open_ended_query
        with patch("openai.OpenAI", side_effect=RuntimeError("network error")):
            query = "What is the LCR minimum?"
            result = _enrich_open_ended_query(query, api_key="test-key")
            assert result == query


# ---------------------------------------------------------------------------
# _generate_hyde_query
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGenerateHydeQuery:
    """_generate_hyde_query returns a combined HyDE passage + article hints."""

    def _hyde(self, query: str, llm_output: str) -> str:
        from src.query.query_engine import _generate_hyde_query
        with patch("openai.OpenAI") as mock_openai:
            mock_resp = MagicMock()
            mock_resp.choices[0].message.content = llm_output
            mock_openai.return_value.chat.completions.create.return_value = mock_resp
            return _generate_hyde_query(query, api_key="test-key")

    def test_combines_passage_and_article_hints(self):
        llm_output = (
            "PASSAGE: Institutions shall maintain a total risk exposure amount (TREA) "
            "calculated in accordance with Article 92(3).\n"
            "ARTICLES: 92, 93"
        )
        result = self._hyde("What is TREA?", llm_output)
        assert "total risk exposure amount" in result
        assert "[Relevant CRR articles: 92, 93]" in result

    def test_passage_only_when_articles_missing(self):
        """Valid passage with no ARTICLES line → return passage without hints suffix."""
        llm_output = "PASSAGE: Institutions shall meet own funds requirements under Article 92."
        result = self._hyde("What are own funds requirements?", llm_output)
        assert "own funds requirements" in result
        assert "Relevant CRR articles" not in result

    def test_passage_only_when_articles_invalid(self):
        """ARTICLES line containing non-numeric text is ignored."""
        llm_output = (
            "PASSAGE: The leverage ratio is calculated under Article 429.\n"
            "ARTICLES: I'm not sure, maybe 429?"
        )
        result = self._hyde("What is the leverage ratio?", llm_output)
        assert "leverage ratio" in result
        assert "I'm not sure" not in result

    def test_returns_original_on_empty_response(self):
        query = "What is the leverage ratio requirement?"
        result = self._hyde(query, "")
        assert result == query

    def test_returns_original_on_api_failure(self):
        from src.query.query_engine import _generate_hyde_query
        with patch("openai.OpenAI", side_effect=RuntimeError("network error")):
            query = "What is the minimum CET1 ratio?"
            result = _generate_hyde_query(query, api_key="test-key")
            assert result == query


# ---------------------------------------------------------------------------
# _generate_sub_queries
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGenerateSubQueries:
    """_generate_sub_queries breaks multi-hop questions into simpler sub-queries."""

    def _generate(self, query: str, llm_output: str) -> list:
        from src.query.query_engine import _generate_sub_queries
        with patch("openai.OpenAI") as mock_openai:
            mock_resp = MagicMock()
            mock_resp.choices[0].message.content = llm_output
            mock_openai.return_value.chat.completions.create.return_value = mock_resp
            return _generate_sub_queries(query, api_key="test-key")

    def test_returns_up_to_three_sub_queries(self):
        output = "What is the IRB approach?\nWhat are the PD requirements?\nHow are LGD floors applied?"
        result = self._generate("Compare IRB and SA approaches", output)
        assert len(result) == 3

    def test_strips_numbering_from_output(self):
        output = "1. What is CET1?\n2. What is AT1?\n3. What is T2?"
        result = self._generate("Explain capital tiers", output)
        for q in result:
            assert not q[0].isdigit() or q.startswith("What")

    def test_caps_at_three(self):
        output = "Q1\nQ2\nQ3\nQ4\nQ5"
        result = self._generate("Complex query", output)
        assert len(result) <= 3

    def test_returns_empty_on_api_failure(self):
        from src.query.query_engine import _generate_sub_queries
        with patch("openai.OpenAI", side_effect=RuntimeError("timeout")):
            result = _generate_sub_queries("Complex query", api_key="test-key")
            assert result == []


# ---------------------------------------------------------------------------
# ArticleTitleBoostPostprocessor
# ---------------------------------------------------------------------------

def _make_titled_node(
    node_id: str,
    article_title: str,
    score: float = 0.5,
) -> MagicMock:
    node = MagicMock()
    node.node.node_id = node_id
    node.node.metadata = {"article_title": article_title}
    node.node.get_content.return_value = f"content of {node_id}"
    node.score = score
    return node


@pytest.mark.unit
class TestArticleTitleBoostPostprocessor:
    """ArticleTitleBoostPostprocessor: score boost, sort, and truncation logic."""

    def _booster(self, weight: float = 0.15, top_n: int = 3):
        from src.query.query_engine import ArticleTitleBoostPostprocessor
        return ArticleTitleBoostPostprocessor(boost_weight=weight, top_n=top_n)

    def _bundle(self, query: str):
        from llama_index.core.schema import QueryBundle
        return QueryBundle(query_str=query)

    def test_matching_title_boosts_score(self):
        """Query tokens that overlap with article title should increase the score."""
        booster = self._booster()
        node = _make_titled_node("n1", "Own funds requirements", score=0.5)
        result = booster._postprocess_nodes([node], self._bundle("own funds"))
        assert result[0].score > 0.5

    def test_no_match_no_boost(self):
        """Unrelated title should leave score unchanged."""
        booster = self._booster()
        node = _make_titled_node("n1", "Leverage ratio calculation", score=0.5)
        result = booster._postprocess_nodes([node], self._bundle("own funds"))
        assert result[0].score == pytest.approx(0.5)

    def test_empty_title_no_boost(self):
        """Empty article_title should not error and not change score."""
        booster = self._booster()
        node = _make_titled_node("n1", "", score=0.4)
        result = booster._postprocess_nodes([node], self._bundle("own funds"))
        assert result[0].score == pytest.approx(0.4)

    def test_stopwords_ignored(self):
        """Stopwords ('requirements', 'institutions') should not be counted as matches."""
        booster = self._booster()
        # Title and query share only stopwords — no match, no boost
        node = _make_titled_node("n1", "General requirements for institutions", score=0.5)
        result = booster._postprocess_nodes([node], self._bundle("requirements institutions"))
        # After stopword removal both sets are empty or disjoint — score unchanged
        assert result[0].score == pytest.approx(0.5)

    def test_partial_match_proportional(self):
        """Boost should scale with the match_ratio (partial overlap < full overlap)."""
        booster = self._booster(weight=0.15, top_n=2)
        full_match = _make_titled_node("n_full", "capital ratio", score=0.5)
        partial_match = _make_titled_node("n_partial", "capital leverage ratio", score=0.5)
        q = self._bundle("capital ratio")
        r_full = booster._postprocess_nodes([full_match], q)
        booster2 = self._booster(weight=0.15, top_n=2)
        full_match2 = _make_titled_node("n_full2", "capital ratio", score=0.5)
        partial_match2 = _make_titled_node("n_partial2", "capital leverage ratio", score=0.5)
        r_partial = booster2._postprocess_nodes([partial_match2], q)
        assert r_full[0].score >= r_partial[0].score

    def test_sorts_and_truncates(self):
        """Output must be sorted descending by score and limited to top_n."""
        booster = self._booster(weight=0.15, top_n=2)
        nodes = [
            _make_titled_node("n1", "Own funds requirements", score=0.9),
            _make_titled_node("n2", "Leverage ratio", score=0.8),
            _make_titled_node("n3", "Eligible capital", score=0.7),
        ]
        result = booster._postprocess_nodes(nodes, self._bundle("own funds"))
        assert len(result) == 2
        assert result[0].score >= result[1].score

    def test_abbreviation_expansion_matches(self):
        """Expanded abbreviations in query should match title tokens."""
        # After abbreviation expansion, "TREA" becomes "Total Risk Exposure Amount"
        booster = self._booster()
        node = _make_titled_node("n1", "Total Risk Exposure Amount", score=0.4)
        result = booster._postprocess_nodes(
            [node], self._bundle("Total Risk Exposure Amount")
        )
        assert result[0].score > 0.4

    def test_zero_weight_no_boost(self):
        """weight=0 means no modification to scores."""
        booster = self._booster(weight=0.0, top_n=3)
        node = _make_titled_node("n1", "Own funds requirements", score=0.5)
        result = booster._postprocess_nodes([node], self._bundle("own funds"))
        assert result[0].score == pytest.approx(0.5)
