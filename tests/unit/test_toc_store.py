"""Unit tests for TocStore.

Tests cover HTML parsing against the real CRR_ToC.html, hierarchy extraction,
article number formats, Qdrant enrichment, format_for_prompt, and JSON cache
roundtrip. All tests are offline except for the real HTML parse tests.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.query.toc_store import TocStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TOC_HTML_PATH = Path(__file__).parent.parent.parent / "CRR_ToC.html"


def _make_store(entries: list[dict], language: str = "en") -> TocStore:
    """Create a TocStore with pre-loaded entries (no Qdrant, no file I/O)."""
    store = object.__new__(TocStore)
    store.vector_store = None
    store._entries = {language: entries}
    store._prompt_cache = {}
    return store


def _sample_entries() -> list[dict]:
    return [
        {
            "article": "92",
            "article_title": "Own funds requirements",
            "part": "THREE",
            "title": "I",
            "chapter": "1",
            "section": "",
            "is_annex": False,
            "key_terms": "institution shall at all times satisfy the following own funds requirements",
        },
        {
            "article": "387",
            "article_title": "Subject matter",
            "part": "FOUR",
            "title": "",
            "chapter": "",
            "section": "",
            "is_annex": False,
            "key_terms": "large exposures subject matter",
        },
        {
            "article": "ANNEX_I",
            "article_title": "Classification of off-balance-sheet items",
            "part": "",
            "title": "",
            "chapter": "",
            "section": "",
            "is_annex": True,
            "key_terms": "",
        },
    ]


# ---------------------------------------------------------------------------
# Group 1: HTML parsing (requires CRR_ToC.html)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.skipif(
    not TOC_HTML_PATH.exists(),
    reason="CRR_ToC.html not present in project root",
)
class TestParseHtml:
    def test_article_count(self):
        entries = TocStore._parse_html()
        articles = [e for e in entries if not e.get("is_annex")]
        # CRR as amended by CRR2/CRR3 has 689 articles in the EBA interactive rulebook
        assert len(articles) == 689, f"Expected 689 articles, got {len(articles)}"

    def test_annex_count(self):
        entries = TocStore._parse_html()
        annexes = [e for e in entries if e.get("is_annex")]
        assert len(annexes) == 4, f"Expected 4 annexes, got {len(annexes)}"

    def test_no_duplicate_article_numbers(self):
        entries = TocStore._parse_html()
        articles = [e["article"] for e in entries if not e.get("is_annex")]
        assert len(articles) == len(set(articles)), "Duplicate article numbers found"

    def test_article_92_hierarchy(self):
        """Article 92 should be in Part THREE, Title I, Chapter 1."""
        entries = TocStore._parse_html()
        art92 = next((e for e in entries if e["article"] == "92"), None)
        assert art92 is not None, "Article 92 not found"
        assert art92["part"] == "THREE"
        assert art92["title"] == "I"
        assert art92["chapter"] == "1"
        assert "own funds" in art92["article_title"].lower()

    def test_article_4_is_definitions(self):
        entries = TocStore._parse_html()
        art4 = next((e for e in entries if e["article"] == "4"), None)
        assert art4 is not None, "Article 4 not found"
        assert "definition" in art4["article_title"].lower()

    def test_letter_suffix_articles(self):
        """Articles with letter suffixes (5a, 47a, 429a) must be parsed."""
        entries = TocStore._parse_html()
        article_nums = {e["article"] for e in entries if not e.get("is_annex")}
        # These are known letter-suffix articles in the CRR
        for expected in ("5a", "10a", "47a", "47b", "47c"):
            assert expected in article_nums, f"Letter-suffix article '{expected}' not parsed"

    def test_part_four_articles_have_no_title(self):
        """Part Four (Large Exposures) has articles directly under the Part, no Title."""
        entries = TocStore._parse_html()
        part_four = [
            e for e in entries
            if e["part"] == "FOUR" and not e.get("is_annex")
        ]
        assert len(part_four) > 0, "No articles found for Part Four"
        # Part Four should have no Title subdivision
        assert all(e["title"] == "" for e in part_four), (
            "Part Four articles unexpectedly have Title set"
        )

    def test_all_entries_have_required_fields(self):
        entries = TocStore._parse_html()
        required = {"article", "article_title", "part", "is_annex", "key_terms"}
        for e in entries:
            missing = required - e.keys()
            assert not missing, f"Entry {e.get('article')} missing fields: {missing}"

    def test_annexes_have_correct_ids(self):
        entries = TocStore._parse_html()
        annex_ids = {e["article"] for e in entries if e.get("is_annex")}
        assert annex_ids == {"ANNEX_I", "ANNEX_II", "ANNEX_III", "ANNEX_IV"}


# ---------------------------------------------------------------------------
# Group 2: Qdrant enrichment
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEnrichmentFromQdrant:
    def test_key_terms_populated_from_payloads(self):
        payloads = [
            {
                "article": "92",
                "language": "en",
                "text": "Institutions shall at all times satisfy the own funds requirement.",
            },
            {
                "article": "387",
                "language": "en",
                "_node_content": json.dumps(
                    {"text": "Large exposures subject matter and scope."}
                ),
            },
        ]
        result = TocStore._build_key_terms_map(payloads)
        assert "92" in result
        assert "institution" in result["92"].lower()
        assert "387" in result
        assert "large" in result["387"].lower()

    def test_node_content_fallback(self):
        """When 'text' is empty, fall back to _node_content JSON blob."""
        payloads = [
            {
                "article": "10",
                "language": "en",
                "text": "",
                "_node_content": json.dumps({"text": "Fallback text content here."}),
            }
        ]
        result = TocStore._build_key_terms_map(payloads)
        assert "10" in result
        assert "fallback" in result["10"].lower()

    def test_deduplication_keeps_first(self):
        """Multiple payloads with same article: keep first non-empty text."""
        payloads = [
            {"article": "92", "text": "First text for ninety-two."},
            {"article": "92", "text": "Second text should be ignored."},
        ]
        result = TocStore._build_key_terms_map(payloads)
        assert "first" in result["92"].lower()
        assert "second" not in result["92"].lower()

    def test_key_terms_truncated_to_50_words(self):
        long_text = " ".join([f"word{i}" for i in range(200)])
        payloads = [{"article": "1", "text": long_text}]
        result = TocStore._build_key_terms_map(payloads)
        assert len(result["1"].split()) == 50

    def test_empty_text_skipped(self):
        payloads = [{"article": "99", "text": "", "_node_content": ""}]
        result = TocStore._build_key_terms_map(payloads)
        assert "99" not in result

    def test_invalid_node_content_json_skipped(self):
        payloads = [{"article": "55", "text": "", "_node_content": "not-json"}]
        result = TocStore._build_key_terms_map(payloads)
        assert "55" not in result


# ---------------------------------------------------------------------------
# Group 3: format_for_prompt
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFormatForPrompt:
    def test_contains_article_numbers_and_titles(self):
        store = _make_store(_sample_entries())
        text = store.format_for_prompt("en")
        assert "Art. 92" in text
        assert "Own funds requirements" in text
        assert "Art. 387" in text

    def test_contains_hierarchy_headers(self):
        store = _make_store(_sample_entries())
        text = store.format_for_prompt("en")
        assert "PART THREE" in text
        assert "TITLE I" in text
        assert "CHAPTER 1" in text
        assert "PART FOUR" in text

    def test_key_terms_included_when_present(self):
        store = _make_store(_sample_entries())
        text = store.format_for_prompt("en")
        assert "Key terms:" in text
        assert "institution" in text

    def test_no_key_terms_suffix_when_empty(self):
        entries = [
            {
                "article": "387",
                "article_title": "Subject matter",
                "part": "FOUR",
                "title": "",
                "chapter": "",
                "section": "",
                "is_annex": False,
                "key_terms": "",
            }
        ]
        store = _make_store(entries)
        text = store.format_for_prompt("en")
        assert "Key terms:" not in text

    def test_annexes_listed_at_end(self):
        store = _make_store(_sample_entries())
        text = store.format_for_prompt("en")
        assert "ANNEX I" in text
        # Annex should appear after PART FOUR content
        assert text.index("ANNEX") > text.index("PART FOUR")

    def test_prompt_cached_on_second_call(self):
        store = _make_store(_sample_entries())
        first = store.format_for_prompt("en")
        second = store.format_for_prompt("en")
        assert first is second  # same object (cached)

    def test_returns_empty_string_for_unloaded_language(self):
        store = _make_store([])
        text = store.format_for_prompt("en")
        assert isinstance(text, str)


# ---------------------------------------------------------------------------
# Group 4: JSON cache roundtrip
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestJsonCacheRoundtrip:
    def test_load_from_cache(self, tmp_path):
        """After _build_and_persist writes the cache, load() reads it back."""
        entries = _sample_entries()

        # Mock VectorStore to return empty payloads (no key-terms enrichment)
        mock_vs = MagicMock()
        mock_vs.scroll_payloads.return_value = []

        store = object.__new__(TocStore)
        store.vector_store = mock_vs
        store._entries = {}
        store._prompt_cache = {}

        # Patch paths to use tmp_path
        with (
            patch("src.query.toc_store.TOC_DIR", tmp_path),
            patch.object(TocStore, "_parse_html", return_value=entries),
        ):
            store.load("en")

        # Verify cache file written
        cache_file = tmp_path / "toc_en.json"
        assert cache_file.exists()

        with open(cache_file, encoding="utf-8") as fh:
            data = json.load(fh)
        assert data["language"] == "en"
        assert data["article_count"] == 2  # 2 non-annex entries in sample
        assert data["annex_count"] == 1
        assert len(data["entries"]) == 3

    def test_load_from_existing_cache(self, tmp_path):
        """If cache exists, load() reads it without calling _parse_html."""
        entries = _sample_entries()
        cache_file = tmp_path / "toc_en.json"
        cache_data = {
            "language": "en",
            "source": "CRR_ToC.html",
            "built_at": "2026-01-01T00:00:00+00:00",
            "article_count": 2,
            "annex_count": 1,
            "entries": entries,
        }
        with open(cache_file, "w", encoding="utf-8") as fh:
            json.dump(cache_data, fh)

        mock_vs = MagicMock()
        store = object.__new__(TocStore)
        store.vector_store = mock_vs
        store._entries = {}
        store._prompt_cache = {}

        with (
            patch("src.query.toc_store.TOC_DIR", tmp_path),
            patch.object(TocStore, "_parse_html") as mock_parse,
        ):
            store.load("en")
            mock_parse.assert_not_called()  # should read from cache

        assert store.is_loaded("en")
        assert len(store._entries["en"]) == 3


# ---------------------------------------------------------------------------
# Group 5: is_loaded
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestIsLoaded:
    def test_not_loaded_initially(self):
        store = object.__new__(TocStore)
        store.vector_store = None
        store._entries = {}
        store._prompt_cache = {}
        assert not store.is_loaded("en")
        assert not store.is_loaded("it")

    def test_loaded_after_entries_set(self):
        store = _make_store(_sample_entries(), language="en")
        assert store.is_loaded("en")
        assert not store.is_loaded("it")
