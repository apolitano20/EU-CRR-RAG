"""
Unit tests for the DOM-based hierarchy extraction functions in eurlex_ingest.py.

Tests _extract_hierarchy() which parses EUR-Lex parent div IDs into hierarchy dicts.
All tests are pure (no file I/O, no network).
"""
import pytest

from src.ingestion.eurlex_ingest import _extract_hierarchy


@pytest.mark.unit
class TestExtractHierarchy:
    def test_full_path(self):
        result = _extract_hierarchy("prt_III.tis_I.cpt_1.sct_1")
        assert result == {"part": "III", "title": "I", "chapter": "1", "section": "1"}

    def test_part_and_title_only(self):
        result = _extract_hierarchy("prt_ONE.tis_II")
        assert result == {"part": "ONE", "title": "II"}

    def test_part_only(self):
        result = _extract_hierarchy("prt_SEVEN")
        assert result == {"part": "SEVEN"}

    def test_chapter_only(self):
        result = _extract_hierarchy("cpt_3")
        assert result == {"chapter": "3"}

    def test_with_subsection(self):
        result = _extract_hierarchy("prt_I.tis_II.cpt_1.sct_2.sbs_1")
        assert result["subsection"] == "1"
        assert result["section"] == "2"

    def test_empty_string_returns_empty_dict(self):
        result = _extract_hierarchy("")
        assert result == {}

    def test_unknown_prefix_ignored(self):
        result = _extract_hierarchy("xxx_123.prt_IV")
        assert result == {"part": "IV"}
        assert "xxx" not in result

    def test_roman_numerals_preserved(self):
        result = _extract_hierarchy("prt_VIII.tis_III.cpt_2")
        assert result["part"] == "VIII"
        assert result["title"] == "III"

    def test_part_sevena_style(self):
        """EUR-Lex sometimes uses combined identifiers like 'SEVENA'."""
        result = _extract_hierarchy("prt_SEVENA.tis_I")
        assert result["part"] == "SEVENA"

    def test_section_identifier_numeric(self):
        result = _extract_hierarchy("prt_TWO.tis_I.cpt_1.sct_3")
        assert result["section"] == "3"

    def test_does_not_mutate_on_repeated_calls(self):
        r1 = _extract_hierarchy("prt_I.tis_II.cpt_1")
        r2 = _extract_hierarchy("prt_IV.tis_III")
        assert r1["part"] == "I"
        assert r2["part"] == "IV"


@pytest.mark.unit
class TestExtractArticleNumber:
    """Article number is extracted by stripping 'art_' prefix from the div id."""

    def _extract(self, div_id: str) -> str:
        """Helper mirroring the logic in _process_article_div."""
        assert div_id.startswith("art_")
        return div_id[4:]

    def test_numeric_article(self):
        assert self._extract("art_92") == "92"

    def test_article_with_letter_suffix(self):
        assert self._extract("art_429a") == "429a"

    def test_article_1(self):
        assert self._extract("art_1") == "1"

    def test_article_large_number(self):
        assert self._extract("art_521b") == "521b"
