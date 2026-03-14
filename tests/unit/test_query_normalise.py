"""Unit tests for _normalise_query, abbreviation expansion, and direct-lookup detection."""
from __future__ import annotations

import pytest

from src.query.query_engine import _detect_direct_article_lookup, _normalise_query


class TestNormaliseQuery:
    def test_art_dot_number(self):
        assert _normalise_query("art. 92 requirements") == "Article 92 requirements"

    def test_art_no_dot(self):
        assert _normalise_query("art 92 requirements") == "Article 92 requirements"

    def test_article_lowercase(self):
        assert _normalise_query("article 92 requirements") == "Article 92 requirements"

    def test_article_uppercase(self):
        assert _normalise_query("ARTICLE 92") == "Article 92"

    def test_article_already_canonical(self):
        assert _normalise_query("Article 92 sets out own funds") == "Article 92 sets out own funds"

    def test_multiple_references(self):
        result = _normalise_query("see art. 92 and art. 114 for details")
        assert "Article 92" in result
        assert "Article 114" in result

    def test_alphanumeric_article_number(self):
        """Article numbers like 141b should be preserved."""
        result = _normalise_query("art. 141b requirements")
        assert "Article 141b" in result

    def test_no_article_reference_unchanged(self):
        original = "What are the capital requirements?"
        assert _normalise_query(original) == original

    def test_empty_string(self):
        assert _normalise_query("") == ""


class TestAbbreviationExpansion:
    def test_cet1_expanded(self):
        result = _normalise_query("What is the CET1 ratio requirement?")
        assert "CET1 (Common Equity Tier 1)" in result

    def test_lcr_expanded(self):
        result = _normalise_query("Explain the LCR calculation.")
        assert "LCR (Liquidity Coverage Ratio)" in result

    def test_nsfr_expanded(self):
        result = _normalise_query("NSFR requirements under CRR")
        assert "NSFR (Net Stable Funding Ratio)" in result

    def test_rwa_expanded(self):
        result = _normalise_query("How are RWA calculated?")
        assert "RWA (risk-weighted assets)" in result

    def test_eba_expanded(self):
        result = _normalise_query("EBA guidelines on capital")
        assert "EBA (European Banking Authority)" in result

    def test_multiple_abbreviations(self):
        result = _normalise_query("CET1 and LCR requirements")
        assert "CET1 (Common Equity Tier 1)" in result
        assert "LCR (Liquidity Coverage Ratio)" in result

    def test_abbreviation_combined_with_article_ref(self):
        result = _normalise_query("CET1 requirements under art. 92")
        assert "CET1 (Common Equity Tier 1)" in result
        assert "Article 92" in result

    def test_lowercase_not_expanded(self):
        """Abbreviation expansion is case-sensitive — lowercase must not match."""
        result = _normalise_query("What are the rwa calculations?")
        assert "risk-weighted assets" not in result

    def test_no_abbreviation_unchanged(self):
        original = "What are the capital requirements?"
        assert _normalise_query(original) == original


class TestDirectArticleLookup:
    def test_bare_article(self):
        assert _detect_direct_article_lookup("Article 92") == "92"

    def test_what_does_article_say(self):
        assert _detect_direct_article_lookup("What does Article 92 say?") == "92"

    def test_explain_article(self):
        assert _detect_direct_article_lookup("Explain Article 114") == "114"

    def test_describe_article(self):
        assert _detect_direct_article_lookup("Describe Article 6") == "6"

    def test_alphanumeric_article(self):
        assert _detect_direct_article_lookup("Article 141b") == "141b"

    def test_article_with_trailing_question(self):
        assert _detect_direct_article_lookup("Article 92?") == "92"

    def test_complex_query_with_single_article_matched(self):
        # Broadened: any query mentioning exactly one article triggers direct lookup
        assert _detect_direct_article_lookup("What are the requirements under Article 92?") == "92"

    def test_multiple_articles_not_matched(self):
        assert _detect_direct_article_lookup("Article 92 and Article 93") is None

    def test_article_with_extra_words_matched(self):
        # Broadened: single article mention triggers direct lookup regardless of surrounding text
        assert _detect_direct_article_lookup("Article 92 requirements for capital") == "92"

    def test_no_article_not_matched(self):
        assert _detect_direct_article_lookup("What are own funds requirements?") is None

    def test_empty_string_not_matched(self):
        assert _detect_direct_article_lookup("") is None
