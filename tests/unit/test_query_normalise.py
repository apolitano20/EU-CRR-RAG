"""Unit tests for _normalise_query, abbreviation expansion, and direct-lookup detection."""
from __future__ import annotations

import pytest

from src.query.query_engine import _detect_direct_article_lookup, _expand_article_ranges, _normalise_query


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

    # --- Item 1 regression: external directive/regulation citations ---

    def test_external_directive_ref_not_matched(self):
        """Article 10 of Directive 2014/59/EU must not trigger direct lookup."""
        assert _detect_direct_article_lookup(
            "Article 10 of Directive 2014/59/EU"
        ) is None

    def test_external_regulation_ref_not_matched(self):
        """Article 4(1)(40) of Regulation (EU) No 648/2012 must not trigger direct lookup."""
        assert _detect_direct_article_lookup(
            "What is covered by Article 4 of Regulation (EU) No 648/2012?"
        ) is None

    def test_external_ref_stripped_leaving_single_crr_article(self):
        """After stripping external ref, a single remaining CRR article triggers lookup."""
        # "Article 92 and Article 10 of Directive 2014/59/EU" — external stripped, 92 remains
        result = _detect_direct_article_lookup(
            "Article 92 and Article 10 of Directive 2014/59/EU"
        )
        assert result == "92"

    def test_multi_article_external_ref_not_matched(self):
        """Articles 74 and 83 of Directive 2013/36/EU — both external, no CRR article."""
        assert _detect_direct_article_lookup(
            "Do Articles 74 and 83 of Directive 2013/36/EU apply to CRR reporting?"
        ) is None

    # --- Item 1 regression: coordinated bare numbers ---

    def test_coordinated_bare_numbers_not_matched(self):
        """'Article 92 and 93' has bare '93' — must not trigger single-article lookup."""
        assert _detect_direct_article_lookup("How do Article 92 and 93 relate?") is None

    def test_coordinated_comma_bare_numbers_not_matched(self):
        """'Article 92, 93' — comma-separated bare numbers."""
        assert _detect_direct_article_lookup("Article 92, 93 and 94 requirements") is None

    def test_coordinated_articles_keyword_not_matched(self):
        """'Articles 89 to 91' phrasing — bare run after initial ref."""
        assert _detect_direct_article_lookup("See Articles 89 and 90 for details") is None


class TestExpandArticleRanges:
    """Fix 4b: _expand_article_ranges expands 'Articles N to M' in query strings."""

    def test_range_to_expanded(self):
        assert _normalise_query("Articles 89 to 91") == "Article 89 Article 90 Article 91"

    def test_range_large_gap_not_expanded(self):
        """Sanity cap: ranges wider than 20 articles are left unchanged."""
        result = _normalise_query("Articles 1 to 575")
        assert "Articles 1 to 575" in result

    def test_range_already_normalised_passthrough(self):
        """No range pattern → query unchanged by range expander."""
        original = "What are the own funds requirements under Article 92?"
        assert _normalise_query(original) == original

    def test_range_expand_helper_directly(self):
        assert _expand_article_ranges("Articles 10 to 12") == "Article 10 Article 11 Article 12"

    def test_range_case_insensitive(self):
        result = _expand_article_ranges("articles 5 to 7")
        assert "Article 5" in result
        assert "Article 6" in result
        assert "Article 7" in result

    def test_range_combined_with_abbreviation(self):
        result = _normalise_query("CET1 requirements in Articles 26 to 28")
        assert "CET1 (Common Equity Tier 1)" in result
        assert "Article 26" in result
        assert "Article 28" in result
