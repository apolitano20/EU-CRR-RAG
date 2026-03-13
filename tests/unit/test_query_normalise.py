"""Unit tests for _normalise_query (article reference canonicalisation)."""
from __future__ import annotations

import pytest

from src.query.query_engine import _normalise_query


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
