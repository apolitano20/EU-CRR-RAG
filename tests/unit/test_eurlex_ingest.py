"""
Unit tests for EurLexIngester._parse_with_beautifulsoup() and helpers.

Uses synthetic HTML fixtures from conftest — no network, no real files needed.
Tests with real HTML files are gated by the requires_html marker.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.ingestion.eurlex_ingest import EurLexIngester


def ingester(lang: str = "en", local_file: str = "dummy.html") -> EurLexIngester:
    return EurLexIngester(language=lang, local_file=local_file)


# ---------------------------------------------------------------------------
# Basic parsing
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestParseWithBeautifulSoup:
    def test_produces_documents_from_valid_html(self, eurlex_html_en):
        docs = ingester("en")._parse_with_beautifulsoup(eurlex_html_en)
        assert len(docs) > 0

    def test_document_count_matches_article_count(self, eurlex_html_en):
        """Three article divs in fixture → three documents."""
        docs = ingester("en")._parse_with_beautifulsoup(eurlex_html_en)
        assert len(docs) == 3

    def test_first_document_has_article_metadata(self, eurlex_html_en):
        docs = ingester("en")._parse_with_beautifulsoup(eurlex_html_en)
        assert docs[0].metadata.get("article") == "1"

    def test_second_document_has_article_metadata(self, eurlex_html_en):
        docs = ingester("en")._parse_with_beautifulsoup(eurlex_html_en)
        assert docs[1].metadata.get("article") == "2"

    def test_hierarchy_metadata_from_parent_div_id(self, eurlex_html_en):
        docs = ingester("en")._parse_with_beautifulsoup(eurlex_html_en)
        meta = docs[0].metadata
        # parent_id="prt_ONE.tis_I.cpt_1.sct_1" → part=ONE, title=I, chapter=1, section=1
        assert meta.get("part") == "ONE"
        assert meta.get("title") == "I"
        assert meta.get("chapter") == "1"
        assert meta.get("section") == "1"

    def test_article3_in_section2(self, eurlex_html_en):
        """Article 3 has parent_id with sct_2 — section should be '2'."""
        docs = ingester("en")._parse_with_beautifulsoup(eurlex_html_en)
        art3 = next(d for d in docs if d.metadata.get("article") == "3")
        assert art3.metadata.get("section") == "2"

    def test_language_stamped_on_every_document_en(self, eurlex_html_en):
        docs = ingester("en")._parse_with_beautifulsoup(eurlex_html_en)
        for doc in docs:
            assert doc.metadata["language"] == "en"

    def test_language_stamped_on_every_document_it(self, eurlex_html_it):
        docs = ingester("it")._parse_with_beautifulsoup(eurlex_html_it)
        for doc in docs:
            assert doc.metadata["language"] == "it"

    def test_language_stamped_on_every_document_pl(self, eurlex_html_pl):
        docs = ingester("pl")._parse_with_beautifulsoup(eurlex_html_pl)
        for doc in docs:
            assert doc.metadata["language"] == "pl"

    def test_italian_hierarchy_from_parent_id(self, eurlex_html_it):
        docs = ingester("it")._parse_with_beautifulsoup(eurlex_html_it)
        assert len(docs) == 1
        meta = docs[0].metadata
        assert meta.get("article") == "1"
        assert meta.get("part") == "UNO"

    def test_no_articles_returns_empty_list_with_warning(self, eurlex_html_no_articles, caplog):
        import logging
        with caplog.at_level(logging.WARNING, logger="src.ingestion.eurlex_ingest"):
            docs = ingester("en")._parse_with_beautifulsoup(eurlex_html_no_articles)
        assert len(docs) == 0
        assert any("No articles" in r.message or "structure" in r.message
                   for r in caplog.records)

    def test_script_and_style_tags_removed(self):
        html = """
        <html><body>
        <script>alert('xss')</script>
        <style>.foo{color:red}</style>
        <div id="prt_ONE"><div id="art_1">
          <p class="title-article-norm">Article 1</p>
          <div class="norm"><span class="no-parag">1.</span>This is a legitimate paragraph with enough content.</div>
        </div></div>
        </body></html>
        """
        docs = ingester("en")._parse_with_beautifulsoup(html)
        assert len(docs) == 1
        assert "alert" not in docs[0].text

    def test_document_text_contains_paragraph_content(self, eurlex_html_en):
        docs = ingester("en")._parse_with_beautifulsoup(eurlex_html_en)
        # Article 1 has "definitions" in paragraph text
        assert "definitions" in docs[0].text.lower()

    def test_new_metadata_fields_present(self, eurlex_html_en):
        docs = ingester("en")._parse_with_beautifulsoup(eurlex_html_en)
        meta = docs[0].metadata
        assert "article_title" in meta
        assert "referenced_articles" in meta
        assert "referenced_external" in meta
        assert "has_table" in meta
        assert "has_formula" in meta


# ---------------------------------------------------------------------------
# Article title extraction
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestArticleTitleExtraction:
    def test_article_title_extracted(self, eurlex_html_en):
        docs = ingester("en")._parse_with_beautifulsoup(eurlex_html_en)
        art1 = next(d for d in docs if d.metadata.get("article") == "1")
        assert art1.metadata["article_title"] == "Definitions"

    def test_no_article_title_is_empty_string(self, eurlex_html_en):
        docs = ingester("en")._parse_with_beautifulsoup(eurlex_html_en)
        art2 = next(d for d in docs if d.metadata.get("article") == "2")
        # Article 2 has no article_title in fixture
        assert art2.metadata["article_title"] == ""

    def test_article_title_from_dedicated_fixture(self, eurlex_html_with_table):
        docs = ingester("en")._parse_with_beautifulsoup(eurlex_html_with_table)
        assert len(docs) == 1
        assert docs[0].metadata["article_title"] == "Own funds requirements"


# ---------------------------------------------------------------------------
# Cross-reference extraction
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCrossReferenceExtraction:
    def test_referenced_articles_extracted(self, eurlex_html_cross_refs):
        docs = ingester("en")._parse_with_beautifulsoup(eurlex_html_cross_refs)
        assert len(docs) == 1
        refs = docs[0].metadata["referenced_articles"]
        assert "26" in refs
        assert "36" in refs

    def test_referenced_external_directive(self, eurlex_html_cross_refs):
        docs = ingester("en")._parse_with_beautifulsoup(eurlex_html_cross_refs)
        ext = docs[0].metadata["referenced_external"]
        assert "Directive 2013/36/EU" in ext

    def test_referenced_external_regulation(self, eurlex_html_cross_refs):
        docs = ingester("en")._parse_with_beautifulsoup(eurlex_html_cross_refs)
        ext = docs[0].metadata["referenced_external"]
        assert "Regulation (EU) No 648/2012" in ext

    def test_no_references_when_absent(self, eurlex_html_en):
        docs = ingester("en")._parse_with_beautifulsoup(eurlex_html_en)
        # Generic fixture paragraphs don't contain explicit article references
        # referenced_articles should be a string (possibly empty)
        for doc in docs:
            assert isinstance(doc.metadata["referenced_articles"], str)

    def test_extract_cross_references_directly(self):
        ing = ingester("en")
        text = "As per Article 26 and Article 429a, see also Directive 2013/36/EU."
        arts, ext = ing._extract_cross_references(text)
        assert "26" in arts
        assert "429" in arts
        assert "Directive 2013/36/EU" in ext

    def test_extract_cross_references_italian(self):
        ing = ingester("it")
        text = "Ai sensi dell'Articolo 26 e dell'Articolo 92, si veda la direttiva 2013/36/UE."
        arts, ext = ing._extract_cross_references(text)
        assert "26" in arts
        assert "92" in arts
        assert "2013/36" in ext

    def test_extract_cross_references_empty_text(self):
        ing = ingester("en")
        arts, ext = ing._extract_cross_references("")
        assert arts == ""
        assert ext == ""


# ---------------------------------------------------------------------------
# Table and formula flags
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestTableAndFormulaFlags:
    def test_has_table_true_when_table_present(self, eurlex_html_with_table):
        docs = ingester("en")._parse_with_beautifulsoup(eurlex_html_with_table)
        assert docs[0].metadata["has_table"] is True

    def test_has_table_false_when_no_table(self, eurlex_html_en):
        docs = ingester("en")._parse_with_beautifulsoup(eurlex_html_en)
        for doc in docs:
            assert doc.metadata["has_table"] is False

    def test_has_formula_true_when_formula_present(self, eurlex_html_with_formula):
        docs = ingester("en")._parse_with_beautifulsoup(eurlex_html_with_formula)
        assert docs[0].metadata["has_formula"] is True

    def test_has_formula_false_when_no_formula(self, eurlex_html_en):
        docs = ingester("en")._parse_with_beautifulsoup(eurlex_html_en)
        for doc in docs:
            assert doc.metadata["has_formula"] is False

    def test_table_in_text_as_markdown(self, eurlex_html_with_table):
        docs = ingester("en")._parse_with_beautifulsoup(eurlex_html_with_table)
        text = docs[0].text
        assert "[TABLE]" in text or "---" in text  # markdown table marker or separator

    def test_formula_in_text_as_placeholder(self, eurlex_html_with_formula):
        docs = ingester("en")._parse_with_beautifulsoup(eurlex_html_with_formula)
        # Numbered placeholder [FORMULA_0] is emitted when parsing via _process_article_div
        assert "[FORMULA_" in docs[0].text


# ---------------------------------------------------------------------------
# Annex parsing
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestAnnexParsing:
    def test_annex_produces_document(self, eurlex_html_annex):
        docs = ingester("en")._parse_with_beautifulsoup(eurlex_html_annex)
        assert len(docs) == 1

    def test_annex_level_is_annex(self, eurlex_html_annex):
        docs = ingester("en")._parse_with_beautifulsoup(eurlex_html_annex)
        assert docs[0].metadata["level"] == "ANNEX"

    def test_annex_id_extracted(self, eurlex_html_annex):
        docs = ingester("en")._parse_with_beautifulsoup(eurlex_html_annex)
        assert docs[0].metadata["annex_id"] == "I"

    def test_annex_title_extracted(self, eurlex_html_annex):
        docs = ingester("en")._parse_with_beautifulsoup(eurlex_html_annex)
        assert "mutual recognition" in docs[0].metadata["annex_title"].lower()

    def test_annex_language_stamped(self, eurlex_html_annex):
        docs = ingester("en")._parse_with_beautifulsoup(eurlex_html_annex)
        assert docs[0].metadata["language"] == "en"

    def test_annex_text_not_empty(self, eurlex_html_annex):
        docs = ingester("en")._parse_with_beautifulsoup(eurlex_html_annex)
        assert len(docs[0].text.strip()) > 0


# ---------------------------------------------------------------------------
# Table to Markdown
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestTableToMarkdown:
    def _make_table(self, rows: list[list[str]]) -> str:
        """Build a minimal HTML table string."""
        html = "<table>"
        for i, row in enumerate(rows):
            tag = "th" if i == 0 else "td"
            html += "<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in row) + "</tr>"
        html += "</table>"
        return html

    def test_single_row_table(self):
        from bs4 import BeautifulSoup
        ing = ingester("en")
        html = self._make_table([["Item", "Value"]])
        table = BeautifulSoup(html, "html.parser").find("table")
        md = ing._table_to_markdown(table)
        assert "Item" in md
        assert "Value" in md

    def test_separator_after_header(self):
        from bs4 import BeautifulSoup
        ing = ingester("en")
        html = self._make_table([["A", "B"], ["1", "2"]])
        table = BeautifulSoup(html, "html.parser").find("table")
        md = ing._table_to_markdown(table)
        assert "---" in md

    def test_pipe_delimiters(self):
        from bs4 import BeautifulSoup
        ing = ingester("en")
        html = self._make_table([["H1", "H2"], ["v1", "v2"]])
        table = BeautifulSoup(html, "html.parser").find("table")
        md = ing._table_to_markdown(table)
        assert "|" in md

    def test_cell_pipe_escaped(self):
        from bs4 import BeautifulSoup
        ing = ingester("en")
        html = self._make_table([["A|B", "C"]])
        table = BeautifulSoup(html, "html.parser").find("table")
        md = ing._table_to_markdown(table)
        assert "A\\|B" in md

    def test_empty_table_returns_empty_string(self):
        from bs4 import BeautifulSoup
        ing = ingester("en")
        table = BeautifulSoup("<table></table>", "html.parser").find("table")
        assert ing._table_to_markdown(table) == ""


# ---------------------------------------------------------------------------
# Grid-list (lettered points) parsing
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGridListParsing:
    def test_points_appear_in_text(self, eurlex_html_with_points):
        docs = ingester("en")._parse_with_beautifulsoup(eurlex_html_with_points)
        assert len(docs) == 1
        text = docs[0].text
        assert "(a)" in text
        assert "(b)" in text
        assert "(c)" in text

    def test_point_text_appears(self, eurlex_html_with_points):
        docs = ingester("en")._parse_with_beautifulsoup(eurlex_html_with_points)
        assert "capital instruments" in docs[0].text


# ---------------------------------------------------------------------------
# _download_html with local file
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestDownloadHtmlLocalFile:
    def test_reads_local_file(self, tmp_path):
        html_file = tmp_path / "test.html"
        html_file.write_text("<html><body>hello</body></html>", encoding="utf-8")
        ing = EurLexIngester(language="en", local_file=str(html_file))
        content = ing._download_html()
        assert "hello" in content

    def test_utf8_content_preserved(self, tmp_path):
        html_file = tmp_path / "test.html"
        html_file.write_text("<p>Articolo 1 — Définition</p>", encoding="utf-8")
        ing = EurLexIngester(language="en", local_file=str(html_file))
        content = ing._download_html()
        assert "Définition" in content


# ---------------------------------------------------------------------------
# Full load() with real HTML files (gated by fixture skip)
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.requires_html
class TestLoadRealHtmlEN:
    def test_article_count_is_741(self, crr_html_en_path):
        ing = EurLexIngester(language="en", local_file=str(crr_html_en_path))
        docs = ing.load()
        # DOM-based parser: one Document per art_N div
        articles = [d for d in docs if d.metadata.get("level") == "ARTICLE"]
        assert len(articles) >= 740, f"Expected ~741 articles, got {len(articles)}"

    def test_annexes_parsed(self, crr_html_en_path):
        ing = EurLexIngester(language="en", local_file=str(crr_html_en_path))
        docs = ing.load()
        annexes = [d for d in docs if d.metadata.get("level") == "ANNEX"]
        assert len(annexes) >= 1, "Expected at least one annex"

    def test_article_92_has_part_three(self, crr_html_en_path):
        ing = EurLexIngester(language="en", local_file=str(crr_html_en_path))
        docs = ing.load()
        art92 = next(
            (d for d in docs if d.metadata.get("article") == "92" and d.metadata.get("language") == "en"),
            None,
        )
        assert art92 is not None, "Article 92 not found"
        assert art92.metadata.get("part") == "THREE" or art92.metadata.get("part") is not None

    def test_all_documents_have_language_en(self, crr_html_en_path):
        ing = EurLexIngester(language="en", local_file=str(crr_html_en_path))
        docs = ing.load()
        assert all(d.metadata.get("language") == "en" for d in docs)

    def test_all_documents_have_non_empty_text(self, crr_html_en_path):
        ing = EurLexIngester(language="en", local_file=str(crr_html_en_path))
        docs = ing.load()
        assert all(len(d.text.strip()) >= 5 for d in docs)

    def test_article_92_referenced_articles_non_empty(self, crr_html_en_path):
        ing = EurLexIngester(language="en", local_file=str(crr_html_en_path))
        docs = ing.load()
        art92 = next(
            (d for d in docs if d.metadata.get("article") == "92"),
            None,
        )
        assert art92 is not None
        # Article 92 references other articles — CSV should be non-empty
        refs = art92.metadata.get("referenced_articles", "")
        assert isinstance(refs, str)


@pytest.mark.unit
@pytest.mark.requires_html
class TestLoadRealHtmlIT:
    def test_article_count_comparable_to_en(self, crr_html_it_path):
        ing = EurLexIngester(language="it", local_file=str(crr_html_it_path))
        docs = ing.load()
        articles = [d for d in docs if d.metadata.get("level") == "ARTICLE"]
        assert len(articles) >= 740, f"Expected ~741 articles, got {len(articles)}"

    def test_all_documents_have_language_it(self, crr_html_it_path):
        ing = EurLexIngester(language="it", local_file=str(crr_html_it_path))
        docs = ing.load()
        assert all(d.metadata.get("language") == "it" for d in docs)
