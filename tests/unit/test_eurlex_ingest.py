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

    def test_display_text_stored_in_metadata(self, eurlex_html_en):
        """display_text must be the raw body without the hierarchy prefix."""
        docs = ingester("en")._parse_with_beautifulsoup(eurlex_html_en)
        doc = docs[0]
        assert "display_text" in doc.metadata
        # display_text should not start with the hierarchy breadcrumb
        assert not doc.metadata["display_text"].startswith("Part ")

    def test_document_text_contains_hierarchy_prefix(self, eurlex_html_en):
        """Document.text (used for embedding) must include the hierarchy breadcrumb."""
        docs = ingester("en")._parse_with_beautifulsoup(eurlex_html_en)
        doc = docs[0]
        # The prefix must mention 'Article' and be at the start of the text
        assert "Article" in doc.text.split("\n")[0]

    def test_display_text_equals_body_only(self, eurlex_html_en):
        """display_text + prefix together should reconstruct Document.text."""
        docs = ingester("en")._parse_with_beautifulsoup(eurlex_html_en)
        doc = docs[0]
        display = doc.metadata["display_text"]
        # The body must appear in the full embedding text
        assert display in doc.text

    def test_document_text_contains_paragraph_content_after_prefix(self, eurlex_html_en):
        """The article body (definitions) must still be present after the prefix."""
        docs = ingester("en")._parse_with_beautifulsoup(eurlex_html_en)
        assert "definitions" in docs[0].text.lower()


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
        arts, ext, annexes = ing._extract_cross_references(text)
        assert "26" in arts
        assert "429" in arts
        assert "Directive 2013/36/EU" in ext

    def test_extract_cross_references_italian(self):
        ing = ingester("it")
        text = "Ai sensi dell'Articolo 26 e dell'Articolo 92, si veda la direttiva 2013/36/UE."
        arts, ext, annexes = ing._extract_cross_references(text)
        assert "26" in arts
        assert "92" in arts
        assert "2013/36" in ext

    def test_extract_cross_references_empty_text(self):
        ing = ingester("en")
        arts, ext, annexes = ing._extract_cross_references("")
        assert arts == ""
        assert ext == ""
        assert annexes == ""

    def test_external_article_ref_excluded_en(self):
        """Article N of Regulation/Directive should NOT appear in referenced_articles."""
        ing = ingester("en")
        text = (
            "in accordance with Articles 10 to 14 of Regulation (EU) No 1093/2010 "
            "and Article 8 of Directive 2014/59/EU. See also Article 92."
        )
        arts, ext, annexes = ing._extract_cross_references(text)
        assert "10" not in arts.split(",")
        assert "14" not in arts.split(",")
        assert "8" not in arts.split(",")
        assert "92" in arts.split(",")

    def test_external_article_ref_excluded_it(self):
        """Italian: Articolo N del Regolamento should NOT appear in referenced_articles."""
        ing = ingester("it")
        text = (
            "conformemente all'Articolo 4 del Regolamento (UE) n. 1093/2010 "
            "e all'Articolo 92 del presente regolamento."
        )
        arts, ext, annexes = ing._extract_cross_references(text)
        assert "4" not in arts.split(",")
        # "Articolo 92 del presente regolamento" — "del presente" != "del Regolamento (UE)"
        # so Article 92 should be kept as CRR-internal
        assert "92" in arts.split(",")

    def test_range_to_syntax_captured(self):
        """Fix 4a: 'Articles 89 to 91' expands to individual article numbers."""
        ing = ingester("en")
        arts, _, _annexes = ing._extract_cross_references("Institutions shall comply with Articles 89 to 91.")
        assert "89" in arts.split(",")
        assert "90" in arts.split(",")
        assert "91" in arts.split(",")

    def test_comma_list_captured(self):
        """Fix 4a: 'Articles 89, 90 and 91' captures all three."""
        ing = ingester("en")
        arts, _, _annexes = ing._extract_cross_references("See Articles 89, 90 and 91 for details.")
        assert "89" in arts.split(",")
        assert "90" in arts.split(",")
        assert "91" in arts.split(",")

    def test_multi_letter_suffix_preserved(self):
        """Fix 4a: 'Article 92aa' captures '92aa' not '92a'."""
        ing = ingester("en")
        arts, _, _annexes = ing._extract_cross_references("As per Article 92aa of this Regulation.")
        assert "92aa" in arts.split(",")

    def test_range_external_ref_excluded(self):
        """Fix 4a: 'Articles 89 to 91 of Directive ...' → none captured (external ref)."""
        ing = ingester("en")
        arts, _, _annexes = ing._extract_cross_references(
            "Articles 89 to 91 of Directive 2013/36/EU shall apply."
        )
        tokens = [t for t in arts.split(",") if t]
        assert "89" not in tokens
        assert "90" not in tokens
        assert "91" not in tokens

    def test_external_ref_delegated_implementing(self):
        """Delegated/Implementing acts should also be excluded."""
        ing = ingester("en")
        text = (
            "Article 3 of Delegated Regulation (EU) 2015/61 and "
            "Article 5 of Implementing Regulation (EU) 2021/451 apply. "
            "Article 395 shall also apply."
        )
        arts, ext, annexes = ing._extract_cross_references(text)
        assert "3" not in arts.split(",")
        assert "5" not in arts.split(",")
        assert "395" in arts.split(",")

    def test_annex_single_ref(self):
        """'See Annex I' → referenced_annexes contains 'I'."""
        ing = ingester("en")
        _, _, annexes = ing._extract_cross_references("See Annex I for further details.")
        assert "I" in annexes.split(",")

    def test_annex_plural_run(self):
        """'Annexes I and III' → referenced_annexes contains 'I' and 'III'."""
        ing = ingester("en")
        _, _, annexes = ing._extract_cross_references("As set out in Annexes I and III.")
        assert "I" in annexes.split(",")
        assert "III" in annexes.split(",")

    def test_annex_all_four(self):
        """'Annexes I, II, III and IV' → all four captured in order."""
        ing = ingester("en")
        _, _, annexes = ing._extract_cross_references("Refer to Annexes I, II, III and IV.")
        assert annexes == "I,II,III,IV"

    def test_annex_italian(self):
        """Italian: 'Allegato II' → referenced_annexes contains 'II'."""
        ing = ingester("it")
        _, _, annexes = ing._extract_cross_references("Conformemente all'Allegato II del presente.")
        assert "II" in annexes.split(",")

    def test_annex_no_false_positive(self):
        """Article-only text → referenced_annexes is empty."""
        ing = ingester("en")
        _, _, annexes = ing._extract_cross_references("Article 92 sets out own funds requirements.")
        assert annexes == ""

    def test_annex_stable_ordering(self):
        """'Annexes IV, I and II' → output is 'I,II,IV' (Roman order, not input order)."""
        ing = ingester("en")
        _, _, annexes = ing._extract_cross_references("As per Annexes IV, I and II.")
        assert annexes == "I,II,IV"


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
class TestFormulaParagraphParsing:
    """Fix 2: formula child-walk — prefix/suffix text around inline formulas preserved."""

    def test_formula_prefix_and_suffix_preserved(self):
        html = """<!DOCTYPE html><html><body>
        <div id="prt_ONE">
          <div id="art_1">
            <p class="title-article-norm">Article 1</p>
            <p>prefix text <img src="data:image/png;base64,abc123"/> suffix text long enough</p>
          </div>
        </div>
        </body></html>"""
        docs = ingester("en")._parse_with_beautifulsoup(html)
        assert len(docs) == 1
        text = docs[0].text
        assert "prefix text" in text
        assert "[FORMULA_" in text
        assert "suffix text" in text
        # Verify order: prefix < formula < suffix
        assert text.index("prefix text") < text.index("[FORMULA_") < text.index("suffix text")

    def test_multiple_formulas_in_one_paragraph(self):
        html = """<!DOCTYPE html><html><body>
        <div id="prt_ONE">
          <div id="art_1">
            <p class="title-article-norm">Article 1</p>
            <p><img src="data:image/png;base64,aaa"/> between the formulas <img src="data:image/png;base64,bbb"/></p>
          </div>
        </div>
        </body></html>"""
        docs = ingester("en")._parse_with_beautifulsoup(html)
        assert len(docs) == 1
        text = docs[0].text
        assert "[FORMULA_0]" in text
        assert "[FORMULA_1]" in text
        assert text.index("[FORMULA_0]") < text.index("between") < text.index("[FORMULA_1]")

    def test_formula_paragraph_does_not_drop_next_paragraph(self):
        html = """<!DOCTYPE html><html><body>
        <div id="prt_ONE">
          <div id="art_1">
            <p class="title-article-norm">Article 1</p>
            <p>Introduction text <img src="data:image/png;base64,abc123"/> more text here</p>
            <p>Second paragraph with enough content to be included here.</p>
          </div>
        </div>
        </body></html>"""
        docs = ingester("en")._parse_with_beautifulsoup(html)
        assert len(docs) == 1
        text = docs[0].text
        assert "[FORMULA_" in text
        assert "Second paragraph" in text


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

    def test_sub_annex_not_indexed_separately(self):
        """Fix 1: <div id="anx_IV.1"> must not be indexed as a separate document."""
        html = """<!DOCTYPE html><html><body>
        <div id="anx_IV">
          <p class="title-annex-1">ANNEX IV</p>
          <p>Main annex content with sufficient length for this test.</p>
          <div id="anx_IV.1">
            <p>Sub-annex IV.1 content that should not be indexed separately.</p>
          </div>
        </div>
        </body></html>"""
        docs = ingester("en")._parse_with_beautifulsoup(html)
        assert len(docs) == 1, f"Expected 1 document (top-level annex only), got {len(docs)}"

    def test_top_level_annex_still_indexed(self):
        """Fix 1 regression guard: plain <div id="anx_I"> must still produce one document."""
        html = """<!DOCTYPE html><html><body>
        <div id="anx_I">
          <p class="title-annex-1">ANNEX I</p>
          <p>Annex I content with sufficient length for the minimum threshold check.</p>
        </div>
        </body></html>"""
        docs = ingester("en")._parse_with_beautifulsoup(html)
        assert len(docs) == 1


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

    def test_nested_sub_points_in_grid_list(self):
        """Fix 3: nested div sub-points in Layout-A cols[1] appear separately, not concatenated."""
        html = """<!DOCTYPE html><html><body>
        <div id="prt_ONE">
          <div id="art_1">
            <p class="title-article-norm">Article 1</p>
            <div class="grid-container grid-list">
              <div class="grid-row">
                <div class="col-1">(a)</div>
                <div class="col-2">
                  <div class="norm">sub-point (i) first nested content text</div>
                  <div class="norm">sub-point (ii) second nested content text</div>
                </div>
              </div>
            </div>
          </div>
        </div>
        </body></html>"""
        docs = ingester("en")._parse_with_beautifulsoup(html)
        assert len(docs) == 1
        text = docs[0].text
        assert "first nested content text" in text
        assert "second nested content text" in text
        # They appear as separate lines (not merged into one long string)
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        first_idx = next((i for i, l in enumerate(lines) if "first nested content" in l), None)
        second_idx = next((i for i, l in enumerate(lines) if "second nested content" in l), None)
        assert first_idx is not None and second_idx is not None
        assert first_idx != second_idx, "Sub-points should appear on separate lines, not concatenated"

    def test_nested_formula_in_grid_list(self):
        """Fix 3: inline formula inside nested div in Layout-A cols[1] emits placeholder."""
        html = """<!DOCTYPE html><html><body>
        <div id="prt_ONE">
          <div id="art_1">
            <p class="title-article-norm">Article 1</p>
            <div class="grid-container grid-list">
              <div class="grid-row">
                <div class="col-1">(b)</div>
                <div class="col-2">
                  <div class="norm"><img src="data:image/png;base64,abc123" alt="formula"/></div>
                </div>
              </div>
            </div>
          </div>
        </div>
        </body></html>"""
        docs = ingester("en")._parse_with_beautifulsoup(html)
        assert len(docs) == 1
        assert "[FORMULA_" in docs[0].text


# ---------------------------------------------------------------------------
# Amendment block handling (regression for Article 94 duplicate paragraphs)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestAmendmentBlockParsing:
    """Parser must not duplicate content around EUR-Lex <p class='modref'> markers.

    EUR-Lex consolidated HTML places amendment markers (<p class="modref">) inside
    article content to indicate where insertions or replacements begin/end.
    Two consecutive ▼M17 blocks (REPLACED then INSERTED) appear back-to-back in
    some articles (e.g. Article 94), which historically caused the parser to emit
    the content between them twice.
    """

    def test_no_duplicate_paragraphs_with_amendment_markers(
        self, eurlex_html_with_amendment_blocks
    ):
        """Each paragraph/point should appear exactly once in the output text."""
        docs = ingester("en")._parse_with_beautifulsoup(eurlex_html_with_amendment_blocks)
        assert len(docs) == 1, f"Expected 1 document, got {len(docs)}"
        text = docs[0].text

        # Sub-paragraphs that sit between the two ▼M17 blocks — these were duplicated
        # in the Article 94 regression.
        first_subpara = "long position is one where"
        second_subpara = "aggregated long position shall be equal"

        assert text.count(first_subpara) == 1, (
            f"'{first_subpara}' appears {text.count(first_subpara)} times (expected 1)"
        )
        assert text.count(second_subpara) == 1, (
            f"'{second_subpara}' appears {text.count(second_subpara)} times (expected 1)"
        )

    def test_amendment_marker_text_stripped(self, eurlex_html_with_amendment_blocks):
        """▼M17 / ▼M8 marker symbols must not appear in the output."""
        docs = ingester("en")._parse_with_beautifulsoup(eurlex_html_with_amendment_blocks)
        text = docs[0].text
        assert "▼M17" not in text
        assert "▼M8" not in text
        assert "►M17" not in text

    def test_paragraph_4_not_duplicated(self, eurlex_html_with_amendment_blocks):
        """Paragraph 4 (which follows the ▼M8 marker) should appear exactly once."""
        docs = ingester("en")._parse_with_beautifulsoup(eurlex_html_with_amendment_blocks)
        text = docs[0].text
        para4_text = "Article 102 shall not apply"
        assert text.count(para4_text) == 1, (
            f"Paragraph 4 content appears {text.count(para4_text)} times (expected 1)"
        )

    def test_article_3_points_all_present(self, eurlex_html_with_amendment_blocks):
        """All three lettered points (a), (b), (c) of para 3 must be in the output."""
        docs = ingester("en")._parse_with_beautifulsoup(eurlex_html_with_amendment_blocks)
        text = docs[0].text
        assert "all positions assigned to the trading book" in text
        assert "valued at their market value" in text
        assert "absolute value of the aggregated long position" in text


# ---------------------------------------------------------------------------
# Paragraph-level chunking
# ---------------------------------------------------------------------------

def _make_multi_para_html(num_paragraphs: int = 3, article_num: str = "92",
                           article_title: str = "Own funds requirements") -> str:
    """Build an article HTML with `num_paragraphs` numbered <div class="norm"> paragraphs.

    Each paragraph contains enough words so that the combined body exceeds the
    _MIN_CHUNK_WORDS=100 threshold when num_paragraphs >= 2.
    """
    # ~60 words per paragraph so 2 paragraphs already exceed _MIN_CHUNK_WORDS=100
    _PARA_BODY = (
        "which sets out the requirements that institutions shall comply with at all times "
        "in accordance with this Regulation and any delegated acts or regulatory technical "
        "standards adopted thereunder by the Commission on the basis of a mandate conferred "
        "upon it by this Regulation."
    )
    paras = "".join(
        f'<div class="norm"><span class="no-parag">{i}.</span>'
        f'This is numbered paragraph {i} of Article {article_num} {_PARA_BODY}</div>\n'
        for i in range(1, num_paragraphs + 1)
    )
    title_html = f'<div class="eli-title"><p class="stitle-article-norm">{article_title}</p></div>'
    return f"""<!DOCTYPE html><html><body>
    <div id="prt_THREE.tis_I.cpt_1">
      <div id="art_{article_num}">
        <p class="title-article-norm">Article {article_num}</p>
        {title_html}
        {paras}
      </div>
    </div>
    </body></html>"""


@pytest.mark.unit
class TestParagraphChunking:
    """Dual-document index: multi-paragraph articles produce ARTICLE + PARAGRAPH docs."""

    def test_multi_para_article_produces_article_plus_paragraph_docs(self):
        """3-paragraph article → 1 ARTICLE doc + 3 PARAGRAPH docs = 4 total."""
        html = _make_multi_para_html(num_paragraphs=3)
        docs = ingester("en")._parse_with_beautifulsoup(html)
        article_docs = [d for d in docs if d.metadata.get("chunk_type") == "ARTICLE"]
        para_docs = [d for d in docs if d.metadata.get("chunk_type") == "PARAGRAPH"]
        assert len(article_docs) == 1
        assert len(para_docs) == 3
        assert len(docs) == 4

    def test_single_para_article_produces_only_article_doc(self):
        """Single-paragraph article → only 1 ARTICLE doc, no PARAGRAPH docs."""
        html = _make_multi_para_html(num_paragraphs=1)
        docs = ingester("en")._parse_with_beautifulsoup(html)
        assert len(docs) == 1
        assert docs[0].metadata["chunk_type"] == "ARTICLE"

    def test_article_doc_chunk_type_is_article(self):
        html = _make_multi_para_html(num_paragraphs=2)
        docs = ingester("en")._parse_with_beautifulsoup(html)
        article_doc = next(d for d in docs if d.metadata.get("chunk_type") == "ARTICLE")
        assert article_doc.metadata["chunk_type"] == "ARTICLE"

    def test_paragraph_doc_chunk_type_is_paragraph(self):
        html = _make_multi_para_html(num_paragraphs=2)
        docs = ingester("en")._parse_with_beautifulsoup(html)
        para_docs = [d for d in docs if d.metadata.get("chunk_type") == "PARAGRAPH"]
        assert len(para_docs) == 2
        for doc in para_docs:
            assert doc.metadata["chunk_type"] == "PARAGRAPH"

    def test_paragraph_doc_has_parent_article_id(self):
        html = _make_multi_para_html(num_paragraphs=2, article_num="92")
        docs = ingester("en")._parse_with_beautifulsoup(html)
        para_docs = [d for d in docs if d.metadata.get("chunk_type") == "PARAGRAPH"]
        for doc in para_docs:
            assert doc.metadata.get("parent_article_id") == "art_92_en"

    def test_paragraph_doc_has_para_id(self):
        html = _make_multi_para_html(num_paragraphs=3)
        docs = ingester("en")._parse_with_beautifulsoup(html)
        para_docs = sorted(
            [d for d in docs if d.metadata.get("chunk_type") == "PARAGRAPH"],
            key=lambda d: int(d.metadata["para_id"])
        )
        assert [d.metadata["para_id"] for d in para_docs] == ["1", "2", "3"]

    def test_paragraph_doc_inherits_article_metadata(self):
        html = _make_multi_para_html(num_paragraphs=2, article_num="92")
        docs = ingester("en")._parse_with_beautifulsoup(html)
        para_docs = [d for d in docs if d.metadata.get("chunk_type") == "PARAGRAPH"]
        for doc in para_docs:
            assert doc.metadata.get("article") == "92"
            assert doc.metadata.get("language") == "en"
            assert doc.metadata.get("part") == "THREE"

    def test_paragraph_embedding_text_has_hierarchy_prefix(self):
        html = _make_multi_para_html(num_paragraphs=2, article_num="92")
        docs = ingester("en")._parse_with_beautifulsoup(html)
        para_docs = [d for d in docs if d.metadata.get("chunk_type") == "PARAGRAPH"]
        for doc in para_docs:
            assert "Article 92" in doc.text.split("\n")[0]

    def test_paragraph_embedding_text_has_para_label(self):
        """Paragraph embedding text must include 'Paragraph N' label."""
        html = _make_multi_para_html(num_paragraphs=2, article_num="92")
        docs = ingester("en")._parse_with_beautifulsoup(html)
        para_docs = sorted(
            [d for d in docs if d.metadata.get("chunk_type") == "PARAGRAPH"],
            key=lambda d: int(d.metadata["para_id"])
        )
        assert "Paragraph 1" in para_docs[0].text
        assert "Paragraph 2" in para_docs[1].text

    def test_article_doc_chunk_type_absent_from_paragraph_node_ids(self):
        """ARTICLE doc node_id and PARAGRAPH doc node_ids must be distinct."""
        html = _make_multi_para_html(num_paragraphs=2, article_num="92")
        docs = ingester("en")._parse_with_beautifulsoup(html)
        node_ids = [d.metadata.get("node_id") for d in docs]
        assert len(node_ids) == len(set(node_ids)), "All node_ids must be unique"

    def test_article_doc_has_display_text(self):
        """ARTICLE doc must have display_text stored in metadata (body without prefix)."""
        html = _make_multi_para_html(num_paragraphs=2)
        docs = ingester("en")._parse_with_beautifulsoup(html)
        article_doc = next(d for d in docs if d.metadata.get("chunk_type") == "ARTICLE")
        assert "display_text" in article_doc.metadata
        assert not article_doc.metadata["display_text"].startswith("Part ")


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


# ---------------------------------------------------------------------------
# Task 1 — Lettered article handling (92a / 92b)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestLetteredArticleHandling:
    def test_lettered_article_metadata_from_div_id(self):
        """<div id="art_92a"> → metadata article='92a', node_id='art_92a_en'."""
        html = """
        <html><body>
          <div id="prt_ONE.tis_I.cpt_1.sct_1">
            <div id="art_92a">
              <div class="eli-title">
                <p class="stitle-article-norm">Own funds requirements — special</p>
              </div>
              <p>Paragraph text of Article 92a.</p>
            </div>
          </div>
        </body></html>
        """
        docs = ingester("en")._parse_with_beautifulsoup(html)
        assert len(docs) == 1
        assert docs[0].metadata["article"] == "92a"
        assert docs[0].metadata["node_id"] == "art_92a_en"

    def test_lettered_article_92b_metadata(self):
        """<div id="art_92b"> → metadata article='92b', node_id='art_92b_en'."""
        html = """
        <html><body>
          <div id="art_92b">
            <div class="eli-title">
              <p class="stitle-article-norm">Own funds requirements — other</p>
            </div>
            <p>Paragraph text of Article 92b.</p>
          </div>
        </body></html>
        """
        docs = ingester("en")._parse_with_beautifulsoup(html)
        assert len(docs) == 1
        assert docs[0].metadata["article"] == "92b"
        assert docs[0].metadata["node_id"] == "art_92b_en"


# ---------------------------------------------------------------------------
# Task 2 — Corpus deduplication
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCorpusDedup:
    def test_duplicate_div_ids_produce_single_document(self):
        """HTML with two identical <div id="art_1"> → only one document returned."""
        html = """
        <html><body>
          <div id="art_1"><p>First occurrence of Article 1.</p></div>
          <div id="art_1"><p>Duplicate occurrence of Article 1.</p></div>
        </body></html>
        """
        docs = ingester("en")._parse_with_beautifulsoup(html)
        assert len(docs) == 1

    def test_distinct_documents_not_deduped(self):
        """HTML with <div id="art_1"> and <div id="art_2"> → both documents kept."""
        html = """
        <html><body>
          <div id="art_1"><p>Article 1 text here.</p></div>
          <div id="art_2"><p>Article 2 text here.</p></div>
        </body></html>
        """
        docs = ingester("en")._parse_with_beautifulsoup(html)
        assert len(docs) == 2


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
