"""
Shared test fixtures for the EU CRR RAG test suite.

Fixtures use the new EUR-Lex DOM structure:
  - Articles: <div id="art_N"> inside a parent <div id="prt_X.tis_Y.cpt_Z...">
  - Annexes: <div id="anx_I">
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# HTML builder helpers
# ---------------------------------------------------------------------------

def _make_article_html(articles: list[dict]) -> str:
    """
    Build a minimal EUR-Lex HTML page containing the given articles.

    Each article dict may have:
      num          : article number (required)
      parent_id    : parent div id encoding hierarchy (default: "prt_ONE.tis_I.cpt_1")
      article_title: stitle-article-norm text (optional)
      paragraphs   : list of paragraph texts (default: one generic paragraph)
      points       : list of (label, text) tuples for grid-list points (optional)
      has_table    : bool — if True, adds a <table class="borderOj"> (optional)
      has_formula  : bool — if True, adds an <img src="data:image/png;base64,..."> (optional)
    """
    blocks = []
    for art in articles:
        num = art["num"]
        parent_id = art.get("parent_id", "prt_ONE.tis_I.cpt_1")
        article_title = art.get("article_title", "")
        paragraphs = art.get("paragraphs", [f"This is the text of Article {num} with sufficient length."])
        points = art.get("points", [])
        has_table = art.get("has_table", False)
        has_formula = art.get("has_formula", False)

        eli_html = ""
        if article_title:
            eli_html = f'<div class="eli-title"><p class="stitle-article-norm">{article_title}</p></div>'

        para_html = ""
        for i, para in enumerate(paragraphs, 1):
            para_html += f'<div class="norm"><span class="no-parag">{i}.</span>{para}</div>\n'

        points_html = ""
        if points:
            rows = "".join(
                f'<div class="grid-row">'
                f'<div class="col-1">{label}</div>'
                f'<div class="col-2">{text}</div>'
                f'</div>'
                for label, text in points
            )
            points_html = f'<div class="grid-container grid-list">{rows}</div>'

        table_html = ""
        if has_table:
            table_html = (
                '<table class="borderOj">'
                "<tr><th>Item</th><th>Value</th></tr>"
                "<tr><td>CET1</td><td>4.5%</td></tr>"
                "</table>"
            )

        formula_html = ""
        if has_formula:
            formula_html = '<img src="data:image/png;base64,abc123" alt="formula"/>'

        blocks.append(textwrap.dedent(f"""\
            <div id="{parent_id}">
              <div id="art_{num}">
                <p class="title-article-norm">Article {num}</p>
                {eli_html}
                {para_html}
                {points_html}
                {table_html}
                {formula_html}
              </div>
            </div>
        """))

    return f"<!DOCTYPE html><html><body>{''.join(blocks)}</body></html>"


def _make_annex_html(annexes: list[dict]) -> str:
    """
    Build a minimal EUR-Lex HTML page containing the given annexes.

    Each annex dict may have:
      annex_id    : annex identifier e.g. "I", "II" (required)
      title       : annex title (optional)
      text        : body text (optional)
    """
    blocks = []
    for anx in annexes:
        annex_id = anx["annex_id"]
        title = anx.get("title", "")
        text = anx.get("text", f"Annex {annex_id} content with sufficient length for testing.")

        title_html = f'<p class="title-annex-2">{title}</p>' if title else ""
        blocks.append(textwrap.dedent(f"""\
            <div id="anx_{annex_id}">
              <p class="title-annex-1">ANNEX {annex_id}</p>
              {title_html}
              <p>{text}</p>
            </div>
        """))

    return f"<!DOCTYPE html><html><body>{''.join(blocks)}</body></html>"


# ---------------------------------------------------------------------------
# Synthetic EUR-Lex HTML fixtures (new DOM structure)
# ---------------------------------------------------------------------------

@pytest.fixture
def eurlex_html_en() -> str:
    """Minimal English CRR HTML with three articles across two sections."""
    return _make_article_html([
        {
            "num": "1",
            "parent_id": "prt_ONE.tis_I.cpt_1.sct_1",
            "article_title": "Definitions",
            "paragraphs": [
                "For the purposes of this Regulation, the following definitions apply.",
                "This paragraph has enough text to pass the minimum length requirement.",
            ],
        },
        {
            "num": "2",
            "parent_id": "prt_ONE.tis_I.cpt_1.sct_1",
            "paragraphs": [
                "Institutions shall apply the provisions set out in this Title.",
            ],
        },
        {
            "num": "3",
            "parent_id": "prt_ONE.tis_I.cpt_1.sct_2",
            "paragraphs": [
                "The competent authority may grant permission for alternative treatment.",
            ],
        },
    ])


@pytest.fixture
def eurlex_html_it() -> str:
    """Minimal Italian CRR HTML with one article."""
    return _make_article_html([
        {
            "num": "1",
            "parent_id": "prt_UNO.tis_I.cpt_1.sct_1",
            "paragraphs": [
                "Ai fini del presente regolamento si applicano le seguenti definizioni.",
                "Questa norma contiene abbastanza testo per il minimo di venti caratteri.",
            ],
        },
    ])


@pytest.fixture
def eurlex_html_pl() -> str:
    """Minimal Polish CRR HTML with one article."""
    return _make_article_html([
        {
            "num": "1",
            "parent_id": "prt_PIERWSZA.tis_I.cpt_1.sct_1",
            "paragraphs": [
                "Do celów niniejszego rozporządzenia stosuje się poniższe definicje.",
                "Ten akapit zawiera wystarczającą ilość tekstu by spełnić minimum znaków.",
            ],
        },
    ])


@pytest.fixture
def eurlex_html_no_articles() -> str:
    """HTML with no article divs at all."""
    return "<html><body><p>No articles here.</p></body></html>"


@pytest.fixture
def eurlex_html_with_table() -> str:
    """HTML with one article that contains a borderOj table."""
    return _make_article_html([
        {
            "num": "92",
            "parent_id": "prt_THREE.tis_I.cpt_1",
            "article_title": "Own funds requirements",
            "paragraphs": ["Institutions shall maintain own funds requirements."],
            "has_table": True,
        },
    ])


@pytest.fixture
def eurlex_html_with_formula() -> str:
    """HTML with one article that contains a formula image."""
    return _make_article_html([
        {
            "num": "429",
            "parent_id": "prt_THREE.tis_II.cpt_1",
            "paragraphs": ["The leverage ratio is calculated as follows."],
            "has_formula": True,
        },
    ])


@pytest.fixture
def eurlex_html_with_points() -> str:
    """HTML with one article that has grid-list lettered points."""
    return _make_article_html([
        {
            "num": "26",
            "parent_id": "prt_TWO.tis_I.cpt_1",
            "article_title": "CET1 items",
            "paragraphs": ["CET1 items of institutions consist of the following:"],
            "points": [
                ("(a)", "capital instruments, provided that conditions laid down in Article 28 are met;"),
                ("(b)", "share premium accounts related to those instruments;"),
                ("(c)", "retained earnings;"),
            ],
        },
    ])


@pytest.fixture
def eurlex_html_annex() -> str:
    """HTML with one annex section."""
    return _make_annex_html([
        {
            "annex_id": "I",
            "title": "List of activities subject to mutual recognition",
            "text": "The following activities shall be subject to mutual recognition under Article 34.",
        },
    ])


@pytest.fixture
def eurlex_html_cross_refs() -> str:
    """HTML with an article that references other articles and external legislation."""
    return _make_article_html([
        {
            "num": "92",
            "parent_id": "prt_THREE.tis_I.cpt_1",
            "paragraphs": [
                "As defined in Article 26 and Article 36, institutions shall comply with "
                "Articles 89, 90 and 91. See also Directive 2013/36/EU and "
                "Regulation (EU) No 648/2012 for related requirements.",
            ],
        },
    ])


@pytest.fixture
def eurlex_html_with_amendment_blocks() -> str:
    """HTML mimicking EUR-Lex amendment markers (<p class='modref'>) inside an article.

    Reproduces the Article 94 structure that triggered duplicate-paragraph output:
    - Paragraph 3 body contains a <p class="modref"> (▼M17) between point (b) and (c).
    - Two consecutive <p class="norm"> sub-paragraphs appear AFTER paragraph 3, directly
      inside the article div (not wrapped in a numbered <div class="norm">).
    - A second <p class="modref"> (▼M8) separates these sub-paragraphs from paragraph 4.

    Expected: the parser emits each text element exactly once, with no duplicates.
    """
    return textwrap.dedent("""\
        <!DOCTYPE html><html><body>
        <div id="prt_THREE.tis_I.cpt_1.sct_1">
          <div id="art_94">
            <p class="title-article-norm">Article 94</p>
            <div class="eli-title">
              <p class="stitle-article-norm">Derogation for small trading book business</p>
            </div>
            <div class="norm">
              <span class="no-parag">3.  </span>
              <div class="norm inline-element">
                <p class="norm inline-element">Institutions shall calculate the size in accordance with the following requirements:</p>
                <div class="grid-container grid-list">
                  <div class="list grid-list-column-1"><span>(a) </span></div>
                  <div class="grid-list-column-2"><p class="norm">all positions assigned to the trading book;</p></div>
                </div>
                <div class="grid-container grid-list">
                  <div class="list grid-list-column-1"><span>(b) </span></div>
                  <div class="grid-list-column-2"><p class="norm">all positions shall be valued at their market value;</p></div>
                </div>
                <p class="modref"><a title="M17: REPLACED">▼M17</a></p>
                <div class="grid-container grid-list">
                  <div class="list grid-list-column-1"><span>(c) </span></div>
                  <div class="grid-list-column-2"><p class="norm">the absolute value of the aggregated long position shall be summed.</p></div>
                </div>
              </div>
            </div>
            <p class="modref"><a title="M17: INSERTED">▼M17</a></p>
            <p class="norm">For the purposes of the first subparagraph, a long position is one where the market value increases.</p>
            <p class="norm">For the purposes of the first subparagraph, the value of the aggregated long position shall be equal to the sum.</p>
            <p class="modref"><a title="M8: REPLACED">▼M8</a></p>
            <div class="norm">
              <span class="no-parag">4.  </span>
              <div class="norm inline-element">Where both conditions are met, Article 102 shall not apply.</div>
            </div>
          </div>
        </div>
        </body></html>
    """)


# ---------------------------------------------------------------------------
# Fixtures for indexer chunking regression tests
# ---------------------------------------------------------------------------

# Rough token count reference: 1 token ≈ 4 chars.  The old LlamaIndex default
# chunk_size was 1024 tokens ≈ 4096 chars.  The long-article fixture below
# uses ~5500 chars so it would have been split by the old default SentenceSplitter
# but must NOT be split after the Settings.transformations = [] fix.
_LONG_PARAGRAPH = (
    "Institutions shall, for the purposes of calculating the own funds requirements "
    "for credit risk under this Part, apply the standardised approach set out in "
    "Chapter 2 or, where permitted by the competent authorities in accordance with "
    "Article 143, the IRB approach set out in Chapter 3. "
    "For trade exposures and default fund contributions to a CCP, institutions shall "
    "apply the treatment set out in Chapter 6, Section 9. "
    "Institutions shall apply the treatment set out in Chapter 5 for securitisation "
    "positions unless the competent authority has decided in accordance with Article 269 "
    "that an institution is not obliged to apply the treatment under that Chapter. "
    "Institutions shall report to their competent authority the different approaches "
    "they use for calculating own-funds requirements and provide evidence that the "
    "criteria for using such approaches are satisfied. "
)

# Repeat enough times to exceed 1024-token default chunk_size (~5500 chars total)
_LONG_ARTICLE_TEXT = " ".join([_LONG_PARAGRAPH] * 8)


@pytest.fixture
def eurlex_html_long_article() -> str:
    """HTML with one article whose text exceeds ~1024 tokens (the old SentenceSplitter default).

    Used to verify that the indexer does not split this into multiple nodes.
    With the old bug (Settings.transformations falling through to [SentenceSplitter]),
    this article would be split into 2+ chunks.  With the fix it must remain 1 document.
    """
    return _make_article_html([
        {
            "num": "92",
            "parent_id": "prt_THREE.tis_I.cpt_1",
            "article_title": "Own funds requirements",
            "paragraphs": [_LONG_ARTICLE_TEXT],
        },
    ])


@pytest.fixture
def eurlex_html_part_three_title_i() -> str:
    """Minimal 'Part Three, Title I, Chapter 1' HTML with five articles.

    Mirrors the real CRR structure (Articles 92–96) used to test that parser
    document count == indexer node count (no extra chunking introduced).
    Three articles have normal-length text; two have long text exceeding the
    old 1024-token default.
    """
    return _make_article_html([
        {
            "num": "92",
            "parent_id": "prt_THREE.tis_I.cpt_1.sct_1",
            "article_title": "Own funds requirements",
            "paragraphs": [_LONG_ARTICLE_TEXT],  # long — would be split by old default
        },
        {
            "num": "93",
            "parent_id": "prt_THREE.tis_I.cpt_1.sct_1",
            "article_title": "Transitional provisions for own funds requirements",
            "paragraphs": [
                "Institutions shall apply the transitional provisions set out in this Article.",
                "Competent authorities may require institutions to apply stricter measures.",
            ],
        },
        {
            "num": "94",
            "parent_id": "prt_THREE.tis_I.cpt_1.sct_1",
            "article_title": "Derogation for small trading book business",
            "paragraphs": [_LONG_ARTICLE_TEXT],  # long — would be split by old default
        },
        {
            "num": "95",
            "parent_id": "prt_THREE.tis_I.cpt_1.sct_1",
            "article_title": "Derogation for small trading book business for small institutions",
            "paragraphs": [
                "Institutions that meet the conditions in Article 94 may apply the "
                "derogation set out in that Article to their entire trading book.",
            ],
        },
        {
            "num": "96",
            "parent_id": "prt_THREE.tis_I.cpt_1.sct_1",
            "article_title": "Institutions with specific own funds requirements",
            "paragraphs": [
                "An institution that uses an internal model for position risk with the "
                "permission of the competent authority may use that model subject to "
                "the conditions set out in Article 363.",
            ],
        },
    ])


# ---------------------------------------------------------------------------
# Paths to real HTML files (may not exist on CI)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent


@pytest.fixture
def crr_html_en_path() -> Path:
    p = REPO_ROOT / "crr_raw_en.html"
    if not p.exists():
        pytest.skip("crr_raw_en.html not present — skipping file-based test")
    return p


@pytest.fixture
def crr_html_it_path() -> Path:
    p = REPO_ROOT / "crr_raw_ita.html"
    if not p.exists():
        pytest.skip("crr_raw_ita.html not present — skipping file-based test")
    return p
