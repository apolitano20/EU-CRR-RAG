"""
EurLexIngester: downloads CRR HTML from EUR-Lex and converts it to LlamaIndex Documents.

Primary path:  BeautifulSoup → DOM-based legal-structure parser → Document per Article/Annex
Optional path: LlamaParse → structured markdown → Document list (opt-in via LLAMA_CLOUD_API_KEY)

Key insight: EUR-Lex HTML encodes the full legal hierarchy in parent div IDs.
For example, art_92's ancestor div has id="prt_III.tis_I.cpt_1.sct_1".
This allows precise, regex-free hierarchy extraction.
"""
from __future__ import annotations

import logging
import os
import re
import uuid
from typing import Optional

import requests
from llama_index.core import Document

from src.ingestion.language_config import LanguageConfig, get_config
from src.models.document import DocumentNode, NodeLevel

logger = logging.getLogger(__name__)

# Stable namespace for deterministic UUID generation from node_id strings.
# Qdrant only accepts unsigned integers or UUIDs as point IDs.
_NODE_ID_NAMESPACE = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")


def _build_hierarchy_prefix(
    *,
    part: Optional[str] = None,
    title: Optional[str] = None,
    chapter: Optional[str] = None,
    section: Optional[str] = None,
    article: Optional[str] = None,
    article_title: Optional[str] = None,
    annex_id: Optional[str] = None,
    annex_title: Optional[str] = None,
) -> str:
    """Build a hierarchy breadcrumb string for contextual embedding.

    Example output:
        Part III > Title I > Chapter 1 > Section 1 > Article 92 — Capital ratios
    """
    parts: list[str] = []
    if part:
        parts.append(f"Part {part}")
    if title:
        parts.append(f"Title {title}")
    if chapter:
        parts.append(f"Chapter {chapter}")
    if section:
        parts.append(f"Section {section}")
    if article:
        label = f"Article {article}"
        if article_title:
            label += f" — {article_title}"
        parts.append(label)
    if annex_id:
        label = f"Annex {annex_id}"
        if annex_title:
            label += f" — {annex_title}"
        parts.append(label)
    return " > ".join(parts)


def _node_id_to_uuid(node_id: str) -> str:
    """Convert a human-readable node_id (e.g. 'art_92_en') to a deterministic UUID.

    uuid5 is SHA-1 based and produces the same UUID for the same input every time,
    making Qdrant upserts idempotent across re-runs without --reset.
    """
    return str(uuid.uuid5(_NODE_ID_NAMESPACE, node_id))


# Language-specific patterns for external legislative references.
# Each value is a single regex string passed to re.findall().
_EXTERNAL_REF_PATTERNS: dict[str, str] = {
    "en": r"Directive\s+\d{4}/\d+/\w+|Regulation\s+\(EU\)\s+No\s+[\d/]+",
    "it": r"direttiva\s+\d{4}/\d+/\w+|regolamento\s+\(UE\)\s+n\.\s*[\d/]+",
    "pl": r"dyrektywa\s+\d{4}/\d+/\w+|rozporządzenie\s+\(UE\)\s+nr\s+[\d/]+",
}

# Language-specific annex keywords used by _extract_cross_references().
_ANNEX_KEYWORDS: dict[str, str] = {
    "en": r"Annex(?:es)?",
    "it": r"Allegat[oi]",
    "pl": r"Załącznik(?:ów|iem|i)?",
}
_ANNEX_NUM_PAT = re.compile(r"\b(I{1,3}|IV)\b")
_ROMAN_ORDER = ["I", "II", "III", "IV"]


def _extract_hierarchy(parent_id: str) -> dict:
    """Parse a EUR-Lex parent div ID into a hierarchy dict.

    Example: 'prt_III.tis_I.cpt_1.sct_1' → {'part': 'III', 'title': 'I', 'chapter': '1', 'section': '1'}
    """
    prefixes = {
        "prt_": "part",
        "tis_": "title",
        "cpt_": "chapter",
        "sct_": "section",
        "sbs_": "subsection",
    }
    result: dict[str, str] = {}
    for seg in parent_id.split("."):
        for pfx, key in prefixes.items():
            if seg.startswith(pfx):
                result[key] = seg[len(pfx):]
    return result


class EurLexIngester:
    """Downloads and parses the CRR consolidated text from EUR-Lex."""

    def __init__(
        self,
        url: Optional[str] = None,
        language: str = "en",
        local_file: Optional[str] = None,
        llama_cloud_api_key: Optional[str] = None,
        use_llama_parse: bool = False,
    ) -> None:
        self.language = language
        self.local_file = local_file
        self._lang_cfg: LanguageConfig = get_config(language)
        self.url = url or self._lang_cfg.build_url()
        self.use_llama_parse = use_llama_parse
        self.llama_cloud_api_key = llama_cloud_api_key or os.getenv("LLAMA_CLOUD_API_KEY")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> list[Document]:
        """Download (or read) CRR and return a list of LlamaIndex Documents."""
        html = self._download_html()

        if self.use_llama_parse and self.llama_cloud_api_key:
            try:
                return self._parse_with_llama_parse(html)
            except Exception as exc:
                logger.warning("LlamaParse failed (%s); falling back to BeautifulSoup.", exc)

        return self._parse_with_beautifulsoup(html)

    # ------------------------------------------------------------------
    # Download / read
    # ------------------------------------------------------------------

    def _download_html(self) -> str:
        if self.local_file:
            logger.info("Reading HTML from local file: %s", self.local_file)
            with open(self.local_file, encoding="utf-8", errors="replace") as f:
                return f.read()
        logger.info("Downloading CRR from %s", self.url)
        headers = {"User-Agent": "eu-crr-rag/1.0 (research tool)"}
        response = requests.get(self.url, headers=headers, timeout=60)
        response.raise_for_status()
        return response.text

    # ------------------------------------------------------------------
    # Primary parser: LlamaParse
    # ------------------------------------------------------------------

    def _parse_with_llama_parse(self, html: str) -> list[Document]:
        from llama_parse import LlamaParse  # lazy import – not always available

        parser = LlamaParse(
            api_key=self.llama_cloud_api_key,
            result_type="markdown",
            verbose=False,
        )
        import tempfile, pathlib

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
            tmp.write(html.encode("utf-8"))
            tmp_path = tmp.name

        try:
            docs = parser.load_data(tmp_path)
        finally:
            pathlib.Path(tmp_path).unlink(missing_ok=True)

        logger.info("LlamaParse returned %d document(s).", len(docs))
        return docs

    # ------------------------------------------------------------------
    # Fallback parser: BeautifulSoup (DOM-based, legal-structure-aware)
    # ------------------------------------------------------------------

    def _parse_with_beautifulsoup(self, html: str) -> list[Document]:
        logger.info("Parsing with BeautifulSoup (language=%s).", self.language)
        try:
            from bs4 import BeautifulSoup, FeatureNotFound, XMLParsedAsHTMLWarning
            import warnings
            warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
            try:
                soup = BeautifulSoup(html, "lxml")
            except FeatureNotFound:
                soup = BeautifulSoup(html, "html.parser")
        except Exception as exc:
            logger.error("BeautifulSoup parse failed: %s", exc)
            raise

        # Remove noise elements
        for tag in soup.select("script, style, nav, footer, .helpPanel, p.modref"):
            tag.decompose()

        documents: list[Document] = []

        # Process articles: each <div id="art_N"> is one Document
        for art_div in soup.find_all("div", id=re.compile(r"^art_[^.]+$")):
            doc = self._process_article_div(art_div)
            if doc and doc.text.strip():
                documents.append(doc)

        # Process annexes: each top-level <div id="anx_X"> is one Document.
        # Exclude sub-annex IDs like anx_IV.1 (contain a dot) to avoid duplicates.
        for anx_div in soup.find_all("div", id=re.compile(r"^anx_[^.]+$")):
            doc = self._process_annex_div(anx_div)
            if doc and doc.text.strip():
                documents.append(doc)

        if not documents:
            logger.warning(
                "No articles parsed — EUR-Lex HTML structure may have changed."
            )
        else:
            logger.info("BeautifulSoup produced %d document chunks.", len(documents))

        # Deduplicate by node_id (deterministic primary key: div ID + language)
        seen_node_ids: set[str] = set()
        deduped: list[Document] = []
        for doc in documents:
            nid = doc.metadata.get("node_id", "")
            if nid in seen_node_ids:
                logger.warning("Skipping duplicate document: node_id=%s", nid)
                continue
            seen_node_ids.add(nid)
            deduped.append(doc)
        if len(deduped) < len(documents):
            logger.info("Deduplication removed %d document(s).", len(documents) - len(deduped))
        return deduped

    # ------------------------------------------------------------------
    # Article processing
    # ------------------------------------------------------------------

    def _process_article_div(self, art_div) -> Optional[Document]:
        """Convert a single <div id="art_N"> into a LlamaIndex Document."""
        div_id = art_div.get("id", "")
        article_num = div_id[4:]  # strip "art_"

        # Hierarchy from parent div ID
        parent = art_div.parent
        parent_id = parent.get("id", "") if parent and hasattr(parent, "get") else ""
        hierarchy = _extract_hierarchy(parent_id)

        # Article title from eli-title > stitle-article-norm.
        # Fallback: any <p> inside eli-title that is not title-article-norm (which
        # contains "Article N" — the number heading, not the descriptive title).
        article_title = ""
        eli_title = art_div.find("div", class_="eli-title")
        if eli_title:
            stitle = eli_title.find("p", class_="stitle-article-norm")
            if stitle:
                article_title = stitle.get_text(" ", strip=True)
            else:
                # Fallback: first <p> that isn't the article-number heading
                for p in eli_title.find_all("p"):
                    if "title-article-norm" not in (p.get("class") or []):
                        candidate = p.get_text(" ", strip=True)
                        if candidate:
                            article_title = candidate
                            break

        formula_images: list[str] = []
        text = self._extract_structured_text(art_div, formula_images=formula_images)
        # Clean EUR-Lex amendment markers (▼, ◄, ►)
        text = re.sub(r"[▼◄►]\s*\w*", "", text).strip()

        has_table = bool(art_div.find("table"))
        has_formula = bool(formula_images)

        if has_formula and self.use_llama_parse and self.llama_cloud_api_key:
            text = self._enrich_formulas_with_llamaparse(art_div, text, len(formula_images))

        ref_articles, ref_external, ref_annexes = self._extract_cross_references(text)

        node = DocumentNode(
            node_id=f"art_{article_num}_{self.language}",
            level=NodeLevel.ARTICLE,
            text=text,
            part=hierarchy.get("part"),
            title=hierarchy.get("title"),
            chapter=hierarchy.get("chapter"),
            section=hierarchy.get("section"),
            article=article_num,
            article_title=article_title or None,
            referenced_articles=ref_articles,
            referenced_external=ref_external,
            referenced_annexes=ref_annexes,
            has_table=has_table,
            has_formula=has_formula,
        )
        meta = node.to_metadata()
        meta["language"] = self.language
        meta["display_text"] = text

        prefix = _build_hierarchy_prefix(
            part=hierarchy.get("part"),
            title=hierarchy.get("title"),
            chapter=hierarchy.get("chapter"),
            section=hierarchy.get("section"),
            article=article_num,
            article_title=article_title or None,
        )
        embedding_text = f"{prefix}\n\n{text}" if prefix else text

        return Document(text=embedding_text, metadata=meta, id_=_node_id_to_uuid(node.node_id))

    # ------------------------------------------------------------------
    # Annex processing
    # ------------------------------------------------------------------

    def _process_annex_div(self, anx_div) -> Optional[Document]:
        """Convert a single <div id="anx_..."> into a LlamaIndex Document."""
        div_id = anx_div.get("id", "")
        annex_id = div_id[4:] if div_id.startswith("anx_") else div_id

        annex_title = ""
        title_p = anx_div.find("p", class_="title-annex-2")
        if title_p:
            annex_title = title_p.get_text(" ", strip=True)

        text = self._extract_structured_text(anx_div)
        text = re.sub(r"[▼◄►]\s*\w*", "", text).strip()

        node = DocumentNode(
            node_id=f"anx_{annex_id}_{self.language}",
            level=NodeLevel.ANNEX,
            text=text,
            annex_id=annex_id,
            annex_title=annex_title or None,
        )
        meta = node.to_metadata()
        meta["language"] = self.language
        meta["display_text"] = text

        prefix = _build_hierarchy_prefix(
            annex_id=annex_id,
            annex_title=annex_title or None,
        )
        embedding_text = f"{prefix}\n\n{text}" if prefix else text

        return Document(text=embedding_text, metadata=meta, id_=_node_id_to_uuid(node.node_id))

    # ------------------------------------------------------------------
    # Structured text extraction
    # ------------------------------------------------------------------

    def _extract_structured_text(self, container, formula_images: Optional[list] = None) -> str:
        """Walk container children and produce structured plain text.

        If formula_images is provided (a mutable list), formula base64 data URIs are appended
        to it and placeholders are numbered: [FORMULA_0], [FORMULA_1], … This allows a
        post-processing step to resolve formulas via LlamaParse or vision models.
        If formula_images is None, placeholders are simply [FORMULA].
        """
        # Classes to skip entirely (article/annex title elements)
        SKIP_CLASSES = {"title-article-norm", "eli-title", "stitle-article-norm", "title-annex-1"}

        parts: list[str] = []
        _consumed: set[int] = set()  # id() of elements whose text was already extracted

        def _formula_placeholder(src: str) -> str:
            if formula_images is not None:
                idx = len(formula_images)
                formula_images.append(src)
                return f"[FORMULA_{idx}]"
            return "[FORMULA]"

        def walk(elem) -> None:
            if not hasattr(elem, "name") or elem.name is None:
                return
            if id(elem) in _consumed:
                return

            classes = set(elem.get("class") or [])
            tag = elem.name

            # Skip title/noise elements
            if classes & SKIP_CLASSES:
                return
            if tag in ("script", "style", "nav", "footer"):
                return

            # Numbered paragraph: <div class="norm"> with a <span class="no-parag">N.</span>
            # EUR-Lex also wraps paragraph content in <div class="norm inline-element"> —
            # those have NO no-parag span and must be recursed into, not flattened.
            if tag == "div" and "norm" in classes:
                span = elem.find("span", class_="no-parag")
                if span:
                    num = span.get_text(strip=True)
                    # Collect only direct inline/text/p children as the intro sentence.
                    # div children (grid-lists, inline-element wrappers) are walked
                    # separately so their internal structure is preserved.
                    intro_parts: list[str] = []
                    div_children = []
                    for child in elem.children:
                        if hasattr(child, "get") and "no-parag" in (child.get("class") or []):
                            continue
                        if not hasattr(child, "name") or child.name is None:
                            t = str(child).strip()
                            if t:
                                intro_parts.append(t)
                        elif child.name in ("p", "span", "a", "em", "strong"):
                            t = child.get_text(" ", strip=True)
                            if t:
                                intro_parts.append(t)
                        elif child.name == "div":
                            div_children.append(child)
                    if intro_parts:
                        parts.append(f"{num} {' '.join(intro_parts)}")
                    elif div_children:
                        # The paragraph text is inside a wrapper div (e.g.
                        # <div class="norm inline-element">).  Extract the
                        # leading text from the first wrapper and combine it
                        # with the paragraph number so we get "1. Text..."
                        # instead of "1." on its own line.
                        first = div_children[0]
                        first_classes = set(first.get("class") or [])
                        first_text = ""
                        if "inline-element" in first_classes:
                            child_elems = [
                                c for c in first.children
                                if hasattr(c, "name") and c.name is not None
                            ]
                            if not child_elems:
                                # Text-only wrapper: grab all text, mark
                                # entire div consumed (walker would add it
                                # again otherwise).
                                first_text = first.get_text(" ", strip=True)
                                if first_text:
                                    _consumed.add(id(first))
                            else:
                                # Has element children — grab first <p> text
                                first_p = first.find("p", recursive=False)
                                if first_p:
                                    first_text = first_p.get_text(" ", strip=True)
                                    if first_text:
                                        _consumed.add(id(first_p))
                        if first_text:
                            parts.append(f"{num} {first_text}")
                        else:
                            parts.append(num)
                    else:
                        parts.append(num)
                    for child in div_children:
                        walk(child)
                else:
                    # Wrapper div (e.g. "norm inline-element") — recurse into children
                    # so nested grid-lists and paragraphs are formatted correctly.
                    # Some wrapper divs contain only text nodes (no <p> wrapper),
                    # e.g. <div class="norm inline-element">Plain text here.</div>.
                    # Detect this and extract text directly to avoid silent data loss.
                    child_elements = [
                        c for c in elem.children
                        if hasattr(c, "name") and c.name is not None
                    ]
                    if child_elements:
                        for child in elem.children:
                            walk(child)
                    else:
                        text = elem.get_text(" ", strip=True)
                        if text and len(text) >= 5:
                            parts.append(text)
                return

            # Lettered/numbered points grid: <div class="grid-container grid-list">
            # EUR-Lex uses two column layouts:
            #   Layout A (older): grid-row > [label-div, text-div]
            #   Layout B (newer): grid-list-column-1 (label) + grid-list-column-2 (text)
            if tag == "div" and "grid-container" in classes and "grid-list" in classes:
                rows = elem.find_all("div", class_="grid-row", recursive=False)
                if rows:
                    # Layout A
                    for row in rows:
                        cols = row.find_all("div", recursive=False)
                        if len(cols) >= 2:
                            label = cols[0].get_text(strip=True)
                            # Collect direct inline/text children of the content column;
                            # descend into nested div children via walk() so sub-point
                            # labels, nested text, and formula placeholders are preserved.
                            col_parts: list[str] = []
                            div_children_in_col: list = []
                            for child in cols[1].children:
                                if not hasattr(child, "name") or child.name is None:
                                    t = str(child).strip()
                                    if t:
                                        col_parts.append(t)
                                elif child.name == "div":
                                    div_children_in_col.append(child)
                                elif child.name in ("p", "span", "em", "strong", "a"):
                                    t = child.get_text(" ", strip=True)
                                    if t:
                                        col_parts.append(t)
                            if col_parts:
                                entry = f"  {label} {' '.join(col_parts)}".rstrip()
                                if entry.strip():
                                    parts.append(entry)
                            elif label.strip():
                                parts.append(f"  {label}")
                            for child in div_children_in_col:
                                walk(child)
                else:
                    # Layout B: direct column children
                    col1 = elem.find("div", class_="grid-list-column-1", recursive=False)
                    col2 = elem.find("div", class_="grid-list-column-2", recursive=False)
                    if col1 and col2:
                        label = col1.get_text(strip=True)
                        # Direct text (p children) forms the item body
                        text_parts: list[str] = []
                        for child in col2.children:
                            if not hasattr(child, "name") or child.name is None:
                                t = str(child).strip()
                                if t:
                                    text_parts.append(t)
                            elif child.name == "p":
                                t = child.get_text(" ", strip=True)
                                if t:
                                    text_parts.append(t)
                        if text_parts:
                            parts.append(f"  {label} {' '.join(text_parts)}")
                        elif label:
                            parts.append(f"  {label}")
                        # Recurse into div children of col2 for nested grid-lists
                        for child in col2.children:
                            if hasattr(child, "name") and child.name == "div":
                                walk(child)
                    else:
                        text = elem.get_text(" ", strip=True)
                        if text:
                            parts.append(text)
                return

            # Table: <table class="borderOj"> (layout table) or plain <table>
            if tag == "table":
                if "borderOj" in classes:
                    parts.append("[TABLE]")
                md = self._table_to_markdown(elem)
                if md:
                    parts.append(md)
                return

            # Inline formula image
            if tag == "img":
                src = elem.get("src", "")
                if src.startswith("data:image"):
                    parts.append(_formula_placeholder(src))
                return

            # Generic paragraph — walk children so prefix/suffix text around
            # inline formula images is preserved in document order.
            if tag == "p":
                tokens: list[str] = []
                for child in elem.children:
                    if not hasattr(child, "name") or child.name is None:
                        t = str(child).strip()
                        if t:
                            tokens.append(t)
                    elif child.name == "img" and child.get("src", "").startswith("data:image"):
                        tokens.append(_formula_placeholder(child.get("src", "")))
                    elif child.name in ("strong", "em", "a", "span", "sub", "sup"):
                        t = child.get_text(" ", strip=True)
                        if t:
                            tokens.append(t)
                text = " ".join(tokens)
                if text and len(text) >= 5:
                    parts.append(text)
                return

            # Recurse into other elements (divs, spans, etc.)
            for child in elem.children:
                walk(child)

        for child in container.children:
            walk(child)

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Formula enrichment via LlamaParse
    # ------------------------------------------------------------------

    def _enrich_formulas_with_llamaparse(self, art_div, text: str, formula_count: int) -> str:
        """Replace [FORMULA_N] placeholders with LaTeX extracted by LlamaParse.

        Sends the raw HTML of the article div to LlamaParse (result_type="markdown").
        LlamaParse renders the page via its vision pipeline and emits LaTeX for formula images.
        We extract LaTeX tokens by order and substitute back into the BS4-structured text.

        If LlamaParse fails or returns fewer formulas than expected, the original
        [FORMULA_N] placeholder is preserved so the text is still valid.
        """
        import tempfile, pathlib

        from llama_parse import LlamaParse  # lazy import – requires llama_parse package

        article_html = f"<html><body>{str(art_div)}</body></html>"

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w", encoding="utf-8") as tmp:
            tmp.write(article_html)
            tmp_path = tmp.name

        try:
            parser = LlamaParse(
                api_key=self.llama_cloud_api_key,
                result_type="markdown",
                parsing_instruction=(
                    "This is a fragment of a legal regulation. "
                    "Extract all mathematical formulas as inline LaTeX (e.g. $K = ...$). "
                    "Preserve the surrounding text structure."
                ),
                verbose=False,
            )
            docs = parser.load_data(tmp_path)
        except Exception as exc:
            logger.warning("LlamaParse formula enrichment failed for article: %s", exc)
            return text
        finally:
            pathlib.Path(tmp_path).unlink(missing_ok=True)

        if not docs:
            return text

        combined_md = "\n".join(d.text for d in docs)
        latex_formulas = self._extract_latex_from_markdown(combined_md)

        for i in range(formula_count):
            placeholder = f"[FORMULA_{i}]"
            if i < len(latex_formulas):
                text = text.replace(placeholder, latex_formulas[i], 1)
            # else: leave placeholder as-is (safer than dropping content)

        return text

    @staticmethod
    def _extract_latex_from_markdown(markdown: str) -> list[str]:
        """Extract LaTeX formula strings from markdown in document order.

        Handles block ($$...$$, \\[...\\]) and inline ($...$) LaTeX delimiters.
        Returns a list of full formula strings including their delimiters.
        """
        # Block formulas first (greedy: double-dollar before single-dollar)
        block = re.findall(r"\$\$.+?\$\$|\\\[.+?\\\]", markdown, re.DOTALL)
        inline = re.findall(r"(?<!\$)\$(?!\$).+?(?<!\$)\$(?!\$)", markdown, re.DOTALL)

        # Reconstruct in document order by finding their positions
        tokens: list[tuple[int, str]] = []
        for pattern, flags in [
            (r"\$\$.+?\$\$", re.DOTALL),
            (r"\\\[.+?\\\]", re.DOTALL),
            (r"(?<!\$)\$(?!\$).+?(?<!\$)\$(?!\$)", re.DOTALL),
        ]:
            for m in re.finditer(pattern, markdown, flags):
                tokens.append((m.start(), m.group()))

        tokens.sort(key=lambda t: t[0])
        return [t[1] for t in tokens]

    # ------------------------------------------------------------------
    # Table → Markdown
    # ------------------------------------------------------------------

    def _table_to_markdown(self, table_tag) -> str:
        """Convert an HTML table to GitHub-Flavored Markdown."""
        rows = table_tag.find_all("tr")
        if not rows:
            return ""
        md_rows = []
        for i, row in enumerate(rows):
            cells = row.find_all(["td", "th"])
            cell_texts = [c.get_text(" ", strip=True).replace("|", "\\|") for c in cells]
            if not cell_texts:
                continue
            md_rows.append("| " + " | ".join(cell_texts) + " |")
            if i == 0:
                md_rows.append("| " + " | ".join(["---"] * len(cell_texts)) + " |")
        return "\n".join(md_rows)

    # ------------------------------------------------------------------
    # Cross-reference extraction
    # ------------------------------------------------------------------

    def _extract_cross_references(self, text: str) -> tuple[str, str, str]:
        """Extract internal article numbers, external directive/regulation references, and annex refs.

        Uses the language-specific article keyword from LanguageConfig so that
        "Articolo 26" (IT) and "Artykuł 26" (PL) are matched correctly.

        Returns (ref_articles_csv, ref_external_csv, ref_annexes_csv).
        """
        # Internal: language-aware article keyword, e.g. "Article 92" / "Articolo 92"
        # Exclude references to external regulations (e.g. "Article 10 of Regulation (EU)...")
        keyword = re.escape(self._lang_cfg.article_keyword)
        external_suffix = re.compile(
            r"\s+(?:to\s+\d[\w]*\s+)?"
            r"(?:of|del(?:la|lo|l')?|dello)\s+"
            r"(?:Regulation|Directive|Decision|Delegated|Implementing"
            r"|Regolamento|Direttiva|Decisione"
            r"|Rozporządzenie|Dyrektywa|Decyzja)",
            re.I,
        )
        art_nums: set[str] = set()
        # Match the full article-reference run first (handles ranges and comma/and lists),
        # then extract all individual numbers from the run.
        run_pat = re.compile(
            rf"\b{keyword}s?\s+\d+[a-z]*(?:\s*(?:,|and|or|to)\s+\d+[a-z]*)*",
            re.I,
        )
        num_pat = re.compile(r"\d+[a-z]*")
        for run_m in run_pat.finditer(text):
            run_text = run_m.group()
            after = text[run_m.end():]
            if external_suffix.match(after):
                continue
            nums_in_run = num_pat.findall(run_text)
            # Expand "N to M" ranges into individual numbers
            if re.search(r"\bto\b", run_text, re.I) and len(nums_in_run) == 2:
                lo = int(re.match(r"\d+", nums_in_run[0]).group())
                hi = int(re.match(r"\d+", nums_in_run[1]).group())
                for n in range(lo, hi + 1):
                    art_nums.add(str(n))
            else:
                for n in nums_in_run:
                    art_nums.add(n)

        # External: language-specific directive/regulation patterns
        ext_patterns = _EXTERNAL_REF_PATTERNS.get(self.language, _EXTERNAL_REF_PATTERNS["en"])
        ext_refs: set[str] = set(re.findall(ext_patterns, text))

        # Annex references: e.g. "Annex I", "Annexes I and III", "Allegato II"
        annex_keyword = _ANNEX_KEYWORDS.get(self.language, _ANNEX_KEYWORDS["en"])
        annex_run_pat = re.compile(
            rf"\b{annex_keyword}\s+(?:IV|I{{1,3}})(?:\s*(?:,|and|or)\s+(?:IV|I{{1,3}}))*",
            re.I,
        )
        annex_nums: set[str] = set()
        for anx_m in annex_run_pat.finditer(text):
            for nm in _ANNEX_NUM_PAT.finditer(anx_m.group()):
                annex_nums.add(nm.group().upper())

        return (
            ",".join(sorted(art_nums, key=lambda x: int(re.sub(r"[^0-9]", "", x) or "0"))),
            ",".join(sorted(ext_refs)),
            ",".join(x for x in _ROMAN_ORDER if x in annex_nums),
        )
