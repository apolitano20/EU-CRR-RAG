"""
EurLexIngester: downloads CRR HTML from EUR-Lex and converts it to LlamaIndex Documents.

Primary path:  BeautifulSoup → class-aware stateful DOM walker → Document list
Optional path: LlamaParse → structured markdown → Document list (opt-in via LLAMA_CLOUD_API_KEY)
"""
from __future__ import annotations

import logging
import os
import re
from typing import Optional

import requests
from llama_index.core import Document

logger = logging.getLogger(__name__)

# Default consolidated CRR URL (update CELEX date stamp as needed)
DEFAULT_CRR_URL = (
    "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/"
    "?uri=CELEX:02013R0575-20260101"
)

# EUR-Lex CSS class → legal level
_CLASS_TO_LEVEL = {
    "sti-doc": "PART",
    "ti-section-1": "PART",
    "ti-section-2": "TITLE",
    "ti-section-3": "CHAPTER",
    "ti-section-4": "SECTION",
    "sti-art": "ARTICLE",
    "ti-art": "ARTICLE",
}
_PARAGRAPH_CLASS = "normal"


class EurLexIngester:
    """Downloads and parses the CRR consolidated text from EUR-Lex."""

    def __init__(
        self,
        url: str = DEFAULT_CRR_URL,
        llama_cloud_api_key: Optional[str] = None,
        use_llama_parse: bool = False,
    ) -> None:
        self.url = url
        self.use_llama_parse = use_llama_parse
        self.llama_cloud_api_key = llama_cloud_api_key or os.getenv("LLAMA_CLOUD_API_KEY")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> list[Document]:
        """Download CRR and return a list of LlamaIndex Documents."""
        logger.info("Downloading CRR from %s", self.url)
        html = self._download_html()

        if self.use_llama_parse and self.llama_cloud_api_key:
            try:
                return self._parse_with_llama_parse(html)
            except Exception as exc:
                logger.warning("LlamaParse failed (%s); falling back to BeautifulSoup.", exc)

        return self._parse_with_beautifulsoup(html)

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def _download_html(self) -> str:
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
    # Fallback parser: BeautifulSoup (class-aware stateful DOM walker)
    # ------------------------------------------------------------------

    def _parse_with_beautifulsoup(self, html: str) -> list[Document]:
        logger.info("Parsing with BeautifulSoup.")
        try:
            from bs4 import BeautifulSoup, FeatureNotFound
            try:
                soup = BeautifulSoup(html, "lxml")
            except FeatureNotFound:
                soup = BeautifulSoup(html, "html.parser")
        except Exception as exc:
            logger.error("BeautifulSoup parse failed: %s", exc)
            raise

        # Remove noise elements
        for tag in soup.select("script, style, nav, footer, .helpPanel"):
            tag.decompose()

        documents: list[Document] = []
        # Hierarchy state
        meta: dict[str, str] = {}
        paragraphs: list[str] = []

        for element in soup.find_all(True):
            classes = element.get("class") or []
            # Resolve the first matching EUR-Lex class
            level = None
            matched_class = None
            for cls in classes:
                if cls in _CLASS_TO_LEVEL:
                    level = _CLASS_TO_LEVEL[cls]
                    matched_class = cls
                    break

            if level is not None:
                # Flush accumulated paragraphs before updating hierarchy
                if paragraphs:
                    documents.append(self._make_document(paragraphs, meta))
                    paragraphs = []

                text = element.get_text(" ", strip=True)
                if level == "ARTICLE":
                    article_meta = self._classify_article_heading(text)
                    meta.update(article_meta)
                else:
                    heading_meta = self._classify_heading(text, level)
                    # When a higher-level heading appears, clear lower-level fields
                    meta = self._reset_lower(meta, level)
                    meta.update(heading_meta)

            elif _PARAGRAPH_CLASS in classes:
                text = element.get_text(" ", strip=True)
                if len(text) >= 20:
                    paragraphs.append(text)

        # Flush any remaining paragraphs
        if paragraphs:
            documents.append(self._make_document(paragraphs, meta))

        if not documents:
            logger.warning(
                "No articles parsed — EUR-Lex HTML class names may have changed."
            )
        else:
            logger.info("BeautifulSoup produced %d document chunks.", len(documents))

        return documents

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_document(paragraphs: list[str], meta: dict[str, str]) -> Document:
        return Document(text="\n\n".join(paragraphs), metadata=dict(meta))

    @staticmethod
    def _classify_heading(text: str, level: str) -> dict[str, str]:
        meta: dict[str, str] = {"level": level}
        if m := re.match(r"PART\s+([\w]+)", text, re.I):
            meta["part"] = m.group(1)
        elif m := re.match(r"TITLE\s+([\w]+)", text, re.I):
            meta["title"] = m.group(1)
        elif m := re.match(r"CHAPTER\s+([\w]+)", text, re.I):
            meta["chapter"] = m.group(1)
        elif m := re.match(r"SECTION\s+([\w]+)", text, re.I):
            meta["section"] = m.group(1)
        return meta

    @staticmethod
    def _classify_article_heading(text: str) -> dict[str, str]:
        if m := re.match(r"Article\s+(\d+[a-z]?)", text, re.I):
            return {"article": m.group(1), "level": "ARTICLE"}
        return {"level": "ARTICLE"}

    @staticmethod
    def _reset_lower(meta: dict[str, str], level: str) -> dict[str, str]:
        """Clear metadata fields that are lower in hierarchy than the new heading level."""
        order = ["PART", "TITLE", "CHAPTER", "SECTION", "ARTICLE"]
        field_map = {
            "PART": "part",
            "TITLE": "title",
            "CHAPTER": "chapter",
            "SECTION": "section",
            "ARTICLE": "article",
        }
        if level not in order:
            return meta
        idx = order.index(level)
        new_meta = dict(meta)
        for l in order[idx:]:
            new_meta.pop(field_map[l], None)
        return new_meta
