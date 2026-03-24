"""
TocStore: CRR Table of Contents index for LLM-guided routing.

Parses the authoritative EBA HTML ToC (CRR_ToC.html) into a structured
per-article index, enriches entries with key terms from Qdrant, and caches
to toc/toc_{lang}.json for fast subsequent loads.

The formatted ToC string (~12K tokens) is sent to GPT-4o-mini at query time
to predict which articles are relevant, providing a parallel retrieval path
that bridges vocabulary gaps between plain-language queries and CRR terminology.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

TOC_DIR = Path(__file__).parent.parent.parent / "toc"
TOC_HTML = Path(__file__).parent.parent.parent / "CRR_ToC.html"

# Patterns to identify hierarchy levels from plain-text container nodes
_RE_PART = re.compile(r"^PART\s+([A-Z]+(?:\s+[A-Z])?)\s*[:\-]", re.IGNORECASE)
_RE_TITLE = re.compile(r"^TITLE\s+([IVXivx\d]+)\s*[:\-]", re.IGNORECASE)
_RE_CHAPTER = re.compile(r"^CHAPTER\s+(\d+)\s*[:\-]", re.IGNORECASE)
_RE_SECTION = re.compile(r"^Section\s+(\d+)\s*[:\-]")
_RE_SUBSECTION = re.compile(r"^Sub-[Ss]ection\s+(\d+)\s*[:\-]")

# Article link text: "Article 92: Own funds requirements"
# Handles: "Article 5a:", "Article 519 e:" (space before letter suffix)
_RE_ARTICLE = re.compile(
    r"^Article\s+(\d+)\s*([a-z]?)\s*[:\-]\s*(.+)", re.IGNORECASE
)

# Annex link text: "ANNEX I: Classification of off-balance-sheet items"
_RE_ANNEX = re.compile(r"^ANNEX\s+([IVXivx]+)\s*[:\-]\s*(.+)", re.IGNORECASE)


class TocStore:
    """Structured index over the CRR Table of Contents.

    Parses the EBA HTML ToC as the authoritative source, enriches entries
    with key terms from Qdrant payloads, and caches the result per language.
    """

    def __init__(self, vector_store) -> None:
        self.vector_store = vector_store
        # lang → list of entry dicts
        self._entries: dict[str, list[dict]] = {}
        # lang → formatted prompt string (lazily built)
        self._prompt_cache: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, language: str) -> None:
        """Load ToC for *language* from JSON cache or build from HTML + Qdrant."""
        json_path = self._json_path(language)
        if json_path.exists():
            logger.info("Loading CRR ToC from %s", json_path)
            with open(json_path, encoding="utf-8") as fh:
                data = json.load(fh)
            entries = data.get("entries", [])
        else:
            logger.info(
                "No ToC cache for language=%s — building from HTML + Qdrant", language
            )
            entries = self._build_and_persist(language)

        self._entries[language] = entries
        self._prompt_cache.pop(language, None)  # invalidate cache
        logger.info(
            "TocStore: loaded %d entries for language=%s",
            len(entries),
            language,
        )

    def is_loaded(self, language: str) -> bool:
        return language in self._entries

    def format_for_prompt(self, language: str) -> str:
        """Return hierarchical ToC string for LLM routing prompt.

        Grouped by Part > Title > Chapter > Section. Cached after first build.
        """
        if language in self._prompt_cache:
            return self._prompt_cache[language]

        entries = self._entries.get(language, [])
        lines: list[str] = []
        cur: dict[str, Optional[str]] = {
            "part": None, "title": None, "chapter": None, "section": None
        }

        articles = [e for e in entries if not e.get("is_annex")]
        annexes = [e for e in entries if e.get("is_annex")]

        for e in articles:
            # Emit hierarchy headers when context changes
            if e["part"] != cur["part"]:
                lines.append(f"\nPART {e['part']}")
                cur.update({"part": e["part"], "title": None, "chapter": None, "section": None})

            if e.get("title") and e["title"] != cur["title"]:
                lines.append(f"  TITLE {e['title']}")
                cur.update({"title": e["title"], "chapter": None, "section": None})
            elif not e.get("title") and cur["title"] is not None:
                cur.update({"title": None, "chapter": None, "section": None})

            if e.get("chapter") and e["chapter"] != cur["chapter"]:
                lines.append(f"    CHAPTER {e['chapter']}")
                cur.update({"chapter": e["chapter"], "section": None})
            elif not e.get("chapter") and cur["chapter"] is not None:
                cur.update({"chapter": None, "section": None})

            if e.get("section") and e["section"] != cur["section"]:
                lines.append(f"      Section {e['section']}")
                cur["section"] = e["section"]
            elif not e.get("section") and cur["section"] is not None:
                cur["section"] = None

            # Indentation: 2 spaces per active hierarchy level above article
            depth = 1 + bool(e.get("title")) + bool(e.get("chapter")) + bool(e.get("section"))
            indent = "  " * depth

            article_line = f"{indent}Art. {e['article']} — {e['article_title']}"
            key_terms = (e.get("key_terms") or "").strip()
            if key_terms:
                words = key_terms.split()[:50]
                article_line += f" | Key terms: {' '.join(words)}"
            lines.append(article_line)

        if annexes:
            lines.append("\nANNEXES")
            for e in annexes:
                ann_id = e["article"].replace("ANNEX_", "")
                lines.append(f"  ANNEX {ann_id} — {e['article_title']}")

        result = "\n".join(lines).strip()
        self._prompt_cache[language] = result
        return result

    # ------------------------------------------------------------------
    # Build / persist
    # ------------------------------------------------------------------

    def _build_and_persist(self, language: str) -> list[dict]:
        """Parse HTML ToC, enrich with Qdrant key terms, write JSON cache."""
        # Step 1: parse EBA HTML (language-independent structure)
        entries = self._parse_html()

        # Step 2: enrich key_terms from Qdrant payloads for this language
        try:
            payloads = self.vector_store.scroll_payloads(language=language)
            key_terms_map = self._build_key_terms_map(payloads)
            for e in entries:
                article_num = e["article"]
                if not e.get("is_annex") and article_num in key_terms_map:
                    e["key_terms"] = key_terms_map[article_num]
        except Exception as exc:
            logger.warning(
                "TocStore: Qdrant enrichment failed for language=%s: %s — "
                "continuing without key terms",
                language,
                exc,
            )

        # Step 3: write JSON cache
        TOC_DIR.mkdir(parents=True, exist_ok=True)
        article_count = sum(1 for e in entries if not e.get("is_annex"))
        annex_count = sum(1 for e in entries if e.get("is_annex"))
        out = {
            "language": language,
            "source": TOC_HTML.name,
            "built_at": datetime.now(timezone.utc).isoformat(),
            "article_count": article_count,
            "annex_count": annex_count,
            "entries": entries,
        }
        json_path = self._json_path(language)
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=2)
        logger.info(
            "TocStore: wrote %d articles + %d annexes to %s",
            article_count,
            annex_count,
            json_path,
        )
        return entries

    # ------------------------------------------------------------------
    # HTML parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_html() -> list[dict]:
        """Parse CRR_ToC.html into a list of article/annex entry dicts.

        Walks the nested <ul>/<li> tree recursively, maintaining a hierarchy
        context (part/title/chapter/section) that is inherited by article leaves.
        """
        try:
            from bs4 import BeautifulSoup
        except ImportError as exc:
            raise ImportError(
                "beautifulsoup4 is required for TocStore HTML parsing. "
                "Install with: pip install beautifulsoup4"
            ) from exc

        if not TOC_HTML.exists():
            raise FileNotFoundError(
                f"CRR ToC HTML not found at {TOC_HTML}. "
                "Download from https://www.eba.europa.eu/regulation-and-policy/"
                "single-rulebook/interactive-single-rulebook/12674"
            )

        with open(TOC_HTML, encoding="utf-8") as fh:
            soup = BeautifulSoup(fh, "html.parser")

        # The ToC lives inside the isrb-tree block
        root = soup.find("div", id="block-isrb-tree")
        if root is None:
            # Fallback: try the view content wrapper
            root = soup.find("div", class_="view-id-isrb_tree")
        if root is None:
            raise ValueError(
                "Could not find ToC root element in CRR_ToC.html. "
                "Expected <div id='block-isrb-tree'>."
            )

        top_ul = root.find("ul")
        if top_ul is None:
            raise ValueError("No <ul> found inside ToC root element.")

        ctx = {"part": "", "title": "", "chapter": "", "section": ""}
        entries: list[dict] = []
        TocStore._walk_ul(top_ul, ctx, entries)
        return entries

    @staticmethod
    def _walk_ul(
        ul_elem,
        ctx: dict,
        entries: list[dict],
    ) -> None:
        """Recursively walk a <ul> element, updating ctx and appending entries."""
        for li in ul_elem.find_all("li", recursive=False):
            # Get the content region for this li
            region = li.find("div", class_=lambda c: c and "contextual-region" in c)
            if region is None:
                # Try the isrb-tree-item div as fallback
                region = li.find("div", class_=lambda c: c and "isrb-tree-item" in c)

            new_ctx = ctx.copy()

            if region is not None:
                a_tag = region.find("a")
                if a_tag is not None:
                    # Article or Annex leaf node
                    link_text = a_tag.get_text(separator=" ", strip=True)
                    # Remove extra whitespace from text
                    link_text = re.sub(r"\s+", " ", link_text).strip()

                    m_art = _RE_ARTICLE.match(link_text)
                    if m_art:
                        num = m_art.group(1) + m_art.group(2)  # e.g. "92" or "5a"
                        entries.append({
                            "article": num,
                            "article_title": m_art.group(3).strip(),
                            "part": new_ctx["part"],
                            "title": new_ctx["title"],
                            "chapter": new_ctx["chapter"],
                            "section": new_ctx["section"],
                            "is_annex": False,
                            "key_terms": "",
                        })
                    else:
                        m_ann = _RE_ANNEX.match(link_text)
                        if m_ann:
                            entries.append({
                                "article": f"ANNEX_{m_ann.group(1).upper()}",
                                "article_title": m_ann.group(2).strip(),
                                "part": "",
                                "title": "",
                                "chapter": "",
                                "section": "",
                                "is_annex": True,
                                "key_terms": "",
                            })
                else:
                    # Hierarchy container: plain text node
                    raw = region.get_text(separator=" ", strip=True)
                    text = re.sub(r"\s+", " ", raw).strip()

                    m = _RE_PART.match(text)
                    if m:
                        new_ctx = {"part": m.group(1).upper(), "title": "", "chapter": "", "section": ""}
                    else:
                        m = _RE_TITLE.match(text)
                        if m:
                            new_ctx["title"] = m.group(1).upper()
                            new_ctx["chapter"] = ""
                            new_ctx["section"] = ""
                        else:
                            m = _RE_CHAPTER.match(text)
                            if m:
                                new_ctx["chapter"] = m.group(1)
                                new_ctx["section"] = ""
                            else:
                                m = _RE_SECTION.match(text)
                                if m:
                                    new_ctx["section"] = m.group(1)
                                else:
                                    m = _RE_SUBSECTION.match(text)
                                    if m:
                                        # Sub-sections fold into section
                                        # Preserve parent section; sub-section disambiguates only
                                        # for prompt formatting (not tracked in entries)
                                        pass

            # Recurse into child <ul> elements with updated context
            child_ul = li.find("ul", recursive=False)
            if child_ul is not None:
                TocStore._walk_ul(child_ul, new_ctx, entries)

    # ------------------------------------------------------------------
    # Qdrant enrichment
    # ------------------------------------------------------------------

    @staticmethod
    def _build_key_terms_map(payloads: list[dict]) -> dict[str, str]:
        """Build article_number → first ~50 words of text from Qdrant payloads.

        Uses the same text extraction pattern as DefinitionsStore:
        try payload['text'] first, fall back to _node_content JSON blob.
        Deduplicates by article number (keeps the first non-empty text found).
        """
        key_terms: dict[str, str] = {}
        for payload in payloads:
            article = str(payload.get("article", "")).strip()
            if not article or article in key_terms:
                continue

            # Extract text: try direct field first, then _node_content blob
            text = payload.get("text", "") or ""
            if not text:
                raw = payload.get("_node_content", "")
                if raw:
                    try:
                        text = json.loads(raw).get("text", "") or ""
                    except Exception:
                        pass

            if not text:
                continue

            # Normalise whitespace, take first ~50 words
            text = re.sub(r"\s+", " ", text).strip()
            words = text.split()[:50]
            key_terms[article] = " ".join(words)

        return key_terms

    # ------------------------------------------------------------------
    # Path helper
    # ------------------------------------------------------------------

    @staticmethod
    def _json_path(language: str) -> Path:
        return TOC_DIR / f"toc_{language}.json"
