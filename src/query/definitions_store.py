"""
DefinitionsStore: fast-path lookup for Article 4 CRR definitions.

Parses Article 4 text into a structured JSON index (by number and by term),
enabling O(1) definition lookups without touching the RAG pipeline.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFINITIONS_DIR = Path(__file__).parent.parent.parent / "definitions"
ARTICLE_4_NUM = "4"

# Two formats used across languages:
#   English:  (1) 'term' means ...    → anchored with opening paren
#   Italian:  1) «term» significa ... → no opening paren
_DEF_SPLIT_RE_PAREN = re.compile(r"(?<!\w)\((\d+[a-z]?)\)\s+")                    # EN: (N) or (Na)
_DEF_SPLIT_RE_BARE  = re.compile(r"(?<!\w)(\d+(?:\s*(?:bis|ter|quater)|[a-z])?)\)\s+")  # IT: N), Na), N bis), N ter)

# Extract quoted term from start of each definition body.
# Handles straight quotes, Unicode curly quotes (EN and IT), and guillemets.
_TERM_RE = re.compile(
    r"^['\u2018\u2019\u201c\u201d\u00ab\u00bb]([^'\u2018\u2019\u201c\u201d\u00ab\u00bb]+)['\u2018\u2019\u201c\u201d\u00ab\u00bb]"
)


class DefinitionsStore:
    """Structured index over Article 4 CRR definitions.

    At startup, loads definitions from a pre-built JSON file (fast path) or
    fetches them from Qdrant and persists them for subsequent runs.
    """

    def __init__(self, vector_store) -> None:
        self.vector_store = vector_store
        # lang → num_str → entry dict
        self._definitions: dict[str, dict[str, dict]] = {}
        # lang → lowercase_term → entry dict
        self._term_index: dict[str, dict[str, dict]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, language: str) -> None:
        """Load definitions for *language* from JSON cache or build from Qdrant."""
        json_path = self._json_path(language)
        if json_path.exists():
            logger.info("Loading Article 4 definitions from %s", json_path)
            with open(json_path, encoding="utf-8") as fh:
                data = json.load(fh)
            definitions = data.get("definitions", [])
        else:
            logger.info(
                "No definitions cache for language=%s — building from Qdrant", language
            )
            definitions = self._build_and_persist(language)

        self._definitions[language] = {d["number"]: d for d in definitions}
        self._term_index[language] = {
            d["term"].lower(): d for d in definitions if d.get("term")
        }
        logger.info(
            "DefinitionsStore: loaded %d definitions for language=%s",
            len(definitions),
            language,
        )

    def is_loaded(self, language: str) -> bool:
        return language in self._definitions

    def lookup_by_number(self, number: str | int, language: str = "en") -> Optional[dict]:
        """Return the definition entry for Article 4(number), or None."""
        if not self.is_loaded(language):
            return None
        return self._definitions[language].get(str(number))

    def lookup_by_term(self, term: str, language: str = "en") -> Optional[dict]:
        """Case-insensitive term lookup; returns None if not found."""
        if not self.is_loaded(language):
            return None
        return self._term_index[language].get(term.strip().lower())

    def summary(self, language: str = "en") -> str:
        """Return a short description of Article 4 with first-N terms listed."""
        if not self.is_loaded(language):
            return (
                "Article 4 of the CRR contains the definitions glossary. "
                "Ask about a specific term, e.g. 'What is the definition of institution?'"
            )
        defs = list(self._definitions[language].values())
        count = len(defs)
        sample_terms = [d["term"] for d in defs if d.get("term")][:10]
        terms_str = ", ".join(sample_terms)
        return (
            f"Article 4 is the definitions article of the CRR. "
            f"It contains {count} definitions including: {terms_str}. "
            f"Ask about a specific term, e.g. 'What is the definition of institution?'"
        )

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse(text: str) -> list[dict]:
        """Parse Article 4 text into a list of definition entries.

        Each entry is: {"number": str, "term": str, "text": str}

        Logic:
        1. Normalise whitespace.
        2. Split on ``(N)`` boundaries (digits only — skips sub-items like ``(a)``).
        3. For each (number, segment): extract leading quoted term via _TERM_RE.
        """
        # Step 1: normalise whitespace (including non-breaking spaces used in IT numbering)
        text = text.replace("\xa0", " ")
        text = re.sub(r"[ \t\r\n]+", " ", text).strip()

        # Step 2: detect format and split on definition boundaries.
        # English uses "(1) 'term'" while Italian uses "1) «term»".
        # Detect by checking whether "(1) " appears in the first 500 chars.
        split_re = _DEF_SPLIT_RE_PAREN if re.search(r"\(1\)\s+", text[:500]) else _DEF_SPLIT_RE_BARE
        parts = split_re.split(text)
        # parts = [pre_text, num_1, seg_1, num_2, seg_2, ...]
        definitions = []
        for number, segment in zip(parts[1::2], parts[2::2]):
            seg = segment.strip().rstrip(";. ")
            m = _TERM_RE.match(seg)
            if not m:
                # No quoted term → false split inside a definition body; skip.
                continue
            term = m.group(1).strip().lower()
            definitions.append({"number": number, "term": term, "text": seg})

        return definitions

    def _build_and_persist(self, language: str) -> list[dict]:
        """Fetch Article 4 from Qdrant, parse, and write JSON to disk."""
        payloads = self.vector_store.scroll_payloads(language=language)
        art4_payload = next(
            (
                p
                for p in payloads
                if str(p.get("article", "")) == ARTICLE_4_NUM
                and p.get("language", "") == language
            ),
            None,
        )
        if art4_payload is None:
            raise ValueError(
                f"Article 4 not found in Qdrant for language={language}. "
                "Run ingestion first."
            )

        text = art4_payload.get("text", "") or ""
        if not text:
            # LlamaIndex stores node text inside _node_content JSON blob
            import json as _json
            raw = art4_payload.get("_node_content", "")
            if raw:
                try:
                    text = _json.loads(raw).get("text", "") or ""
                except Exception:
                    pass
        definitions = self._parse(text)

        DEFINITIONS_DIR.mkdir(parents=True, exist_ok=True)
        out = {
            "language": language,
            "article": ARTICLE_4_NUM,
            "built_at": datetime.now(timezone.utc).isoformat(),
            "count": len(definitions),
            "definitions": definitions,
        }
        json_path = self._json_path(language)
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=2)
        logger.info(
            "DefinitionsStore: wrote %d definitions to %s", len(definitions), json_path
        )
        return definitions

    @staticmethod
    def _json_path(language: str) -> Path:
        return DEFINITIONS_DIR / f"definitions_{language}.json"
