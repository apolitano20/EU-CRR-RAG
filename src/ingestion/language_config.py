"""
Language configuration for EUR-Lex ingestion.

Defines per-language heading keywords and URL templates for EN/IT/PL.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

SUPPORTED_LANGUAGES = ("en", "it", "pl")
DEFAULT_CELEX = "02013R0575-20260101"


@dataclass(frozen=True)
class LanguageConfig:
    language: str
    url_template: str  # contains {celex}
    heading_keywords: Dict[str, List[str]]  # level → regex prefixes (sub-variants first)
    article_keyword: str

    def build_url(self, celex: str = DEFAULT_CELEX) -> str:
        return self.url_template.format(celex=celex)


LANGUAGE_CONFIGS: Dict[str, LanguageConfig] = {
    "en": LanguageConfig(
        language="en",
        url_template="https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:{celex}",
        heading_keywords={
            "PART": ["PART"],
            "TITLE": ["TITLE"],
            "CHAPTER": ["CHAPTER"],
            "SECTION": ["SUB-SECTION", "SUBSECTION", "SECTION"],
        },
        article_keyword="Article",
    ),
    "it": LanguageConfig(
        language="it",
        url_template="https://eur-lex.europa.eu/legal-content/IT/TXT/HTML/?uri=CELEX:{celex}",
        heading_keywords={
            "PART": ["PARTE"],
            "TITLE": ["TITOLO"],
            "CHAPTER": ["CAPO"],
            "SECTION": ["SOTTOSEZIONE", "SEZIONE"],
        },
        article_keyword="Articolo",
    ),
    "pl": LanguageConfig(
        language="pl",
        url_template="https://eur-lex.europa.eu/legal-content/PL/TXT/HTML/?uri=CELEX:{celex}",
        heading_keywords={
            "PART": ["CZĘŚĆ"],
            "TITLE": ["TYTUŁ"],
            "CHAPTER": ["ROZDZIAŁ"],
            "SECTION": ["PODSEKCJA", "SEKCJA"],
        },
        article_keyword="Artykuł",
    ),
}


def get_config(language: str) -> LanguageConfig:
    """Return LanguageConfig for the given language code. Raises ValueError if unknown."""
    try:
        return LANGUAGE_CONFIGS[language]
    except KeyError:
        raise ValueError(
            f"Unsupported language: {language!r}. "
            f"Supported: {', '.join(SUPPORTED_LANGUAGES)}"
        )
