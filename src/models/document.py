from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class NodeLevel(str, Enum):
    PART = "PART"
    TITLE = "TITLE"
    CHAPTER = "CHAPTER"
    SECTION = "SECTION"
    ARTICLE = "ARTICLE"
    ANNEX = "ANNEX"


@dataclass
class DocumentNode:
    node_id: str
    level: NodeLevel
    text: str
    part: Optional[str] = None
    title: Optional[str] = None
    chapter: Optional[str] = None
    section: Optional[str] = None
    article: Optional[str] = None
    article_title: Optional[str] = None        # e.g. "Own funds requirements"
    annex_id: Optional[str] = None             # e.g. "I", "II" (annex nodes only)
    annex_title: Optional[str] = None          # annex subtitle
    referenced_articles: str = ""              # comma-separated "4,13,92"
    referenced_external: str = ""              # e.g. "Directive 2013/36/EU"
    referenced_annexes: str = ""              # comma-separated Roman numeral IDs, e.g. "I,III"
    has_table: bool = False
    has_formula: bool = False
    metadata: dict = field(default_factory=dict)

    def to_metadata(self) -> dict:
        """Flatten hierarchy fields into a metadata dict for LlamaIndex nodes."""
        return {
            "node_id": self.node_id,
            "level": self.level.value,
            "part": self.part or "",
            "title": self.title or "",
            "chapter": self.chapter or "",
            "section": self.section or "",
            "article": self.article or "",
            "article_title": self.article_title or "",
            "annex_id": self.annex_id or "",
            "annex_title": self.annex_title or "",
            "referenced_articles": self.referenced_articles,
            "referenced_external": self.referenced_external,
            "referenced_annexes": self.referenced_annexes,
            "has_table": self.has_table,
            "has_formula": self.has_formula,
            **self.metadata,
        }

    @property
    def citation(self) -> str:
        """Human-readable citation string, e.g. 'Part III, Title II, Chapter 1, Article 92'."""
        if self.annex_id:
            return f"Annex {self.annex_id}"
        parts = []
        if self.part:
            parts.append(f"Part {self.part}")
        if self.title:
            parts.append(f"Title {self.title}")
        if self.chapter:
            parts.append(f"Chapter {self.chapter}")
        if self.section:
            parts.append(f"Section {self.section}")
        if self.article:
            parts.append(f"Article {self.article}")
        return ", ".join(parts) if parts else self.node_id
