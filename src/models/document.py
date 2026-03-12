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
    PARAGRAPH = "PARAGRAPH"
    POINT = "POINT"


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
            **self.metadata,
        }

    @property
    def citation(self) -> str:
        """Human-readable citation string, e.g. 'Part III, Title II, Chapter 1, Article 92'."""
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
