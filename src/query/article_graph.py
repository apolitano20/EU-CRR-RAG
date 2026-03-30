"""
ArticleGraph: directed graph of CRR article cross-references.

Built at query-engine startup from existing Qdrant payload metadata — no
re-ingest required.  Edges are typed by scanning surrounding text for
legal register patterns (prerequisite, procedural, back_reference, general).

BFS expansion replaces the flat CSV + alphabetical-sort logic in
_expand_cross_references, enabling smarter traversal that prioritises
high-value relationships (prerequisites first, back-references last).
"""
from __future__ import annotations

import logging
import re
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Edge type constants (priority order: lower index = higher priority)
# ---------------------------------------------------------------------------
_REF_TYPE_PRIORITY: dict[str, int] = {
    "prerequisite":   0,
    "procedural":     1,
    "general":        2,
    "back_reference": 3,
    "definition":     4,  # Article 4 skipped at expansion time
}

# ---------------------------------------------------------------------------
# Regex patterns for edge-type classification.
# Scanned against the full article text at graph-build time.
# Each pattern is checked in order; first match wins. 
# ---------------------------------------------------------------------------
_PREREQ_PATTERNS = [
    re.compile(r"\bsubject to (?:Articles?|the requirements of Article)\s+(\d+\w*)", re.I),
    re.compile(r"\bshall (?:comply|meet the requirements?|satisfy the (?:conditions?|requirements?))\b.{0,80}Article\s+(\d+\w*)", re.I),
    re.compile(r"\bin compliance with (?:Article|paragraph)\s+(\d+\w*)", re.I),
    re.compile(r"\bprovided (?:that|for) in Article\s+(\d+\w*)", re.I),
]
_PROCEDURAL_PATTERNS = [
    re.compile(r"\bin accordance with (?:Article|paragraph|point)\s+(\d+\w*)", re.I),
    re.compile(r"\bcalculated (?:in accordance with|pursuant to|as set out in) Article\s+(\d+\w*)", re.I),
    re.compile(r"\bas laid down in Article\s+(\d+\w*)", re.I),
    re.compile(r"\bas (?:provided|specified|set out) in Article\s+(\d+\w*)", re.I),
    re.compile(r"\bpursuant to Article\s+(\d+\w*)", re.I),
]
_BACKREF_PATTERNS = [
    re.compile(r"\breferred to in Article\s+(\d+\w*)", re.I),
    re.compile(r"\bmentioned in Article\s+(\d+\w*)", re.I),
]
_DEFINITION_PATTERNS = [
    re.compile(r"\bas defined in (?:Article\s+4|point \(\d+\) of Article\s+4)", re.I),
]


def _classify_ref_type(article_text: str, target_art: str) -> str:
    """Return the ref_type for `target_art` as cited within `article_text`."""
    # Build a lookahead window around each occurrence of target_art in the text
    # to check which pattern it matches.
    # We search for Article <target> with surrounding context.
    art_pattern = re.compile(
        r"(?:Articles?\s+)" + re.escape(target_art) + r"\b",
        re.I,
    )
    for m in art_pattern.finditer(article_text):
        start = max(0, m.start() - 120)
        end = min(len(article_text), m.end() + 30)
        context = article_text[start:end]

        # Check definition pattern first (very specific)
        for pat in _DEFINITION_PATTERNS:
            if pat.search(context):
                return "definition"

        for pat in _PREREQ_PATTERNS:
            if pat.search(context):
                return "prerequisite"

        for pat in _PROCEDURAL_PATTERNS:
            if pat.search(context):
                return "procedural"

        for pat in _BACKREF_PATTERNS:
            if pat.search(context):
                return "back_reference"

    return "general"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ArticleEdge:
    """Directed edge from a source article to a target article."""
    target: str       # e.g. "36"
    ref_type: str     # one of _REF_TYPE_PRIORITY keys


@dataclass
class ArticleGraph:
    """Directed graph of CRR article cross-references.

    Built from Qdrant ARTICLE-chunk payloads at startup.  Thread-safe for
    concurrent reads after construction (build_from_qdrant() must be called
    from a single thread).
    """
    _forward:    dict[str, list[ArticleEdge]] = field(default_factory=lambda: defaultdict(list))
    _reverse:    dict[str, list[ArticleEdge]] = field(default_factory=lambda: defaultdict(list))
    _structural: dict[str, dict] = field(default_factory=dict)
    # Sub-article family clusters: parent → frozenset of all family members (incl. parent).
    # Built from sub_article_of payloads when present (requires re-ingest after adding field).
    # Falls back to regex-derived clusters at query time if metadata is absent.
    _sub_article_clusters: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    _build_lock: threading.Lock = field(default_factory=threading.Lock)
    _built:      bool = False

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def build_from_qdrant(
        self,
        client,
        collection_name: str,
        language: str = "en",
        batch_size: int = 200,
    ) -> None:
        """Scroll all ARTICLE-level nodes and build the graph.

        Args:
            client: QdrantClient instance (already connected).
            collection_name: Qdrant collection to read from.
            language: Only process nodes for this language (keeps graph focused).
            batch_size: Scroll page size.
        """
        from qdrant_client.http.models import FieldCondition, Filter, MatchValue

        with self._build_lock:
            if self._built:
                return

            logger.info("ArticleGraph: building from Qdrant collection '%s' (lang=%s)…", collection_name, language)
            t0 = __import__("time").perf_counter()

            must = [
                FieldCondition(key="chunk_type", match=MatchValue(value="ARTICLE")),
                FieldCondition(key="language",   match=MatchValue(value=language)),
            ]
            scroll_filter = Filter(must=must)

            offset = None
            total_nodes = 0
            total_edges = 0

            while True:
                try:
                    records, next_offset = client.scroll(
                        collection_name=collection_name,
                        scroll_filter=scroll_filter,
                        with_payload=True,
                        with_vectors=False,
                        limit=batch_size,
                        offset=offset,
                    )
                except Exception as exc:
                    logger.warning("ArticleGraph: Qdrant scroll failed: %s", exc)
                    break

                if not records:
                    break

                for record in records:
                    payload = record.payload or {}
                    src_art = str(payload.get("article", "")).strip()
                    if not src_art:
                        continue

                    # Structural location
                    self._structural[src_art] = {
                        "part":    payload.get("part", ""),
                        "title":   payload.get("title", ""),
                        "chapter": payload.get("chapter", ""),
                        "section": payload.get("section", ""),
                    }

                    # Sub-article family clustering (populated when sub_article_of is
                    # stored in the Qdrant payload — requires re-ingest after field added).
                    sub_art_of = str(payload.get("sub_article_of", "") or "").strip()
                    if sub_art_of:
                        # Both parent and sub-article are in the same cluster.
                        self._sub_article_clusters[sub_art_of].add(sub_art_of)
                        self._sub_article_clusters[sub_art_of].add(src_art)

                    # Edges from referenced_articles CSV
                    refs_csv = payload.get("referenced_articles", "") or ""
                    if not refs_csv:
                        total_nodes += 1
                        continue

                    # Get article text for edge-type classification
                    import json as _json
                    text = payload.get("text", "") or ""
                    if not text:
                        raw = payload.get("_node_content", "")
                        if raw:
                            try:
                                text = _json.loads(raw).get("text", "") or ""
                            except Exception:
                                pass

                    for ref in refs_csv.split(","):
                        ref = ref.strip()
                        if not ref or ref == src_art:
                            continue
                        ref_type = _classify_ref_type(text, ref)
                        edge = ArticleEdge(target=ref, ref_type=ref_type)
                        self._forward[src_art].append(edge)
                        # Reverse edge (the back-reference direction)
                        self._reverse[ref].append(ArticleEdge(target=src_art, ref_type="back_reference"))
                        total_edges += 1

                    total_nodes += 1

                if next_offset is None:
                    break
                offset = next_offset

            elapsed = round((__import__("time").perf_counter() - t0) * 1000)
            n_clusters = len(self._sub_article_clusters)
            logger.info(
                "ArticleGraph: built %d nodes, %d edges, %d sub-article families in %dms",
                total_nodes, total_edges, n_clusters, elapsed,
            )
            if n_clusters == 0:
                logger.info(
                    "ArticleGraph: no sub_article_of metadata found — re-ingest to enable "
                    "sub-article family expansion (sub_article_cluster() will return [])."
                )
            if elapsed > 3000:
                logger.warning("ArticleGraph: build took %dms (>3s threshold)", elapsed)

            self._built = True

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def bfs_expand(
        self,
        seeds: list[str],
        max_depth: int = 1,
        budget: int = 5,
        exclude: Optional[set[str]] = None,
        direction: str = "forward",
    ) -> list[str]:
        """BFS expansion from seed articles with budget constraint.

        Returns article numbers in BFS order (closer refs first), sorted within
        each BFS level by ref_type priority (prerequisite first, back_reference last).
        Article 4 (definitions glossary) is always skipped.

        Args:
            seeds: Starting article numbers.
            max_depth: How many hops to follow (1 = direct refs only).
            budget: Maximum total articles to return.
            exclude: Article numbers to never return (e.g. already-retrieved set).
            direction: "forward" = articles this article references;
                       "reverse" = articles that reference this article.
        """
        if not self._built:
            return []

        exclude = exclude or set()
        seen = set(seeds) | exclude | {"4"}  # skip Article 4
        result: list[str] = []
        queue: deque[tuple[str, int]] = deque((s, 0) for s in seeds)

        while queue and len(result) < budget:
            art, depth = queue.popleft()
            if depth >= max_depth:
                continue

            graph = self._forward if direction == "forward" else self._reverse
            edges = graph.get(art, [])

            # Sort candidates at this level by ref_type priority
            candidates = sorted(
                [e for e in edges if e.target not in seen],
                key=lambda e: _REF_TYPE_PRIORITY.get(e.ref_type, 99),
            )

            for edge in candidates:
                if len(result) >= budget:
                    break
                target = edge.target
                if target in seen:
                    continue
                seen.add(target)
                result.append(target)
                if depth + 1 < max_depth:
                    queue.append((target, depth + 1))

        return result

    def sub_article_cluster(self, article: str) -> list[str]:
        """Return all members of the same sub-article family, excluding `article` itself.

        For a parent article (e.g. "429"), returns its sub-articles (429a, 429b, …).
        For a sub-article (e.g. "429b"), returns the parent and all siblings.
        Returns [] when no family data is available (pre-re-ingest index).

        Uses metadata-derived clusters when available (populated after re-ingest);
        falls back to an empty list otherwise.  The family is useful for synthesis
        context enrichment: when a seed article is retrieved, also fetch its family
        members to ensure the full regulatory cluster is in context.
        """
        if not self._built:
            return []
        # article may be the parent key itself
        if article in self._sub_article_clusters:
            return [m for m in self._sub_article_clusters[article] if m != article]
        # article may be a sub-article — find its parent cluster
        for parent, members in self._sub_article_clusters.items():
            if article in members:
                return [m for m in members if m != article]
        return []

    def structural_siblings(self, article: str) -> list[str]:
        """Return articles in the same Section as `article` (or same Chapter if no Section).

        Useful for diluted_embedding failures where the target article is in the same
        structural cluster as the retrieved article but not directly cross-referenced.
        """
        if not self._built or article not in self._structural:
            return []
        loc = self._structural[article]
        section = loc.get("section", "")
        chapter = loc.get("chapter", "")
        title   = loc.get("title", "")
        part    = loc.get("part", "")

        siblings = []
        for art, art_loc in self._structural.items():
            if art == article:
                continue
            # Match on section first, then chapter, then title
            if section and art_loc.get("section") == section and art_loc.get("chapter") == chapter:
                siblings.append(art)
            elif not section and art_loc.get("chapter") == chapter and art_loc.get("title") == title and art_loc.get("part") == part:
                siblings.append(art)
        return siblings

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def node_count(self) -> int:
        return len(self._structural)

    @property
    def edge_count(self) -> int:
        return sum(len(v) for v in self._forward.values())

    @property
    def sub_article_family_count(self) -> int:
        """Number of distinct sub-article families known to the graph."""
        return len(self._sub_article_clusters)

    @property
    def is_built(self) -> bool:
        return self._built
