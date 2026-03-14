"""One-off script: recompute referenced_articles metadata in Qdrant,
excluding references to external regulations/directives.

Usage:
    python scripts/fix_cross_refs.py [--dry-run]
"""

import json
import os
import re
import sys

from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

COLLECTION = "eu_crr"

# Language-specific article keywords
ARTICLE_KEYWORDS = {"en": "Article", "it": "Articolo", "pl": "Artykuł"}

# Matches text after "Article N" that signals an external regulation
EXTERNAL_SUFFIX = re.compile(
    r"\s+(?:to\s+\d[\w]*\s+)?"
    r"(?:of|del(?:la|lo|l')?|dello)\s+"
    r"(?:Regulation|Directive|Decision|Delegated|Implementing"
    r"|Regolamento|Direttiva|Decisione"
    r"|Rozporządzenie|Dyrektywa|Decyzja)",
    re.I,
)


def recompute_refs(text: str, language: str) -> str:
    """Extract CRR-internal article refs, excluding external regulation refs."""
    keyword = re.escape(ARTICLE_KEYWORDS.get(language, "Article"))
    art_nums: set[str] = set()
    for m in re.finditer(rf"{keyword}s?\s+(\d+[a-z]?)", text, re.I):
        after = text[m.end():]
        if not EXTERNAL_SUFFIX.match(after):
            art_nums.add(m.group(1))
    return ",".join(sorted(art_nums, key=lambda x: int(re.sub(r"[^0-9]", "", x) or "0")))


def main():
    dry_run = "--dry-run" in sys.argv

    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )

    # Scroll through all points
    updated = 0
    offset = None
    batch_size = 100

    while True:
        results, next_offset = client.scroll(
            collection_name=COLLECTION,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        if not results:
            break

        for point in results:
            payload = point.payload or {}
            # Article text is inside _node_content JSON → "text" field
            raw_content = payload.get("_node_content", "")
            try:
                text = json.loads(raw_content).get("text", "") if raw_content else ""
            except (json.JSONDecodeError, TypeError):
                text = ""
            language = payload.get("language", "en")
            old_refs = payload.get("referenced_articles", "")

            new_refs = recompute_refs(text, language)

            if old_refs != new_refs:
                updated += 1
                article = payload.get("article", "?")
                print(f"  Article {article} ({language}): {old_refs!r} -> {new_refs!r}")

                if not dry_run:
                    client.set_payload(
                        collection_name=COLLECTION,
                        payload={"referenced_articles": new_refs},
                        points=[point.id],
                    )

        offset = next_offset
        if offset is None:
            break

    mode = "DRY RUN" if dry_run else "APPLIED"
    print(f"\n{mode}: {updated} points updated.")


if __name__ == "__main__":
    main()
