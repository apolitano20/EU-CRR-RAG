"""One-off script: recompute referenced_annexes metadata in Qdrant.

Usage:
    python scripts/fix_annex_refs.py [--dry-run] [--language en]
"""

import json
import os
import re
import sys

from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

COLLECTION = "eu_crr"

# Language-specific annex keywords
ANNEX_KEYWORDS = {
    "en": r"Annex(?:es)?",
    "it": r"Allegat[oi]",
    "pl": r"Załącznik(?:ów|iem|i)?",
}
_ANNEX_NUM_PAT = re.compile(r"\b(I{1,3}|IV)\b")
_ROMAN_ORDER = ["I", "II", "III", "IV"]


def recompute_annex_refs(text: str, language: str) -> str:
    """Extract Annex Roman numeral references from article text."""
    annex_keyword = ANNEX_KEYWORDS.get(language, ANNEX_KEYWORDS["en"])
    annex_run_pat = re.compile(
        rf"\b{annex_keyword}\s+(?:IV|I{{1,3}})(?:\s*(?:,|and|or)\s+(?:IV|I{{1,3}}))*",
        re.I,
    )
    annex_nums: set[str] = set()
    for anx_m in annex_run_pat.finditer(text):
        for nm in _ANNEX_NUM_PAT.finditer(anx_m.group()):
            annex_nums.add(nm.group().upper())
    return ",".join(x for x in _ROMAN_ORDER if x in annex_nums)


def main():
    dry_run = "--dry-run" in sys.argv

    # Optional --language filter
    language_filter = None
    if "--language" in sys.argv:
        idx = sys.argv.index("--language")
        if idx + 1 < len(sys.argv):
            language_filter = sys.argv[idx + 1]

    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )

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
            language = payload.get("language", "en")

            if language_filter and language != language_filter:
                continue

            raw_content = payload.get("_node_content", "")
            try:
                text = json.loads(raw_content).get("text", "") if raw_content else ""
            except (json.JSONDecodeError, TypeError):
                text = ""

            old_refs = payload.get("referenced_annexes", "")
            new_refs = recompute_annex_refs(text, language)

            if old_refs != new_refs:
                updated += 1
                article = payload.get("article", "?")
                annex_id = payload.get("annex_id", "")
                label = f"Annex {annex_id}" if annex_id else f"Article {article}"
                print(f"  Updated Annex refs for {label} ({language}): {old_refs!r} -> {new_refs!r}")

                if not dry_run:
                    client.set_payload(
                        collection_name=COLLECTION,
                        payload={"referenced_annexes": new_refs},
                        points=[point.id],
                    )

        offset = next_offset
        if offset is None:
            break

    mode = "DRY RUN" if dry_run else "APPLIED"
    print(f"\n{mode}: {updated} points updated.")


if __name__ == "__main__":
    main()
