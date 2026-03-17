"""
Diagnose Qdrant item count to identify root cause of unexpected 2151 items.

Usage:
    python scripts/diagnose_qdrant.py
    python scripts/diagnose_qdrant.py --no-parser   # skip local HTML parse (faster)

Checks:
  1. Count by language in Qdrant
  2. Duplicate node_ids (confirms H1: non-deterministic Document IDs)
  3. Parser ground-truth count from local HTML files (if --no-parser not set)
  4. Annex breakdown by language (confirms H0: ^anx_ regex over-matching)
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict

from dotenv import load_dotenv

load_dotenv()

# Ensure src/ is importable when running as a script from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.indexing.vector_store import VectorStore


def _count_by_language(payloads: list[dict]) -> Counter:
    return Counter(p.get("language", "unknown") for p in payloads)


def _find_duplicate_node_ids(payloads: list[dict]) -> dict[str, int]:
    counts: Counter = Counter(p.get("node_id", "") for p in payloads)
    return {nid: cnt for nid, cnt in counts.items() if cnt > 1 and nid}


def _annex_breakdown(payloads: list[dict]) -> dict[str, list[str]]:
    """Return {language: [annex_id, ...]} for ANNEX-level items."""
    result: dict[str, list[str]] = defaultdict(list)
    for p in payloads:
        if p.get("level") == "ANNEX":
            lang = p.get("language", "unknown")
            result[lang].append(p.get("annex_id", p.get("node_id", "?")))
    return dict(result)


def _run_parser_count(lang: str, local_file: str | None) -> int | None:
    """Instantiate EurLexIngester and return document count without indexing."""
    try:
        from src.ingestion.eurlex_ingest import EurLexIngester

        ingester = EurLexIngester(language=lang, local_file=local_file)
        docs = ingester.load()
        return len(docs)
    except Exception as exc:
        print(f"  [WARN] Parser count for {lang} failed: {exc}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose Qdrant item count.")
    parser.add_argument(
        "--no-parser",
        action="store_true",
        help="Skip running the parser ground-truth check (faster).",
    )
    parser.add_argument(
        "--en-file",
        default=None,
        help="Path to local EN HTML file (used for parser ground-truth).",
    )
    parser.add_argument(
        "--it-file",
        default=None,
        help="Path to local IT HTML file (used for parser ground-truth).",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # 1. Connect and scroll all payloads
    # ------------------------------------------------------------------ #
    print("Connecting to Qdrant …")
    vs = VectorStore()
    vs.connect()
    total = vs.item_count
    print(f"Qdrant total items: {total}\n")

    print("Scrolling all payloads (no vectors) …")
    payloads = vs.scroll_payloads()
    print(f"Scrolled {len(payloads)} records.\n")

    # ------------------------------------------------------------------ #
    # 2. Count by language
    # ------------------------------------------------------------------ #
    lang_counts = _count_by_language(payloads)
    print("=== Items per language ===")
    for lang, cnt in sorted(lang_counts.items()):
        print(f"  {lang}: {cnt}")
    print(f"  TOTAL: {sum(lang_counts.values())}\n")

    # ------------------------------------------------------------------ #
    # 3. Duplicate node_ids → confirms H1
    # ------------------------------------------------------------------ #
    dupes = _find_duplicate_node_ids(payloads)
    print("=== Duplicate node_ids ===")
    if dupes:
        print(f"  FOUND {len(dupes)} duplicate node_id(s) - H1 (non-deterministic IDs) likely!")
        for nid, cnt in sorted(dupes.items())[:20]:
            print(f"    {nid}: {cnt}x")
        if len(dupes) > 20:
            print(f"    ... and {len(dupes) - 20} more")
    else:
        print("  No duplicates found - H1 ruled out.\n")

    # ------------------------------------------------------------------ #
    # 4. Annex breakdown → confirms H0
    # ------------------------------------------------------------------ #
    annex_breakdown = _annex_breakdown(payloads)
    print("=== ANNEX items per language ===")
    if not annex_breakdown:
        print("  No ANNEX-level items found.")
    for lang, annex_ids in sorted(annex_breakdown.items()):
        print(f"  {lang}: {len(annex_ids)} annex items")
        # Show distinct annex_id prefixes (first segment before ".")
        prefixes = sorted({aid.split(".")[0] for aid in annex_ids})
        print(f"    Top-level annex IDs: {prefixes}")
        if len(annex_ids) <= 30:
            for aid in sorted(annex_ids):
                print(f"      {aid}")
        else:
            for aid in sorted(annex_ids)[:15]:
                print(f"      {aid}")
            print(f"      ... and {len(annex_ids) - 15} more")
    print()

    # ------------------------------------------------------------------ #
    # 5. Non-annex count per language (article count in Qdrant)
    # ------------------------------------------------------------------ #
    print("=== Article (non-ANNEX) items per language ===")
    for lang in sorted(lang_counts.keys()):
        lang_payloads = [p for p in payloads if p.get("language") == lang]
        articles = [p for p in lang_payloads if p.get("level") != "ANNEX"]
        print(f"  {lang}: {len(articles)} article items")
    print()

    # ------------------------------------------------------------------ #
    # 6. Parser ground-truth count (optional)
    # ------------------------------------------------------------------ #
    if not args.no_parser:
        print("=== Parser ground-truth counts ===")
        lang_files: dict[str, str | None] = {
            "en": args.en_file,
            "it": args.it_file,
        }
        for lang, fpath in lang_files.items():
            if fpath:
                print(f"  Parsing {lang} from {fpath} …")
            else:
                print(f"  Parsing {lang} from EUR-Lex (network) …")
            count = _run_parser_count(lang, fpath)
            if count is not None:
                qdrant_count = lang_counts.get(lang, 0)
                diff = qdrant_count - count
                diff_str = f"+{diff}" if diff >= 0 else str(diff)
                print(f"  {lang}: parser={count}, qdrant={qdrant_count}, diff={diff_str}")
        print()

    # ------------------------------------------------------------------ #
    # Summary / hypothesis verdict
    # ------------------------------------------------------------------ #
    print("=== Hypothesis verdict ===")
    if dupes:
        print("  H1 CONFIRMED: Duplicate node_ids detected - non-deterministic Document IDs.")
        print("  Fix: set id_=node.node_id in _process_article_div() and _process_annex_div().")
        print("  Then: run --reset and re-ingest both languages.")
    else:
        annex_total = sum(len(ids) for ids in annex_breakdown.values())
        if annex_total > 20:
            print(f"  H0 LIKELY: {annex_total} ANNEX items found - ^anx_ regex picks up sub-annex divs.")
            print("  Consider restricting annex regex to top-level divs only.")
        else:
            print("  H2/H3 possible: check whether EN and IT counts differ significantly.")
            for lang, cnt in sorted(lang_counts.items()):
                print(f"    {lang}: {cnt}")


if __name__ == "__main__":
    main()
