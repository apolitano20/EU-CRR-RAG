"""
Ingestion pipeline CLI.

Usage:
    python -m src.pipelines.ingest_pipeline --reset --language en --file crr_raw_en.html
    python -m src.pipelines.ingest_pipeline --language it --file crr_raw_ita.html
    python -m src.pipelines.ingest_pipeline --url <EUR-Lex URL> [--reset] [--language en]
    python -m src.pipelines.ingest_pipeline --reset --language en --file crr_raw_en.html --llama-parse

NOTE: --reset drops the ENTIRE Qdrant collection (all languages). Only use when starting
      completely fresh, not when adding a second language to an existing index.

NOTE: --llama-parse enables LlamaParse formula enrichment for articles that contain formula
      images. Requires LLAMA_CLOUD_API_KEY in .env. Only formula-containing articles (~50-100)
      are sent to LlamaParse; the rest use BeautifulSoup exclusively.
"""
from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

from src.utils.logging_config import setup_logging
setup_logging(json_output=False)  # human-readable for CLI use
logger = logging.getLogger(__name__)

from src.ingestion.eurlex_ingest import EurLexIngester
from src.ingestion.language_config import SUPPORTED_LANGUAGES, get_config
from src.indexing.index_builder import HierarchicalIndexer
from src.indexing.vector_store import VectorStore


def run(
    url: Optional[str],
    reset: bool,
    language: str = "en",
    local_file: Optional[str] = None,
    use_llama_parse: bool = False,
) -> None:
    logger.info("=== EU CRR Ingestion Pipeline (language=%s) ===", language)

    # Derive URL from language config if not explicitly provided
    if url is None:
        url = get_config(language).build_url()

    # 1. Download + parse
    ingester = EurLexIngester(url=url, language=language, local_file=local_file, use_llama_parse=use_llama_parse)
    documents = ingester.load()
    if not documents:
        logger.error("No documents returned from ingestion. Aborting.")
        sys.exit(1)

    # 2. Index
    vector_store = VectorStore()
    indexer = HierarchicalIndexer(vector_store=vector_store, reset_store=reset)
    indexer.build(documents)

    logger.info("=== Ingestion complete. ===")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest EU CRR from EUR-Lex into the RAG index.")
    parser.add_argument(
        "--url",
        default=None,
        help="EUR-Lex HTML URL. Auto-derived from --language if not provided.",
    )
    parser.add_argument(
        "--language",
        default="en",
        choices=SUPPORTED_LANGUAGES,
        help="Language of the CRR HTML to ingest (default: en).",
    )
    parser.add_argument(
        "--file",
        dest="local_file",
        default=None,
        help="Path to a local HTML file; skips network download entirely.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help=(
            "Drop and recreate the Qdrant collection before ingesting. "
            "WARNING: removes ALL languages from the index."
        ),
    )
    parser.add_argument(
        "--llama-parse",
        dest="use_llama_parse",
        action="store_true",
        help=(
            "Enable LlamaParse formula enrichment. Articles containing formula images are "
            "sent to LlamaParse to extract LaTeX. Requires LLAMA_CLOUD_API_KEY in .env."
        ),
    )
    args = parser.parse_args()
    run(
        url=args.url,
        reset=args.reset,
        language=args.language,
        local_file=args.local_file,
        use_llama_parse=args.use_llama_parse,
    )


if __name__ == "__main__":
    main()
