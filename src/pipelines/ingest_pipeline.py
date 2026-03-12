"""
Ingestion pipeline CLI.

Usage:
    python -m src.pipelines.ingest_pipeline
    python -m src.pipelines.ingest_pipeline --url <EUR-Lex URL> [--reset]
"""
from __future__ import annotations

import argparse
import logging
import sys

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

from src.ingestion.eurlex_ingest import DEFAULT_CRR_URL, EurLexIngester
from src.indexing.index_builder import HierarchicalIndexer
from src.indexing.vector_store import VectorStore


def run(url: str, reset: bool) -> None:
    logger.info("=== EU CRR Ingestion Pipeline ===")

    # 1. Download + parse
    ingester = EurLexIngester(url=url)
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
        default=DEFAULT_CRR_URL,
        help="EUR-Lex HTML URL for the consolidated CRR text.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop and recreate the vector store before ingesting.",
    )
    args = parser.parse_args()
    run(url=args.url, reset=args.reset)


if __name__ == "__main__":
    main()
