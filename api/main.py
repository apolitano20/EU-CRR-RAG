"""
FastAPI application for the EU CRR RAG system.

Endpoints:
  POST /api/query   – submit a compliance question, get answer + sources
  POST /api/ingest  – trigger the ingestion pipeline
"""
from __future__ import annotations

import logging
import os
import time
import threading
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.utils.logging_config import setup_logging

setup_logging(json_output=True)

from src.indexing.index_builder import HierarchicalIndexer
from src.indexing.vector_store import VectorStore
from src.ingestion.eurlex_ingest import EurLexIngester
from src.ingestion.language_config import get_config
from src.query.query_engine import QueryEngine, QueryResult

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Application state (module-level singletons)
# ------------------------------------------------------------------

_vector_store = VectorStore()
_indexer = HierarchicalIndexer(vector_store=_vector_store)
_query_engine = QueryEngine(vector_store=_vector_store, indexer=_indexer)
_ingestion_lock = threading.Lock()


_REQUIRED_ENV_VARS = ("OPENAI_API_KEY", "QDRANT_URL", "QDRANT_API_KEY")


@asynccontextmanager
async def lifespan(app: FastAPI):
    missing = [v for v in _REQUIRED_ENV_VARS if not os.getenv(v)]
    if missing:
        raise RuntimeError(f"Missing required env vars: {', '.join(missing)}. Check your .env file.")

    logger.info("Loading query engine from persisted index...")
    try:
        _query_engine.load()
        logger.info("Query engine ready.")
    except FileNotFoundError:
        logger.warning(
            "No persisted index found. Run ingestion first: "
            "python -m src.pipelines.ingest_pipeline"
        )
    yield


app = FastAPI(
    title="EU CRR RAG API",
    description="Regulatory compliance Q&A over EU Capital Requirements Regulation (CRR).",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every request with method, path, status code, and latency."""
    t0 = time.perf_counter()
    response = await call_next(request)
    latency_ms = round((time.perf_counter() - t0) * 1000)
    logger.info(
        "HTTP %s %s -> %d (%dms)",
        request.method,
        request.url.path,
        response.status_code,
        latency_ms,
    )
    return response

# ------------------------------------------------------------------
# Request / Response models
# ------------------------------------------------------------------


class QueryRequest(BaseModel):
    query: str
    preferred_language: Optional[str] = None
    max_cross_ref_expansions: Optional[int] = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    trace_id: str
    language: Optional[str] = None


class IngestRequest(BaseModel):
    url: Optional[str] = None
    language: str = "en"
    reset: bool = False


class IngestResponse(BaseModel):
    status: str
    message: str


class ArticleResponse(BaseModel):
    article: str
    article_title: str
    text: str
    part: Optional[str] = None
    title: Optional[str] = None
    chapter: Optional[str] = None
    section: Optional[str] = None
    referenced_articles: list[str]
    referenced_external: list[str] = []
    language: str


class CitingArticleItem(BaseModel):
    article: str
    article_title: str
    part: Optional[str] = None
    title: Optional[str] = None
    chapter: Optional[str] = None
    section: Optional[str] = None
    language: str


class CitingArticlesResponse(BaseModel):
    article: str
    citing_articles: list[CitingArticleItem]
    language: Optional[str] = None


# ------------------------------------------------------------------
# Language detection heuristic
# ------------------------------------------------------------------


def _detect_language(text: str) -> Optional[str]:
    """Guess query language from character set. Returns None for no filter (cross-language).

    Limitations (acceptable for EN/IT/PL MVP):
    - Polish is detected reliably via its unique diacritics (ą ć ę ł ń ś ź ż).
    - Italian detection uses àèéìíîòóùú, which are also present in French, Romanian,
      and Portuguese. Queries in those languages will be incorrectly routed to the
      Italian index. If additional Romance languages are added, replace this heuristic
      with a proper language-detection library (e.g. langdetect or lingua-py).
    - Plain English (no diacritics) always returns None → cross-language retrieval.
    """
    if set(text) & set("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ"):
        return "pl"
    if set(text) & set("àèéìíîòóùúÀÈÉÌÍÎÒÓÙÚ"):
        return "it"
    return None  # no filter → cross-language retrieval


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------


@app.post("/api/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    if not _query_engine.is_loaded():
        raise HTTPException(
            status_code=503,
            detail="Index not loaded. Run ingestion first.",
        )

    lang = request.preferred_language or _detect_language(request.query)
    result: QueryResult = _query_engine.query(
        request.query,
        language=lang,
        max_cross_ref_expansions=request.max_cross_ref_expansions,
    )
    return QueryResponse(
        answer=result.answer,
        sources=result.sources,
        trace_id=result.trace_id,
        language=lang,
    )


@app.get("/api/article/{article_id}", response_model=ArticleResponse)
def get_article(article_id: str, language: Optional[str] = None) -> ArticleResponse:
    if not _query_engine.is_loaded():
        raise HTTPException(
            status_code=503,
            detail="Index not loaded. Run ingestion first.",
        )
    result = _query_engine.get_article(article_id, language=language)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Article {article_id} not found.",
        )
    return ArticleResponse(**result)


@app.get("/api/article/{article_id}/citing", response_model=CitingArticlesResponse)
def get_citing_articles(article_id: str, language: Optional[str] = None) -> CitingArticlesResponse:
    """Return all articles that reference the given article number."""
    if not _query_engine.is_loaded():
        raise HTTPException(
            status_code=503,
            detail="Index not loaded. Run ingestion first.",
        )
    results = _query_engine.get_citing_articles(article_id, language=language)
    return CitingArticlesResponse(
        article=article_id,
        citing_articles=[CitingArticleItem(**r) for r in results],
        language=language,
    )


@app.post("/api/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest, background_tasks: BackgroundTasks) -> IngestResponse:
    if not _ingestion_lock.acquire(blocking=False):
        return IngestResponse(status="busy", message="Ingestion already in progress.")

    def _run() -> None:
        try:
            url = request.url or get_config(request.language).build_url()
            ingester = EurLexIngester(url=url, language=request.language)
            docs = ingester.load()
            indexer = HierarchicalIndexer(
                vector_store=_vector_store, reset_store=request.reset
            )
            indexer.build(docs)
            _query_engine.load()
        finally:
            _ingestion_lock.release()

    background_tasks.add_task(_run)
    return IngestResponse(status="started", message="Ingestion started in background.")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "index_loaded": _query_engine.is_loaded(),
        "vector_store_items": _vector_store.item_count,
    }
