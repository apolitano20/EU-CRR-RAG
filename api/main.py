"""
FastAPI application for the EU CRR RAG system.

Endpoints:
  POST /api/query   – submit a compliance question, get answer + sources
  POST /api/ingest  – trigger the ingestion pipeline
"""
from __future__ import annotations

import asyncio
import json
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
from fastapi.responses import JSONResponse, StreamingResponse
from openai import AsyncOpenAI
from pydantic import BaseModel

from src.utils.logging_config import setup_logging

setup_logging(json_output=True)

from src.indexing.index_builder import HierarchicalIndexer
from src.indexing.vector_store import VectorStore
from src.ingestion.eurlex_ingest import EurLexIngester
from src.ingestion.language_config import get_config
from src.query.query_engine import (
    QueryEngine, QueryResult, LLM_MODEL,
    _LEGAL_QA_TEMPLATE, _LEGAL_QA_TEMPLATE_WITH_HISTORY,
    _format_history, _rewrite_query_with_history,
)

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


class HistoryTurn(BaseModel):
    question: str
    answer: str


class QueryRequest(BaseModel):
    query: str
    preferred_language: Optional[str] = None
    max_cross_ref_expansions: Optional[int] = None
    history: list[HistoryTurn] = []


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


class FeedbackRequest(BaseModel):
    query: str
    answer: str
    feedback: str
    sources: list[dict] = []
    viewed_article: Optional[dict] = None


class FeedbackResponse(BaseModel):
    status: str
    filename: str


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
    history_dicts = [t.model_dump() for t in request.history]
    result: QueryResult = _query_engine.query(
        request.query,
        language=lang,
        max_cross_ref_expansions=request.max_cross_ref_expansions,
        history=history_dicts,
    )
    return QueryResponse(
        answer=result.answer,
        sources=result.sources,
        trace_id=result.trace_id,
        language=lang,
    )


@app.post("/api/query/stream")
async def query_stream(request: QueryRequest) -> StreamingResponse:
    if not _query_engine.is_loaded():
        raise HTTPException(status_code=503, detail="Index not loaded. Run ingestion first.")

    lang = request.preferred_language or _detect_language(request.query)
    history_dicts = [t.model_dump() for t in request.history]

    # Rewrite follow-up query in a thread before retrieval
    effective_query = request.query
    if history_dicts:
        effective_query = await asyncio.to_thread(
            _rewrite_query_with_history,
            request.query, history_dicts,
            os.getenv("OPENAI_API_KEY"), LLM_MODEL,
        )
        if effective_query != request.query:
            logger.info("Stream query rewritten: '%s' → '%s'", request.query, effective_query)

    # Run synchronous retrieval in a thread to avoid blocking the event loop
    all_nodes, sources, trace_id, normalised_query, _engine = await asyncio.to_thread(
        _query_engine.retrieve,
        effective_query,
        lang,
        request.max_cross_ref_expansions,
    )

    async def generate():
        # Build article-labelled context string
        context_parts = []
        for node in all_nodes:
            meta = node.node.metadata
            art = meta.get("article", "")
            art_title = meta.get("article_title", "")
            header = f"Article {art}" + (f" — {art_title}" if art_title else "")
            context_parts.append(f"{header}\n\n{node.node.get_content()}")
        context_str = "\n\n---\n\n".join(context_parts)

        history_str = _format_history(history_dicts)
        if history_str:
            prompt = _LEGAL_QA_TEMPLATE_WITH_HISTORY.format(
                history_str=history_str, context_str=context_str, query_str=normalised_query
            )
        else:
            prompt = _LEGAL_QA_TEMPLATE.format(
                context_str=context_str,
                query_str=normalised_query,
            )

        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        stream = await client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            timeout=120.0,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield f"data: {json.dumps({'type': 'token', 'content': delta})}\n\n"

        yield f"data: {json.dumps({'type': 'sources', 'sources': sources, 'trace_id': trace_id, 'language': lang})}\n\n"
        yield 'data: {"type": "done"}\n\n'

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
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


@app.post("/api/feedback", response_model=FeedbackResponse)
def submit_feedback(request: FeedbackRequest) -> FeedbackResponse:
    import re
    from pathlib import Path

    cases_dir = Path("evals/cases")
    cases_dir.mkdir(parents=True, exist_ok=True)

    existing = list(cases_dir.glob("case_*.md"))
    numbers = [
        int(m.group(1))
        for f in existing
        if (m := re.match(r"case_(\d+)\.md", f.name))
    ]
    next_num = max(numbers, default=0) + 1
    filename = f"case_{next_num:03d}.md"

    lines: list[str] = [f"# Case {next_num:03d}", ""]
    lines += ["## Query", "", request.query, ""]
    lines += ["## Answer (AI)", "", request.answer, ""]

    if request.sources:
        lines += ["## Sources", ""]
        for s in request.sources:
            meta = s.get("metadata", {})
            art = meta.get("article", "unknown")
            title = meta.get("article_title", "")
            score = s.get("score", 0)
            entry = f"- **Article {art}**"
            if title:
                entry += f" — {title}"
            entry += f" (score: {score:.3f})"
            lines.append(entry)
        lines.append("")

    if request.viewed_article:
        art = request.viewed_article
        art_num = art.get("article", "")
        art_title = art.get("article_title", "")
        lines += ["## Article Viewed in Viewer", ""]
        lines.append(f"**Article {art_num} – {art_title}**")
        lines.append("")
        lines.append(art.get("text", ""))
        lines.append("")

    lines += ["## Feedback / Notes", "", request.feedback, ""]

    (cases_dir / filename).write_text("\n".join(lines), encoding="utf-8")
    logger.info("Feedback saved to evals/cases/%s", filename)
    return FeedbackResponse(status="ok", filename=filename)


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
