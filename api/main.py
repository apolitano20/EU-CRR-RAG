"""
FastAPI application for the EU CRR RAG system.

Endpoints:
  POST /api/query        – submit a compliance question, get answer + sources
  POST /api/query/stream – streaming version via SSE
  POST /api/ingest       – trigger the ingestion pipeline
"""
from __future__ import annotations

import asyncio
import logging
import os
import platform
import re
import sys
import threading
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from dotenv import load_dotenv

load_dotenv()

# ── Windows / Python 3.13 WMI hang workaround ────────────────────────────────
# SQLAlchemy (a transitive dependency of llama_index) calls platform.machine()
# at import time.  On Windows, platform.machine() → platform.uname() →
# platform._wmi_query() which can hang indefinitely when the WMI COM service is
# in a bad state (e.g. after a hard crash).  Pre-populating the uname cache
# bypasses the WMI call entirely and has no functional impact.
if sys.platform == "win32" and not getattr(platform, "_uname_cache", None):
    try:
        import struct
        _bits = struct.calcsize("P") * 8
        _machine = "AMD64" if _bits == 64 else "x86"
        platform._uname_cache = platform.uname_result.__new__(
            platform.uname_result, "Windows", "", "", "", _machine
        )
    except Exception:
        pass  # Non-fatal — worst case WMI query runs normally
# ─────────────────────────────────────────────────────────────────────────────

import faulthandler as _faulthandler
_faulthandler.enable()  # print native crash tracebacks to stderr

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from src.utils.logging_config import setup_logging

setup_logging(json_output=True)

from src.indexing.index_builder import HierarchicalIndexer
from src.indexing.vector_store import VectorStore
from src.ingestion.eurlex_ingest import EurLexIngester
from src.ingestion.language_config import get_config
from src.query.orchestrator import QueryOrchestrator, detect_language
from src.query.query_engine import QueryEngine, QueryResult

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Application state (module-level singletons)
# ------------------------------------------------------------------

_vector_store = VectorStore()
_indexer = HierarchicalIndexer(vector_store=_vector_store)
_query_engine = QueryEngine(vector_store=_vector_store, indexer=_indexer)
_orchestrator = QueryOrchestrator(query_engine=_query_engine)
_ingestion_lock = threading.Lock()
_warmup_ok: bool = False


_REQUIRED_ENV_VARS = ("OPENAI_API_KEY", "QDRANT_URL", "QDRANT_API_KEY")

# Server-side deadline for /api/query. Prevents FastAPI's sync-worker pool from
# filling up with hung requests that never return.
_QUERY_TIMEOUT = int(os.getenv("QUERY_TIMEOUT_SECONDS", "150"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    missing = [v for v in _REQUIRED_ENV_VARS if not os.getenv(v)]
    if missing:
        raise RuntimeError(f"Missing required env vars: {', '.join(missing)}. Check your .env file.")

    logger.info("Loading query engine from persisted index...")
    try:
        _orchestrator.load()
        logger.info("Query engine ready.")
    except FileNotFoundError:
        logger.warning(
            "No persisted index found. Run ingestion first: "
            "python -m src.pipelines.ingest_pipeline"
        )

    # Pre-warm the BGE-M3 sparse encoder so it is in memory before the first
    # query arrives.  Without this, the first batch of concurrent eval requests
    # all block on _model_lock while the 570 MB model loads, causing timeouts.
    # Skipped on Windows: PyTorch's XLM-RoBERTa embedding layer raises a native
    # access violation (SIGSEGV) on some Windows/Python 3.13 combinations that
    # cannot be caught by `except Exception` and kills the server process.
    # The model loads lazily on first query instead — acceptable with workers=1.
    global _warmup_ok
    if platform.system() == "Windows":
        logger.info("BGE-M3 warm-up skipped on Windows (lazy load on first query).")
        _warmup_ok = True
    else:
        try:
            from src.indexing.bge_m3_sparse import _encode_query_both
            logger.info("Pre-warming BGE-M3 sparse encoder…")
            _encode_query_both("warm-up")
            logger.info("BGE-M3 warm-up complete.")
            _warmup_ok = True
        except Exception as exc:
            logger.warning("BGE-M3 warm-up skipped: %s", exc)

    yield


app = FastAPI(
    title="EU CRR RAG API",
    description="Regulatory compliance Q&A over EU Capital Requirements Regulation (CRR).",
    version="0.3.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def _global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("GLOBAL_EXC_HANDLER: %s: %s", type(exc).__name__, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"{type(exc).__name__}: {exc}"},
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
# Endpoints
# ------------------------------------------------------------------


@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    if not _orchestrator.is_loaded():
        raise HTTPException(
            status_code=503,
            detail="Index not loaded. Run ingestion first.",
        )

    lang = request.preferred_language  # None → orchestrator auto-detects
    history_dicts = [t.model_dump() for t in request.history]
    cancel = threading.Event()
    try:
        result: QueryResult = await asyncio.wait_for(
            asyncio.to_thread(
                _orchestrator.query,
                request.query,
                language=lang,
                max_cross_ref_expansions=request.max_cross_ref_expansions,
                history=history_dicts,
                cancel=cancel,
            ),
            timeout=_QUERY_TIMEOUT,
        )
    except asyncio.TimeoutError:
        cancel.set()  # signal the background thread to stop at the next checkpoint
        raise HTTPException(status_code=504, detail=f"Query timed out after {_QUERY_TIMEOUT}s.")
    except BaseException as exc:
        logger.error("Query failed: %s: %s", type(exc).__name__, exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {type(exc).__name__}: {exc}",
        )
    # Resolve final language for response (explicit preference or auto-detected)
    try:
        response_lang = lang or detect_language(request.query)
        return QueryResponse(
            answer=result.answer,
            sources=result.sources,
            trace_id=result.trace_id,
            language=response_lang,
        )
    except Exception as exc:
        logger.error("Response construction failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Response construction failed: {type(exc).__name__}: {exc}",
        )


@app.post("/api/query/stream")
async def query_stream(request: QueryRequest, req: Request) -> StreamingResponse:
    if not _orchestrator.is_loaded():
        raise HTTPException(status_code=503, detail="Index not loaded. Run ingestion first.")

    lang = request.preferred_language  # None → orchestrator auto-detects
    history_dicts = [t.model_dump() for t in request.history]
    cancel = threading.Event()

    async def _event_gen() -> AsyncGenerator[str, None]:
        try:
            async for event in _orchestrator.query_stream(
                request.query,
                lang,
                history_dicts,
                request.max_cross_ref_expansions,
                cancel=cancel,
                timeout=_QUERY_TIMEOUT,
            ):
                if await req.is_disconnected():
                    cancel.set()
                    break
                yield event
        except asyncio.TimeoutError:
            cancel.set()
            yield 'data: {"type": "error", "message": "Query timed out."}\n\n'
            yield 'data: {"type": "done"}\n\n'

    return StreamingResponse(
        _event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/article/{article_id}", response_model=ArticleResponse)
def get_article(article_id: str, language: Optional[str] = None) -> ArticleResponse:
    if not _orchestrator.is_loaded():
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
    if not _orchestrator.is_loaded():
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
    from pathlib import Path

    cases_dir = Path("evals/cases/manual_cases")
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
    logger.info("Feedback saved to evals/cases/manual_cases/%s", filename)
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
            _orchestrator.load()
        finally:
            _ingestion_lock.release()

    background_tasks.add_task(_run)
    return IngestResponse(status="started", message="Ingestion started in background.")


@app.get("/api/debug/crash-test")
def debug_crash_test() -> JSONResponse:
    """Run case_161 synchronously (in the main thread) with faulthandler to capture native crashes."""
    import faulthandler
    import traceback as _tb
    debug_log = "case161_api_debug.log"
    with open(debug_log, "w") as fh:
        faulthandler.enable(fh)
        fh.write("=== crash-test start ===\n")
        fh.flush()
        try:
            query = (
                "For an exposure to a corporate without an external rating, what is the risk weight, "
                "and how does this compare to an exposure to an unrated institution assigned to Grade B?"
            )
            from src.query.orchestrator import _MULTI_HOP_RE
            fh.write(f"is_multi_hop: {bool(_MULTI_HOP_RE.search(query))}\n")
            fh.flush()
            result = _orchestrator.query(query, language="en")
            fh.write(f"SUCCESS: {result.answer[:200]}\n")
        except BaseException as exc:
            fh.write(f"EXCEPTION: {type(exc).__name__}: {exc}\n")
            _tb.print_exc(file=fh)
        finally:
            fh.write("=== crash-test end ===\n")
            faulthandler.disable()
    return JSONResponse({"status": "done", "log": debug_log})


@app.get("/health")
def health() -> JSONResponse:
    loaded = _orchestrator.is_loaded()
    ready = loaded and _warmup_ok
    status_code = 200 if ready else 503
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ok" if ready else "degraded",
            "index_loaded": loaded,
            "warmup_ok": _warmup_ok,
            "vector_store_items": _vector_store.item_count,
        },
    )
