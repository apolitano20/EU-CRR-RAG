"""
FastAPI application for the EU CRR RAG system.

Endpoints:
  POST /api/query   – submit a compliance question, get answer + sources
  POST /api/ingest  – trigger the ingestion pipeline
"""
from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

from src.indexing.index_builder import HierarchicalIndexer
from src.indexing.vector_store import VectorStore
from src.ingestion.eurlex_ingest import DEFAULT_CRR_URL, EurLexIngester
from src.query.query_engine import QueryEngine, QueryResult

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Application state (module-level singletons)
# ------------------------------------------------------------------

_vector_store = VectorStore()
_indexer = HierarchicalIndexer(vector_store=_vector_store)
_query_engine = QueryEngine(vector_store=_vector_store, indexer=_indexer)
_ingestion_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
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
    version="0.1.0",
    lifespan=lifespan,
)

# ------------------------------------------------------------------
# Request / Response models
# ------------------------------------------------------------------


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    trace_id: str


class IngestRequest(BaseModel):
    url: str = DEFAULT_CRR_URL
    reset: bool = False


class IngestResponse(BaseModel):
    status: str
    message: str


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------


@app.post("/api/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    if _query_engine._engine is None:
        raise HTTPException(
            status_code=503,
            detail="Index not loaded. Run ingestion first.",
        )
    result: QueryResult = _query_engine.query(request.query)
    return QueryResponse(
        answer=result.answer,
        sources=result.sources,
        trace_id=result.trace_id,
    )


@app.post("/api/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest, background_tasks: BackgroundTasks) -> IngestResponse:
    if not _ingestion_lock.acquire(blocking=False):
        return IngestResponse(status="busy", message="Ingestion already in progress.")

    def _run() -> None:
        try:
            ingester = EurLexIngester(url=request.url)
            docs = ingester.load()
            indexer = HierarchicalIndexer(vector_store=_vector_store, reset_store=request.reset)
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
        "index_loaded": _query_engine._engine is not None,
        "vector_store_items": _vector_store.item_count,
    }
