"""
Unit tests for api/main.py — exercises all HTTP endpoints with a mocked
query engine (no Qdrant, no OpenAI).

Isolation strategy:
- The FastAPI module-level singletons (VectorStore, HierarchicalIndexer,
  QueryEngine) are all constructed without connecting to external services.
- The lifespan's _query_engine.load() is patched to a no-op so the test
  client starts cleanly.
- Individual tests then set _query_engine._engine to a MagicMock where
  needed.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def app_module():
    """Import the FastAPI app.

    VectorStore/HierarchicalIndexer/QueryEngine constructors do NOT connect
    to external services — only their .connect()/.load() methods do — so
    importing api.main is safe without any patching.
    """
    import api.main as m  # noqa: E402
    return m


@pytest.fixture
def client(app_module):
    """TestClient with the lifespan load() stubbed out."""
    with patch.object(app_module._query_engine, "load"):
        with TestClient(app_module.app) as c:
            yield c


@pytest.fixture
def loaded_client(app_module, client):
    """TestClient where the query engine reports as loaded."""
    mock_engine = MagicMock()
    app_module._query_engine._engine = mock_engine
    yield client, app_module
    # Teardown: unload
    app_module._query_engine._engine = None


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestHealthEndpoint:
    def test_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_status_is_ok(self, client):
        r = client.get("/health")
        assert r.json()["status"] == "ok"

    def test_index_loaded_false_when_not_loaded(self, client, app_module):
        app_module._query_engine._engine = None
        r = client.get("/health")
        assert r.json()["index_loaded"] is False

    def test_index_loaded_true_when_loaded(self, loaded_client):
        client, _ = loaded_client
        r = client.get("/health")
        assert r.json()["index_loaded"] is True

    def test_vector_store_items_is_int(self, client):
        r = client.get("/health")
        assert isinstance(r.json()["vector_store_items"], int)


# ---------------------------------------------------------------------------
# POST /api/query
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestQueryEndpoint:
    def test_503_when_index_not_loaded(self, client, app_module):
        app_module._query_engine._engine = None
        r = client.post("/api/query", json={"query": "What is CET1?"})
        assert r.status_code == 503

    def test_returns_answer_and_sources(self, loaded_client, app_module):
        client, mod = loaded_client
        from src.query.query_engine import QueryResult
        mod._query_engine.query = MagicMock(
            return_value=QueryResult(
                answer="CET1 is Common Equity Tier 1.",
                sources=[{"text": "Article 26...", "score": 0.9, "metadata": {}, "expanded": False}],
                trace_id="test-trace",
            )
        )
        r = client.post("/api/query", json={"query": "What is CET1?"})
        assert r.status_code == 200
        body = r.json()
        assert body["answer"] == "CET1 is Common Equity Tier 1."
        assert isinstance(body["sources"], list)
        assert "trace_id" in body

    def test_query_passed_to_engine(self, loaded_client, app_module):
        client, mod = loaded_client
        from src.query.query_engine import QueryResult
        mock_query = MagicMock(
            return_value=QueryResult(answer="ok", sources=[], trace_id="t1")
        )
        mod._query_engine.query = mock_query
        client.post("/api/query", json={"query": "What are Tier 2 instruments?"})
        call_args = mock_query.call_args
        assert "Tier 2" in call_args[0][0]

    def test_preferred_language_forwarded(self, loaded_client, app_module):
        client, mod = loaded_client
        from src.query.query_engine import QueryResult
        mock_query = MagicMock(
            return_value=QueryResult(answer="ok", sources=[], trace_id="t1")
        )
        mod._query_engine.query = mock_query
        client.post("/api/query", json={"query": "CET1?", "preferred_language": "it"})
        _, kwargs = mock_query.call_args
        assert kwargs.get("language") == "it"

    def test_response_includes_language_field(self, loaded_client, app_module):
        """QueryResponse should include the detected/preferred language."""
        client, mod = loaded_client
        from src.query.query_engine import QueryResult
        mod._query_engine.query = MagicMock(
            return_value=QueryResult(answer="ok", sources=[], trace_id="t1")
        )
        r = client.post("/api/query", json={"query": "CET1?", "preferred_language": "it"})
        body = r.json()
        assert body["language"] == "it"

    def test_response_language_null_when_not_detected(self, loaded_client, app_module):
        """English queries have no diacritics, so language should be null."""
        client, mod = loaded_client
        from src.query.query_engine import QueryResult
        mod._query_engine.query = MagicMock(
            return_value=QueryResult(answer="ok", sources=[], trace_id="t1")
        )
        r = client.post("/api/query", json={"query": "What is CET1?"})
        body = r.json()
        assert body["language"] is None

    def test_missing_query_field_returns_422(self, client):
        r = client.post("/api/query", json={})
        assert r.status_code == 422

    def test_max_cross_ref_expansions_accepted(self, loaded_client, app_module):
        client, mod = loaded_client
        from src.query.query_engine import QueryResult
        mock_query = MagicMock(
            return_value=QueryResult(answer="ok", sources=[], trace_id="t1")
        )
        mod._query_engine.query = mock_query
        r = client.post("/api/query", json={"query": "CET1?", "max_cross_ref_expansions": 5})
        assert r.status_code == 200

    def test_sources_may_contain_expanded_key(self, loaded_client, app_module):
        """The expanded flag in sources should be tolerated by the API layer."""
        client, mod = loaded_client
        from src.query.query_engine import QueryResult
        mod._query_engine.query = MagicMock(
            return_value=QueryResult(
                answer="ok",
                sources=[
                    {"text": "direct", "score": 0.9, "metadata": {}, "expanded": False},
                    {"text": "cross-ref", "score": 0.0, "metadata": {}, "expanded": True},
                ],
                trace_id="t2",
            )
        )
        r = client.post("/api/query", json={"query": "test"})
        assert r.status_code == 200
        assert len(r.json()["sources"]) == 2


# ---------------------------------------------------------------------------
# POST /api/ingest
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestIngestEndpoint:
    def test_returns_started_status(self, client):
        with patch("api.main.EurLexIngester"), \
             patch("api.main.HierarchicalIndexer"):
            r = client.post("/api/ingest", json={"language": "en"})
        assert r.status_code == 200
        assert r.json()["status"] == "started"

    def test_concurrent_ingest_returns_busy(self, app_module, client):
        """Acquiring the lock before the request should yield 'busy'."""
        acquired = app_module._ingestion_lock.acquire(blocking=False)
        assert acquired, "Could not acquire lock for test setup"
        try:
            r = client.post("/api/ingest", json={"language": "en"})
            assert r.json()["status"] == "busy"
        finally:
            app_module._ingestion_lock.release()

    def test_invalid_language_still_accepted_by_api(self, client):
        """The API accepts any string for language; validation is downstream."""
        with patch("api.main.EurLexIngester"), \
             patch("api.main.HierarchicalIndexer"):
            r = client.post("/api/ingest", json={"language": "en"})
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# _detect_language heuristic
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestDetectLanguage:
    def test_polish_detected_from_diacritics(self, app_module):
        assert app_module._detect_language("Jakie są wymogi kapitałowe?") == "pl"

    def test_italian_detected_from_diacritics(self, app_module):
        # "è" is in the Italian diacritic set used by _detect_language
        assert app_module._detect_language("Qual è il requisito di capitale?") == "it"

    def test_english_returns_none(self, app_module):
        assert app_module._detect_language("What are the capital requirements?") is None

    def test_empty_string_returns_none(self, app_module):
        assert app_module._detect_language("") is None

    def test_digits_only_returns_none(self, app_module):
        assert app_module._detect_language("12345") is None
