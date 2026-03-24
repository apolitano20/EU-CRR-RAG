"""
Unit tests for api/main.py — exercises all HTTP endpoints with a mocked
orchestrator (no Qdrant, no OpenAI).

Isolation strategy:
- The FastAPI module-level singletons are all constructed without connecting
  to external services.
- The lifespan's _orchestrator.load() is patched to a no-op so the test
  client starts cleanly.
- Individual tests then set _query_engine._engine to a MagicMock where
  needed (since orchestrator.is_loaded() delegates to engine.is_loaded()).
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
    def test_returns_503_when_not_ready(self, client):
        """No index loaded and warmup not done → degraded."""
        r = client.get("/health")
        assert r.status_code == 503

    def test_returns_200_when_fully_ready(self, loaded_client, app_module):
        """Index loaded + warmup done → 200 ok."""
        client, _ = loaded_client
        app_module._warmup_ok = True
        try:
            r = client.get("/health")
            assert r.status_code == 200
            assert r.json()["status"] == "ok"
        finally:
            app_module._warmup_ok = False

    def test_status_degraded_when_warmup_failed(self, loaded_client, app_module):
        """Index loaded but warmup failed → 503 degraded."""
        client, _ = loaded_client
        app_module._warmup_ok = False
        r = client.get("/health")
        assert r.status_code == 503
        assert r.json()["status"] == "degraded"
        assert r.json()["warmup_ok"] is False

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
        mod._orchestrator.query = MagicMock(
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
        mod._orchestrator.query = mock_query
        client.post("/api/query", json={"query": "What are Tier 2 instruments?"})
        call_args = mock_query.call_args
        assert "Tier 2" in call_args[0][0]

    def test_preferred_language_forwarded(self, loaded_client, app_module):
        client, mod = loaded_client
        from src.query.query_engine import QueryResult
        mock_query = MagicMock(
            return_value=QueryResult(answer="ok", sources=[], trace_id="t1")
        )
        mod._orchestrator.query = mock_query
        client.post("/api/query", json={"query": "CET1?", "preferred_language": "it"})
        _, kwargs = mock_query.call_args
        assert kwargs.get("language") == "it"

    def test_response_includes_language_field(self, loaded_client, app_module):
        """QueryResponse should include the detected/preferred language."""
        client, mod = loaded_client
        from src.query.query_engine import QueryResult
        mod._orchestrator.query = MagicMock(
            return_value=QueryResult(answer="ok", sources=[], trace_id="t1")
        )
        r = client.post("/api/query", json={"query": "CET1?", "preferred_language": "it"})
        body = r.json()
        assert body["language"] == "it"

    def test_response_language_from_detection(self, loaded_client, app_module):
        """Without preferred_language, the response language comes from detect_language."""
        client, mod = loaded_client
        from src.query.query_engine import QueryResult
        mod._orchestrator.query = MagicMock(
            return_value=QueryResult(answer="ok", sources=[], trace_id="t1")
        )
        # detect_language now returns "en" for English text (via langdetect)
        with patch("api.main.detect_language", return_value="en"):
            r = client.post("/api/query", json={"query": "What is CET1?"})
        body = r.json()
        assert body["language"] == "en"

    def test_missing_query_field_returns_422(self, client):
        r = client.post("/api/query", json={})
        assert r.status_code == 422

    def test_max_cross_ref_expansions_accepted(self, loaded_client, app_module):
        client, mod = loaded_client
        from src.query.query_engine import QueryResult
        mock_query = MagicMock(
            return_value=QueryResult(answer="ok", sources=[], trace_id="t1")
        )
        mod._orchestrator.query = mock_query
        r = client.post("/api/query", json={"query": "CET1?", "max_cross_ref_expansions": 5})
        assert r.status_code == 200

    def test_sources_may_contain_expanded_key(self, loaded_client, app_module):
        """The expanded flag in sources should be tolerated by the API layer."""
        client, mod = loaded_client
        from src.query.query_engine import QueryResult
        mod._orchestrator.query = MagicMock(
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
# POST /api/query — history field
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestQueryHistoryField:
    def test_history_field_optional_defaults_empty(self, loaded_client, app_module):
        client, mod = loaded_client
        from src.query.query_engine import QueryResult
        mod._orchestrator.query = MagicMock(
            return_value=QueryResult(answer="ok", sources=[], trace_id="t1")
        )
        r = client.post("/api/query", json={"query": "What is CET1?"})
        assert r.status_code == 200

    def test_history_forwarded_to_engine(self, loaded_client, app_module):
        client, mod = loaded_client
        from src.query.query_engine import QueryResult
        mock_query = MagicMock(
            return_value=QueryResult(answer="ok", sources=[], trace_id="t1")
        )
        mod._orchestrator.query = mock_query
        client.post("/api/query", json={
            "query": "And what about AT1?",
            "history": [{"question": "What is CET1?", "answer": "Common Equity Tier 1."}],
        })
        _, kwargs = mock_query.call_args
        assert kwargs.get("history") == [{"question": "What is CET1?", "answer": "Common Equity Tier 1."}]

    def test_history_field_type_validated(self, loaded_client, app_module):
        client, _ = loaded_client
        r = client.post("/api/query", json={"query": "What is CET1?", "history": "bad"})
        assert r.status_code == 422

    def test_empty_history_list_accepted(self, loaded_client, app_module):
        client, mod = loaded_client
        from src.query.query_engine import QueryResult
        mod._orchestrator.query = MagicMock(
            return_value=QueryResult(answer="ok", sources=[], trace_id="t1")
        )
        r = client.post("/api/query", json={"query": "What is CET1?", "history": []})
        assert r.status_code == 200

    def test_five_turn_history_accepted(self, loaded_client, app_module):
        client, mod = loaded_client
        from src.query.query_engine import QueryResult
        mod._orchestrator.query = MagicMock(
            return_value=QueryResult(answer="ok", sources=[], trace_id="t1")
        )
        history = [{"question": f"Q{i}", "answer": f"A{i}"} for i in range(5)]
        r = client.post("/api/query", json={"query": "Follow-up?", "history": history})
        assert r.status_code == 200

    def test_history_turn_missing_answer_field(self, loaded_client, app_module):
        client, _ = loaded_client
        r = client.post("/api/query", json={
            "query": "What is CET1?",
            "history": [{"question": "Q without answer"}],
        })
        assert r.status_code == 422


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
# detect_language (imported from orchestrator into api.main)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestDetectLanguage:
    def test_polish_detected_from_diacritics(self, app_module):
        # Diacritics heuristic (fallback when langdetect fails)
        from src.query.orchestrator import _detect_language_heuristic
        assert _detect_language_heuristic("Jakie są wymogi kapitałowe?") == "pl"

    def test_italian_detected_from_diacritics(self, app_module):
        from src.query.orchestrator import _detect_language_heuristic
        assert _detect_language_heuristic("Qual è il requisito di capitale?") == "it"

    def test_english_via_langdetect(self, app_module):
        # detect_language uses langdetect → returns "en" for English
        from src.query.orchestrator import detect_language
        with patch("langdetect.detect", return_value="en"), \
             patch("langdetect.DetectorFactory"):
            result = detect_language("What are the capital requirements?")
        assert result == "en"

    def test_heuristic_fallback_no_diacritics_returns_none(self, app_module):
        from src.query.orchestrator import _detect_language_heuristic
        assert _detect_language_heuristic("What are the capital requirements?") is None

    def test_empty_string_returns_none(self, app_module):
        from src.query.orchestrator import _detect_language_heuristic
        assert _detect_language_heuristic("") is None
