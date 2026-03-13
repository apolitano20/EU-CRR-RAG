"""
Unit-test configuration.

Stubs out packages that are not installed in the development venv but are
imported at module level by the production code.  Stubs are injected into
sys.modules BEFORE any test module is imported so that `import api.main`
(and similar) succeed without a live Qdrant / BGE-M3 environment.

WARNING — test isolation:
These stubs are injected into sys.modules for the lifetime of the process and
are never removed.  Running unit tests and integration tests in the same pytest
invocation (e.g. `pytest tests/`) will cause the stubs to shadow the real
qdrant_client and FlagEmbedding packages, breaking the integration suite.

Always run each suite in a separate invocation:
    pytest tests/unit/
    pytest tests/integration/
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock


def _stub(name: str) -> MagicMock:
    m = MagicMock(name=name)
    sys.modules[name] = m
    return m


# Qdrant client (not installed in unit-test venv)
if "qdrant_client" not in sys.modules:
    qc = _stub("qdrant_client")
    _stub("qdrant_client.http")
    models_mock = _stub("qdrant_client.http.models")
    # Provide named constants used at import time in vector_store.py
    models_mock.Distance = MagicMock()
    models_mock.Distance.COSINE = "Cosine"
    models_mock.VectorParams = MagicMock()
    models_mock.SparseVectorParams = MagicMock()
    models_mock.SparseIndexParams = MagicMock()
    models_mock.PayloadSchemaType = MagicMock()
    models_mock.PayloadSchemaType.KEYWORD = "keyword"

# LlamaIndex Qdrant vector store adapter
if "llama_index.vector_stores.qdrant" not in sys.modules:
    _stub("llama_index.vector_stores.qdrant")

# BGE-M3 / FlagEmbedding (large model, not installed in unit-test venv)
if "FlagEmbedding" not in sys.modules:
    _stub("FlagEmbedding")
