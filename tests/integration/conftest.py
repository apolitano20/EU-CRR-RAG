"""
Integration test configuration.

Loads .env and skips the entire integration suite if required env vars
are missing — so CI without credentials doesn't fail noisily.
"""
from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

# Load .env before any test in this directory
load_dotenv()

REQUIRED_VARS = ("QDRANT_URL", "QDRANT_API_KEY", "OPENAI_API_KEY")


def pytest_collection_modifyitems(config, items):
    missing = [v for v in REQUIRED_VARS if not os.getenv(v)]
    if not missing:
        return
    skip = pytest.mark.skip(
        reason=f"Integration env vars not set: {', '.join(missing)}"
    )
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(skip)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def qdrant_url() -> str:
    return os.environ["QDRANT_URL"]


@pytest.fixture(scope="session")
def qdrant_api_key() -> str:
    return os.environ["QDRANT_API_KEY"]


@pytest.fixture(scope="session")
def openai_api_key() -> str:
    return os.environ["OPENAI_API_KEY"]


@pytest.fixture(scope="session")
def test_collection_name() -> str:
    """Use a dedicated test collection to avoid clobbering the production index."""
    return "eu_crr_test"
