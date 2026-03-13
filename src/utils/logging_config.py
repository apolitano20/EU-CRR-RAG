"""
Structured logging setup for the EU CRR RAG system.

Provides a JSON formatter so every log line is machine-parseable by
any log aggregator (Datadog, Loki, CloudWatch, etc.).

Usage:
    from src.utils.logging_config import setup_logging
    setup_logging()          # JSON output (default)
    setup_logging(json=False)  # human-readable for local dev
"""
from __future__ import annotations

import json
import logging
import sys
import time
from typing import Optional


class _JsonFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "ts": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # Pass-through any extra fields attached to the record
        for key, val in record.__dict__.items():
            if key not in {
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "message",
                "taskName",
            }:
                payload[key] = val
        return json.dumps(payload, default=str)


def setup_logging(
    level: int = logging.INFO,
    json_output: bool = True,
    stream=None,
) -> None:
    """Configure root logger with either JSON or plain-text formatting.

    Call once at process startup (ingest_pipeline, api/main).
    Subsequent calls are idempotent — the root logger is only configured
    if it has no handlers yet.
    """
    root = logging.getLogger()
    if root.handlers:
        return  # already configured

    handler = logging.StreamHandler(stream or sys.stdout)
    if json_output:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )

    root.setLevel(level)
    root.addHandler(handler)

    # Quiet noisy third-party loggers
    for noisy in ("httpx", "httpcore", "sentence_transformers", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
