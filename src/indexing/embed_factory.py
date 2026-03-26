"""
Embedding model factory — routes EMBED_MODEL env var to the correct embedding
class and sparse configuration.

Usage:
    from src.indexing.embed_factory import get_embed_config
    cfg = get_embed_config()
    Settings.embed_model = cfg.embed_model

EMBED_MODEL values:
  "bge-m3"           (default) — BAAI/bge-m3: 1024-dim dense + sparse, enable_hybrid=True
  "e5-large-instruct"          — multilingual-e5-large-instruct: 1024-dim dense only, enable_hybrid=False
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class EmbedConfig:
    embed_model: object
    sparse_doc_fn: Optional[Callable]
    sparse_query_fn: Optional[Callable]
    enable_hybrid: bool


def get_embed_config() -> EmbedConfig:
    """Return the embedding config for the currently configured EMBED_MODEL."""
    model_key = os.getenv("EMBED_MODEL", "bge-m3").lower().strip()

    if model_key == "e5-large-instruct":
        from src.indexing.e5_instruct_embed import E5InstructEmbedding
        return EmbedConfig(
            embed_model=E5InstructEmbedding(),
            sparse_doc_fn=None,
            sparse_query_fn=None,
            enable_hybrid=False,
        )

    # Default: bge-m3
    from src.indexing.bge_m3_sparse import BGEm3Embedding, sparse_doc_fn, sparse_query_fn
    return EmbedConfig(
        embed_model=BGEm3Embedding(),
        sparse_doc_fn=sparse_doc_fn,
        sparse_query_fn=sparse_query_fn,
        enable_hybrid=True,
    )
