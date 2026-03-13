"""
BGE-M3 sparse encoding helpers for Qdrant hybrid search.

Uses a module-level singleton to avoid reloading the 570 MB model on each call.
Thread-safe via double-checked locking.
"""
from __future__ import annotations

import threading

_model = None
_model_lock = threading.Lock()


def _get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from FlagEmbedding import BGEM3FlagModel
                _model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    return _model


def sparse_doc_fn(texts: list[str]) -> tuple[list[list[int]], list[list[float]]]:
    """Encode a batch of texts and return (indices, values) for Qdrant hybrid search."""
    output = _get_model().encode(
        texts,
        return_dense=False,
        return_sparse=True,
        return_colbert_vecs=False,
        batch_size=12,
    )
    indices, values = [], []
    for w in output["lexical_weights"]:
        indices.append([int(k) for k in w])
        values.append([float(v) for v in w.values()])
    return indices, values


def sparse_query_fn(query: list[str]) -> tuple[list[list[int]], list[list[float]]]:
    """Encode a query batch and return (indices, values) in batch format for Qdrant.

    LlamaIndex's Qdrant adapter calls this with a list[str] (one element), and
    then indexes into the result as sparse_indices[0] / sparse_embedding[0].
    Therefore we must return the full batch format — NOT the unpacked single result.
    """
    return sparse_doc_fn(query)
