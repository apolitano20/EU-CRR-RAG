"""
BGE-M3 sparse encoding helpers for Qdrant hybrid search, plus a LlamaIndex
BaseEmbedding wrapper (BGEm3Embedding) that reuses the same singleton to avoid
loading a second copy of the 570 MB model in memory.

Thread-safe via double-checked locking.
"""
from __future__ import annotations

import threading
from llama_index.core.embeddings import BaseEmbedding
from pydantic import ConfigDict

_model = None
_model_lock = threading.Lock()
# Serialize query-time encode() calls on CPU.  FlagEmbedding is not guaranteed
# re-entrant, and concurrent CPU encodes cause latency spikes under parallel
# load.  Throughput drops but latency becomes predictable.
_encode_lock = threading.Lock()

# Per-thread cache: stores the last {query → (dense_vec, sparse_indices, sparse_values)}
# so that when LlamaIndex calls _get_query_embedding() then sparse_query_fn() for the
# same query string, the second call skips the encode entirely.  Thread-local storage
# means workers don't share or race on cache entries.
_query_cache = threading.local()


def _get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                import torch
                from FlagEmbedding import BGEM3FlagModel
                _model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
                # FlagEmbedding 1.x doesn't reliably honour a device= constructor
                # arg — explicitly move to CUDA if available.
                if torch.cuda.is_available():
                    _model.model = _model.model.to("cuda")
    return _model


def _encode_query_both(query: str) -> dict:
    """Encode a single query string for BOTH dense and sparse in one lock acquisition.

    LlamaIndex's Qdrant adapter calls dense and sparse encoders separately.  Doing
    two separate encode() calls means acquiring _encode_lock twice per query: with N
    concurrent workers that doubles the queue depth and roughly doubles median latency.
    This function fetches both representations in a single locked call and caches the
    result on the calling thread so the second adapter call is a free cache hit.
    """
    cached = getattr(_query_cache, "last", None)
    if cached is not None and cached[0] == query:
        return cached[1]

    with _encode_lock:
        # No inner re-check needed: _query_cache is threading.local(), so no other
        # thread can populate this thread's cache entry between the outer check and here.
        output = _get_model().encode(
            [query],
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

    result = {
        "dense": output["dense_vecs"][0].tolist(),
        "sparse_indices": [int(k) for k in output["lexical_weights"][0]],
        "sparse_values": [float(v) for v in output["lexical_weights"][0].values()],
    }
    _query_cache.last = (query, result)
    return result


def sparse_doc_fn(texts: list[str]) -> tuple[list[list[int]], list[list[float]]]:
    """Encode a batch of texts and return (indices, values) for Qdrant hybrid search."""
    with _encode_lock:
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

    Uses _encode_query_both() so that if BGEm3Embedding._get_query_embedding() was
    already called for this query on the same thread, the result is a free cache hit
    (no second lock acquisition, no second encode).
    """
    if len(query) == 1:
        result = _encode_query_both(query[0])
        return [[result["sparse_indices"]], [result["sparse_values"]]]
    return sparse_doc_fn(query)


class BGEm3Embedding(BaseEmbedding):
    """LlamaIndex BaseEmbedding wrapper around the BGE-M3 FlagEmbedding singleton.

    Reuses the already-loaded model instead of creating a second 570 MB instance
    via HuggingFaceEmbedding / sentence_transformers.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_query_embedding(self, query: str) -> list[float]:
        # _encode_query_both() computes dense + sparse together and caches on this
        # thread.  When LlamaIndex then calls sparse_query_fn() for the same query,
        # it will be a free cache hit — one lock acquisition instead of two.
        return _encode_query_both(query)["dense"]

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._get_query_embedding(text)

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        with _encode_lock:
            output = _get_model().encode(
                texts,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
                batch_size=12,
            )
        return [v.tolist() for v in output["dense_vecs"]]

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._get_text_embedding(text)
