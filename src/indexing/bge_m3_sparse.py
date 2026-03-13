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


def _get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                import torch
                from FlagEmbedding import BGEM3FlagModel
                device = "cuda" if torch.cuda.is_available() else "cpu"
                _model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device=device)
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


class BGEm3Embedding(BaseEmbedding):
    """LlamaIndex BaseEmbedding wrapper around the BGE-M3 FlagEmbedding singleton.

    Reuses the already-loaded model instead of creating a second 570 MB instance
    via HuggingFaceEmbedding / sentence_transformers.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_query_embedding(self, query: str) -> list[float]:
        output = _get_model().encode(
            [query],
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return output["dense_vecs"][0].tolist()

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._get_query_embedding(text)

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
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
