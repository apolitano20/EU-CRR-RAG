"""
E5-large-instruct embedding wrapper for LlamaIndex.

Wraps intfloat/multilingual-e5-large-instruct (560 MB, 1024-dim dense vectors).
Uses instruction-tuned query/passage prefixes for domain-specific retrieval.

Thread-safe via double-checked locking (same pattern as bge_m3_sparse.py).
"""
from __future__ import annotations

import threading

from llama_index.core.embeddings import BaseEmbedding
from pydantic import ConfigDict

_QUERY_INSTRUCTION = (
    "Instruct: Retrieve the relevant EU Capital Requirements Regulation article "
    "for this regulatory question\nQuery: "
)
_PASSAGE_PREFIX = "passage: "

_model = None
_model_lock = threading.Lock()
_encode_lock = threading.Lock()


def _get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                import torch
                from sentence_transformers import SentenceTransformer

                m = SentenceTransformer(
                    "intfloat/multilingual-e5-large-instruct",
                    trust_remote_code=False,
                )
                if torch.cuda.is_available():
                    m = m.to("cuda")
                _model = m
    return _model


class E5InstructEmbedding(BaseEmbedding):
    """LlamaIndex BaseEmbedding wrapper for multilingual-e5-large-instruct.

    Produces 1024-dim dense vectors only (no sparse). The instruction prefix
    and passage prefix are applied transparently inside this class.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_query_embedding(self, query: str) -> list[float]:
        text = _QUERY_INSTRUCTION + query
        with _encode_lock:
            vec = _get_model().encode(text, normalize_embeddings=True)
        return vec.tolist()

    def _get_text_embedding(self, text: str) -> list[float]:
        prefixed = _PASSAGE_PREFIX + text
        with _encode_lock:
            vec = _get_model().encode(prefixed, normalize_embeddings=True)
        return vec.tolist()

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        prefixed = [_PASSAGE_PREFIX + t for t in texts]
        with _encode_lock:
            vecs = _get_model().encode(
                prefixed,
                normalize_embeddings=True,
                batch_size=12,
            )
        return [v.tolist() for v in vecs]

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._get_text_embedding(text)
