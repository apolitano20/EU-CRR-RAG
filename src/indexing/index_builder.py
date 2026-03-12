"""
HierarchicalIndexer: parses Documents into a legal-hierarchy node tree,
builds a Chroma vector index and a BM25 index, then persists both.
"""
from __future__ import annotations

import logging
import os
import pickle
import shutil
from pathlib import Path
from typing import Optional

from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.schema import BaseNode
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.retrievers.bm25 import BM25Retriever

from src.indexing.vector_store import VectorStore

logger = logging.getLogger(__name__)

# Chunk sizes per level (parent → child): PART/TITLE → CHAPTER/SECTION → ARTICLE/PARAGRAPH
CHUNK_SIZES = [2048, 512, 128]
BM25_INDEX_PATH = "./bm25_index.pkl"
DOCSTORE_PERSIST_DIR = "./docstore"


class HierarchicalIndexer:
    """Builds and persists the vector + BM25 indices from a list of Documents."""

    def __init__(
        self,
        vector_store: VectorStore,
        embed_model_name: str = "BAAI/bge-small-en-v1.5",
        bm25_index_path: str = BM25_INDEX_PATH,
        docstore_persist_dir: str = DOCSTORE_PERSIST_DIR,
        reset_store: bool = False,
    ) -> None:
        self.vector_store = vector_store
        self.embed_model_name = embed_model_name
        self.bm25_index_path = bm25_index_path
        self.docstore_persist_dir = docstore_persist_dir
        self.reset_store = reset_store

        self._leaf_nodes: Optional[list[BaseNode]] = None
        self._vector_index: Optional[VectorStoreIndex] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, documents: list[Document]) -> VectorStoreIndex:
        """Parse documents, index them, persist BM25. Returns the vector index."""
        self._configure_settings()

        if self.reset_store:
            self.vector_store.reset()
            shutil.rmtree(self.docstore_persist_dir, ignore_errors=True)
        else:
            self.vector_store.connect()

        logger.info("Parsing %d document(s) with HierarchicalNodeParser.", len(documents))
        all_nodes = self._parse_nodes(documents)
        self._leaf_nodes = get_leaf_nodes(all_nodes)
        logger.info("Total nodes: %d  |  Leaf nodes: %d", len(all_nodes), len(self._leaf_nodes))

        self._vector_index = self._build_vector_index(all_nodes, self._leaf_nodes)
        self._save_bm25_index(self._leaf_nodes)

        return self._vector_index

    def load(self) -> VectorStoreIndex:
        """Re-open an existing persisted index (no re-ingestion)."""
        self._configure_settings()
        self.vector_store.connect()

        storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store.as_llama_vector_store(),
            persist_dir=self.docstore_persist_dir,  # reloads docstore.json
        )
        self._vector_index = VectorStoreIndex.from_vector_store(
            self.vector_store.as_llama_vector_store(),
            storage_context=storage_context,
        )
        logger.info("Loaded existing vector index (%d items).", self.vector_store.item_count)
        return self._vector_index

    def load_bm25_retriever(self, similarity_top_k: int = 12) -> BM25Retriever:
        nodes = self._load_bm25_nodes()
        return BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=similarity_top_k)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _configure_settings(self) -> None:
        Settings.embed_model = HuggingFaceEmbedding(model_name=self.embed_model_name)
        Settings.llm = None  # LLM is set per-query in QueryEngine

    def _parse_nodes(self, documents: list[Document]) -> list[BaseNode]:
        parser = HierarchicalNodeParser.from_defaults(chunk_sizes=CHUNK_SIZES)
        return parser.get_nodes_from_documents(documents)

    def _build_vector_index(
        self, all_nodes: list[BaseNode], leaf_nodes: list[BaseNode]
    ) -> VectorStoreIndex:
        llama_vector_store = self.vector_store.as_llama_vector_store()

        # ALL nodes go into the docstore so AutoMergingRetriever can climb the tree
        docstore = SimpleDocumentStore()
        docstore.add_documents(all_nodes)

        storage_context = StorageContext.from_defaults(
            vector_store=llama_vector_store,
            docstore=docstore,
        )
        # Only LEAF nodes are embedded into the vector index
        index = VectorStoreIndex(
            nodes=leaf_nodes,
            storage_context=storage_context,
            show_progress=True,
        )
        storage_context.persist(persist_dir=self.docstore_persist_dir)
        logger.info("Vector index built with %d items.", self.vector_store.item_count)
        return index

    def _save_bm25_index(self, leaf_nodes: list[BaseNode]) -> None:
        path = Path(self.bm25_index_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(leaf_nodes, f)
        logger.info("BM25 leaf nodes saved to %s.", path)

    def _load_bm25_nodes(self) -> list[BaseNode]:
        path = Path(self.bm25_index_path)
        if not path.exists():
            raise FileNotFoundError(
                f"BM25 index not found at {path}. Run ingestion first."
            )
        with open(path, "rb") as f:
            return pickle.load(f)
