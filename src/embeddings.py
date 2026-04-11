"""
embeddings.py — Embedding generation (sentence-transformers) + FAISS index.
Handles batch encoding, index construction, and similarity search.
"""

from __future__ import annotations
from typing import List, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from .config import EMBEDDING_MODEL, EMBEDDING_DIM, TOP_K_CHUNKS
from .models import PageChunk
from .logger import get_logger

logger = get_logger(__name__)

# Module-level singleton for the embedding model (lazy loaded)
_embedding_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    """Lazy-load and cache the embedding model."""
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Embedding model loaded")
    return _embedding_model


class FAISSIndex:
    """
    FAISS flat inner-product index for chunk retrieval.
    Wraps the index + chunk list for easy retrieval.
    """

    def __init__(self):
        self.index: faiss.IndexFlatIP | None = None
        self.chunks: List[PageChunk] = []
        self._model = None

    def build(self, chunks: List[PageChunk]) -> None:
        """
        Encode chunks and build FAISS index.

        Args:
            chunks: List of PageChunk objects to index.
        """
        if not chunks:
            raise ValueError("Cannot build index from empty chunk list")

        self._model = get_embedding_model()
        self.chunks = chunks

        texts = [c.text for c in chunks]

        logger.info(f"Encoding {len(texts)} chunks...")
        embeddings = self._model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,   # Required for inner product = cosine sim
            convert_to_numpy=True,
        ).astype(np.float32)

        dim = embeddings.shape[1]
        if dim != EMBEDDING_DIM:
            logger.warning(f"Embedding dim mismatch: expected {EMBEDDING_DIM}, got {dim}")

        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        logger.info(f"FAISS index built: {self.index.ntotal} vectors, dim={dim}")

    def search(
        self,
        query: str,
        top_k: int = TOP_K_CHUNKS,
    ) -> List[Tuple[PageChunk, float]]:
        """
        Retrieve top-k most relevant chunks for a query.

        Args:
            query: Natural language query string.
            top_k: Number of results to return.

        Returns:
            List of (PageChunk, similarity_score) sorted by score descending.
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build() first.")

        model = self._model or get_embedding_model()

        query_embedding = model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        actual_k = min(top_k, len(self.chunks))
        scores, indices = self.index.search(query_embedding, actual_k)

        results: List[Tuple[PageChunk, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:  # FAISS returns -1 for padding
                results.append((self.chunks[idx], float(score)))

        logger.debug(
            f"Query: '{query[:60]}...' → top scores: "
            f"{[round(s, 3) for _, s in results[:3]]}"
        )
        return results

    def search_multi(
        self,
        queries: List[str],
        top_k: int = TOP_K_CHUNKS,
        deduplicate: bool = True,
    ) -> List[Tuple[PageChunk, float]]:
        """
        Search with multiple queries and merge results.
        Useful for searching with synonymous clause names.
        """
        seen_indices: set[int] = set()
        all_results: List[Tuple[PageChunk, float]] = []

        for query in queries:
            results = self.search(query, top_k=top_k)
            for chunk, score in results:
                if not deduplicate or chunk.chunk_index not in seen_indices:
                    seen_indices.add(chunk.chunk_index)
                    all_results.append((chunk, score))

        # Re-sort by score and return top_k
        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results[:top_k]
