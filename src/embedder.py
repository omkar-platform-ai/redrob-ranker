"""Embedder — wraps SentenceTransformer for batch encoding.

Singleton pattern: the model loads once at process start, cached via
module-level variable. Subsequent calls to get_embedder() return the same
instance (no repeated 430MB downloads).
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBED_BATCH_SIZE, EMBEDDING_MODEL

logger = logging.getLogger(__name__)
_embedder_instance: "Embedder | None" = None


class Embedder:
    def __init__(self, device: str | None = "cpu") -> None:
        logger.info(
            "Loading embedding model: %s (device=%s, first run downloads ~430MB)",
            EMBEDDING_MODEL, device or "auto",
        )
        self.model = SentenceTransformer(EMBEDDING_MODEL, device=device)
        logger.info("Model loaded on device=%s", self.model.device)

    def embed_text(self, text: str) -> np.ndarray:
        return self._embed_cached(text)

    @lru_cache(maxsize=8)
    def _embed_cached(self, text: str) -> np.ndarray:
        vec = self.model.encode(text, normalize_embeddings=True)
        vec.setflags(write=False)
        return vec

    def embed_batch(self, texts: list[str], batch_size: int = EMBED_BATCH_SIZE) -> np.ndarray:
        """Encode a list of texts; returns float32 matrix (N × DIM), L2-normalised."""
        logger.info("Embedding %d texts in batches of %d...", len(texts), batch_size)
        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )


def get_embedder(device: str | None = "cpu") -> Embedder:
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = Embedder(device=device)
    return _embedder_instance
