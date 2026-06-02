"""FAISS Index — build and query the candidate ANN index.

Uses IndexFlatIP (inner product on L2-normalised vectors = cosine similarity).
For 100K vectors × 768 dims this is ~295MB in RAM — well within the 16GB budget.

If the dataset grows to 1M+, swap to IndexHNSWFlat for sub-linear query time.
Current design: exact search, ~200ms query on 100K on a single CPU core.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import faiss
import numpy as np

from src.config import EMBEDDING_DIM, FAISS_INDEX_PATH, CANDIDATE_IDS_PATH

logger = logging.getLogger(__name__)


def build_index(embeddings: np.ndarray, candidate_ids: list[str]) -> faiss.Index:
    """Build a flat inner-product FAISS index from pre-computed embeddings."""
    assert embeddings.shape[1] == EMBEDDING_DIM, (
        f"Expected {EMBEDDING_DIM}-dim embeddings, got {embeddings.shape[1]}"
    )
    embeddings = embeddings.astype(np.float32)

    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings)
    logger.info("Built FAISS index with %d vectors", index.ntotal)
    return index


def save_index(index: faiss.Index, candidate_ids: list[str]) -> None:
    FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    with open(CANDIDATE_IDS_PATH, "w") as f:
        json.dump(candidate_ids, f)
    logger.info(
        "Saved FAISS index → %s  |  IDs → %s",
        FAISS_INDEX_PATH, CANDIDATE_IDS_PATH,
    )


def load_index() -> tuple[faiss.Index, list[str]]:
    if not FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {FAISS_INDEX_PATH}. "
            "Run: python rank.py --precompute --candidates <path>"
        )
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    with open(CANDIDATE_IDS_PATH) as f:
        candidate_ids = json.load(f)
    logger.info("Loaded FAISS index (%d vectors)", index.ntotal)
    return index, candidate_ids


def query_index(
    index: faiss.Index,
    candidate_ids: list[str],
    jd_vector: np.ndarray,
    top_k: int,
) -> list[tuple[str, float]]:
    """ANN search. Returns [(candidate_id, cosine_similarity), ...]."""
    jd_vec = jd_vector.astype(np.float32).reshape(1, -1)
    distances, indices = index.search(jd_vec, top_k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(candidate_ids):
            continue
        results.append((candidate_ids[idx], float(dist)))
    return results
