"""FAISS index tests — build/save/load round-trip and the row↔ID alignment that
the whole ranking depends on. Closes the previously-untested src/index.py.

Deterministic: uses one-hot unit vectors so the nearest neighbour is unambiguous
(self cosine = 1.0, pairwise = 0.0). No model, no network.
"""

from __future__ import annotations

import os

# faiss and torch each vendor their own libomp; on macOS dev boxes the duplicate
# OpenMP runtime aborts on the first faiss.search (OMP Error #15). This is the
# documented workaround and a harmless no-op on Linux/Docker (single libomp).
# Must be set before src.index (→ faiss) is imported below.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import src.index as index_mod  # noqa: E402
from src.config import EMBEDDING_DIM  # noqa: E402
from src.index import build_index, load_index, query_index, save_index  # noqa: E402


def _onehot(n: int) -> np.ndarray:
    """n unit vectors where vector i = e_i (1.0 at position i, else 0)."""
    m = np.zeros((n, EMBEDDING_DIM), dtype=np.float32)
    for i in range(n):
        m[i, i] = 1.0
    return m


@pytest.fixture
def index_paths(tmp_path, monkeypatch):
    """Redirect the module-level index paths into a tmp dir."""
    faiss_path = tmp_path / "candidates.faiss"
    ids_path = tmp_path / "candidate_ids.json"
    monkeypatch.setattr(index_mod, "FAISS_INDEX_PATH", faiss_path)
    monkeypatch.setattr(index_mod, "CANDIDATE_IDS_PATH", ids_path)
    return faiss_path, ids_path


def test_roundtrip_preserves_count_and_ids(index_paths):
    ids = [f"CAND_{i}" for i in range(5)]
    save_index(build_index(_onehot(5), ids), ids)

    loaded_index, loaded_ids = load_index()
    assert loaded_index.ntotal == 5
    assert loaded_ids == ids


def test_query_returns_matching_id_first(index_paths):
    ids = [f"CAND_{i}" for i in range(5)]
    vecs = _onehot(5)
    save_index(build_index(vecs, ids), ids)
    loaded_index, loaded_ids = load_index()

    # Query with row 2's exact vector → CAND_2 must rank first at cosine ~1.0.
    results = query_index(loaded_index, loaded_ids, vecs[2], top_k=5)
    assert results[0][0] == "CAND_2"
    assert results[0][1] == pytest.approx(1.0, abs=1e-5)


def test_query_results_descending(index_paths):
    ids = [f"CAND_{i}" for i in range(5)]
    vecs = _onehot(5)
    save_index(build_index(vecs, ids), ids)
    loaded_index, loaded_ids = load_index()

    # Most mass on row 0, some on row 1 → CAND_0 first, scores non-increasing.
    q = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    q[0], q[1] = 0.9, 0.4
    results = query_index(loaded_index, loaded_ids, q, top_k=5)

    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)
    assert results[0][0] == "CAND_0"


def test_query_topk_exceeding_n_drops_invalid_rows(index_paths):
    ids = [f"CAND_{i}" for i in range(3)]
    vecs = _onehot(3)
    save_index(build_index(vecs, ids), ids)
    loaded_index, loaded_ids = load_index()

    # top_k far larger than the 3 rows: FAISS pads indices with -1. query_index's
    # idx<0 guard must drop those — never raise, never emit a bad id.
    results = query_index(loaded_index, loaded_ids, vecs[0], top_k=10)
    assert len(results) == 3
    assert all(cid in ids for cid, _ in results)
