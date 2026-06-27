"""Pre-build the FAISS index for the baked-in sample candidates.

Run once at Docker IMAGE BUILD time (CPU is fast here, no constraints) so the
live demo does not have to embed 100 candidates on the HF free-tier CPU — that
takes ~50-74s per run there and makes the demo look frozen. With a pre-built
index on disk, the runtime demo only embeds the JD text (~seconds), mirroring
how production ``rank.py`` works (load a pre-built index + parsed cache).

Outputs to ``sample_index/`` (next to the app, baked into the image):
  - candidates.faiss        (IndexFlatIP, 100 × 768 float32)
  - candidate_ids.json      (id list, index-aligned)
  - parsed_candidates.jsonl (parsed candidate dicts, one per line)

Local use (Option A in the README): ``python scripts/build_sample_index.py``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))  # so `from src...` works when run directly

# Import torch before faiss: on macOS, initialising faiss's OpenMP runtime first
# segfaults on the first CPU parallel region (the encode). No-op on Linux.
import torch  # noqa: F401  -- MUST precede `import faiss`
import faiss

from src.parsers.candidate import parse_redrob_candidate
from src.embedder import get_embedder
from src.index import build_index

SAMPLE_JSON = REPO / "data" / "sample_candidates.json"
OUT_DIR = REPO / "sample_index"


def main() -> None:
    raw = SAMPLE_JSON.read_text()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"No candidates found in {SAMPLE_JSON}")

    parsed = [parse_redrob_candidate(r) for r in rows]
    texts = [c["embedding_text"] for c in parsed]
    ids = [c["candidate_id"] for c in parsed]

    print(f"Embedding {len(texts)} sample candidates (build-time, CPU)...")
    embeddings = get_embedder().embed_batch(texts)

    index = build_index(embeddings, ids)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(OUT_DIR / "candidates.faiss"))
    (OUT_DIR / "candidate_ids.json").write_text(json.dumps(ids))
    with open(OUT_DIR / "parsed_candidates.jsonl", "w", encoding="utf-8") as f:
        for c in parsed:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(
        f"Built sample index: {index.ntotal} vectors x {index.d} dims -> {OUT_DIR} "
        f"(candidates.faiss, candidate_ids.json, parsed_candidates.jsonl)"
    )


if __name__ == "__main__":
    main()
