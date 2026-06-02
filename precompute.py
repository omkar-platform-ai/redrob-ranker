#!/usr/bin/env python3
"""Pre-computation step — run ONCE before ranking.

Memory-efficient streaming version: candidates are processed in chunks so
peak RAM stays low (~600 MB total vs ~1.7 GB for the naive approach).

What this does:
  1. Parse the JD with LLM → save parsed_jd.json
  2. Stream candidates.jsonl in chunks of CHUNK_SIZE
  3. Parse + embed each chunk → add vectors to FAISS index
  4. Write parsed candidates to disk line-by-line (no full list in RAM)
  5. Save FAISS index + candidate_ids to data/index/

Pre-computation has NO time or memory constraint per the hackathon spec
(submission_spec Section 3 and 10.3). Only rank.py is constrained.

Usage:
  python precompute.py --candidates data/candidates.jsonl --jd data/job_description.txt
  python precompute.py --candidates data/candidates.jsonl.gz --jd data/job_description.txt

Optional:
  --chunk-size 500   candidates per embedding batch (default 500, lower = less RAM)
  --resume           skip chunks already embedded (for resuming interrupted runs)
"""

import argparse
import gzip
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("precompute")

DEFAULT_CHUNK_SIZE = 500  # ~150 MB peak per chunk; lower if still too heavy


# ── streaming reader ──────────────────────────────────────────────────────────

def stream_jsonl(path: Path):
    """Yield raw dicts one at a time from .jsonl or .jsonl.gz — no full-file RAM."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping malformed line: %s", exc)


def chunked(iterable, size):
    """Yield successive chunks of `size` from an iterable."""
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-compute embeddings and FAISS index")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--jd", required=True)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                        help=f"Candidates per embedding batch (default {DEFAULT_CHUNK_SIZE})")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last completed chunk (skips re-embedding)")
    args = parser.parse_args()

    candidates_path = Path(args.candidates)
    jd_path = Path(args.jd)

    for p in (candidates_path, jd_path):
        if not p.exists():
            logger.error("File not found: %s", p)
            sys.exit(1)

    from src.config import INDEX_DIR, PARSED_JD_PATH, EMBEDDING_DIM
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Parse JD ─────────────────────────────────────────────────────
    logger.info("=== Step 1/4: Parsing JD ===")
    from src.parsers.jd import parse_jd_with_llm, save_parsed_jd

    jd_text = jd_path.read_text(encoding="utf-8")
    parsed_jd = parse_jd_with_llm(jd_text)
    save_parsed_jd(parsed_jd, PARSED_JD_PATH)
    logger.info("JD parsed: %d required skills, %d nice-to-have",
                len(parsed_jd.required_skills), len(parsed_jd.nice_to_have_skills))

    # ── Step 2: Load embedding model ONCE ────────────────────────────────────
    logger.info("=== Step 2/4: Loading embedding model (downloads ~430MB on first run) ===")
    from src.embedder import get_embedder
    embedder = get_embedder()

    # ── Step 3: Stream candidates, embed in chunks ────────────────────────────
    logger.info("=== Step 3/4: Streaming + embedding candidates in chunks of %d ===",
                args.chunk_size)

    import faiss
    import numpy as np

    # Build an empty index we'll add to incrementally
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    all_ids: list[str] = []

    parsed_cache_path = INDEX_DIR / "parsed_candidates.jsonl"
    # Open cache file for writing (or appending if resuming)
    cache_mode = "a" if args.resume else "w"
    already_done = 0

    if args.resume and parsed_cache_path.exists():
        # Count already-processed candidates
        with open(parsed_cache_path) as f:
            already_done = sum(1 for l in f if l.strip())
        logger.info("Resume mode: skipping first %d already-processed candidates", already_done)

    from src.parsers.candidate import parse_redrob_candidate

    t0 = time.perf_counter()
    total_processed = 0
    skipped = 0

    with open(parsed_cache_path, cache_mode, encoding="utf-8") as cache_f:
        for chunk_idx, raw_chunk in enumerate(chunked(stream_jsonl(candidates_path), args.chunk_size)):
            # Resume: skip chunks we already embedded
            chunk_start = chunk_idx * args.chunk_size
            if args.resume and chunk_start + len(raw_chunk) <= already_done:
                skipped += len(raw_chunk)
                continue

            # Parse chunk
            parsed_chunk = [parse_redrob_candidate(r) for r in raw_chunk]

            # Write parsed records to cache (line by line — no big list in RAM)
            for c in parsed_chunk:
                row = {k: v for k, v in c.items() if k != "embedding_text"}
                row["skills_with_meta"] = c.get("skills_with_meta", [])
                cache_f.write(json.dumps(row) + "\n")

            # Embed chunk — larger batch = better MPS/GPU throughput
            texts = [c["embedding_text"] for c in parsed_chunk]
            vecs = embedder.model.encode(
                texts,
                batch_size=min(256, args.chunk_size),
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype("float32")

            # Add to FAISS index
            index.add(vecs)
            all_ids.extend(c["candidate_id"] for c in parsed_chunk)

            # Free chunk memory explicitly
            del parsed_chunk, texts, vecs

            total_processed += len(raw_chunk)
            elapsed = time.perf_counter() - t0
            rate = total_processed / elapsed
            logger.info(
                "Chunk %d done — %d/~100K candidates (%.0f/s, ~%.0f min remaining)",
                chunk_idx + 1,
                total_processed + skipped,
                rate,
                max(0, (100_000 - total_processed - skipped) / max(1, rate) / 60),
            )

    logger.info("Embedded %d candidates total", len(all_ids))

    # ── Step 4: Save FAISS index ──────────────────────────────────────────────
    logger.info("=== Step 4/4: Saving FAISS index ===")
    from src.index import save_index
    save_index(index, all_ids)

    elapsed_total = time.perf_counter() - t0
    logger.info("✅ Pre-computation complete in %.1f min.", elapsed_total / 60)
    logger.info(
        "   Next: python rank.py --candidates %s --jd %s --out submission.csv",
        candidates_path, jd_path,
    )


if __name__ == "__main__":
    main()
