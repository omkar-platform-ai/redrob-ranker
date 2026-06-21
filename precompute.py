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
SAVE_EVERY_CHUNKS = 20    # persist index incrementally so an interrupt is resumable


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


def _resync_cache(cache_path: Path, keep_ids: set[str]) -> set[str]:
    """Rewrite the parsed-candidate cache to hold exactly one row per id in
    `keep_ids`, dropping rows for ids not in the index and any duplicates.

    Used on --resume to reconcile the cache with the loaded index after an
    interrupted run (the cache is flushed before each embed, so it can run
    ahead of the index). Returns the set of candidate_ids kept.
    """
    tmp_path = cache_path.with_suffix(".jsonl.tmp")
    written: set[str] = set()
    with open(cache_path, encoding="utf-8") as fin, \
            open(tmp_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            cid = json.loads(line)["candidate_id"]
            if cid in keep_ids and cid not in written:
                written.add(cid)
                fout.write(line + "\n")
    tmp_path.replace(cache_path)
    return written


def _repair_cache_rows(candidates_path: Path, cache_path: Path,
                       missing_ids: set[str]) -> int:
    """Append cache rows for ids that are in the index but absent from the cache.

    The vectors for these ids already exist in the index, so no re-embedding is
    needed — only their parsed metadata is rebuilt. We stream the raw file and
    re-parse just the missing ids (parser is stdlib + dateutil, cheap). Writes
    the same row shape as the main loop. Returns the number of rows written.
    """
    if not missing_ids:
        return 0

    from src.parsers.candidate import parse_redrob_candidate

    written: set[str] = set()
    with open(cache_path, "a", encoding="utf-8") as cache_f:
        for r in stream_jsonl(candidates_path):
            cid = str(r.get("candidate_id", ""))
            if cid not in missing_ids or cid in written:
                continue
            c = parse_redrob_candidate(r)
            row = {k: v for k, v in c.items() if k != "embedding_text"}
            row["skills_with_meta"] = c.get("skills_with_meta", [])
            cache_f.write(json.dumps(row) + "\n")
            written.add(cid)
            if len(written) == len(missing_ids):
                break
    return len(written)


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

    from src.config import INDEX_DIR, PARSED_JD_PATH, EMBEDDING_DIM, FAISS_INDEX_PATH
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
    embedder = get_embedder(device=None)  # auto-select MPS/CUDA — pre-compute is unconstrained

    # ── Step 3: Stream candidates, embed in chunks ────────────────────────────
    logger.info("=== Step 3/4: Streaming + embedding candidates in chunks of %d ===",
                args.chunk_size)

    import faiss

    from src.index import load_index, save_index

    parsed_cache_path = INDEX_DIR / "parsed_candidates.jsonl"

    # Resume reuses the existing index/ids as the source of truth; otherwise we
    # start fresh. `seen_ids` drives dedup below — seeding it from the loaded
    # index is what makes resume skip only already-EMBEDDED candidates.
    if args.resume and FAISS_INDEX_PATH.exists() and parsed_cache_path.exists():
        index, all_ids = load_index()
        seen_ids: set[str] = set(all_ids)
        logger.info("Resume mode: loaded existing index with %d vectors", index.ntotal)
        # The cache is flushed before each embed, so it may run ahead of the index
        # after an interrupt. Reconcile it to the index (drops stray/dup rows).
        cached_ids = _resync_cache(parsed_cache_path, seen_ids)
        logger.info("Resume mode: cache re-synced to %d rows matching the index",
                    len(cached_ids))
        # Conversely, the index may carry ids with no cache row (e.g. an old index
        # segment reused on resume). Rebuild those rows from raw — no re-embedding,
        # the vectors are already in the index.
        missing = seen_ids - cached_ids
        if missing:
            repaired = _repair_cache_rows(candidates_path, parsed_cache_path, missing)
            logger.info("Resume mode: repaired %d cache rows for indexed-but-uncached ids",
                        repaired)
            if repaired != len(missing):
                logger.warning(
                    "Resume mode: %d indexed ids not found in raw file — cache still short",
                    len(missing) - repaired,
                )
        cache_mode = "a"
    else:
        index = faiss.IndexFlatIP(EMBEDDING_DIM)
        all_ids = []
        seen_ids = set()
        cache_mode = "w"

    from src.parsers.candidate import parse_redrob_candidate

    t0 = time.perf_counter()
    total_seen = 0
    skipped = 0

    with open(parsed_cache_path, cache_mode, encoding="utf-8") as cache_f:
        for chunk_idx, raw_chunk in enumerate(chunked(stream_jsonl(candidates_path), args.chunk_size)):
            total_seen += len(raw_chunk)

            # Parse + dedup by candidate_id. Skips ids already embedded (resume)
            # AND duplicate ids within the raw dataset — each id is embedded once.
            parsed_chunk = []
            for r in raw_chunk:
                c = parse_redrob_candidate(r)
                cid = c["candidate_id"]
                if cid in seen_ids:
                    skipped += 1
                    continue
                seen_ids.add(cid)
                parsed_chunk.append(c)

            if not parsed_chunk:
                continue

            # Write parsed records to cache (line by line — no big list in RAM),
            # flushing BEFORE the embed so an interrupt leaves cache >= index
            # (the next --resume reconciles via _resync_cache).
            for c in parsed_chunk:
                row = {k: v for k, v in c.items() if k != "embedding_text"}
                row["skills_with_meta"] = c.get("skills_with_meta", [])
                cache_f.write(json.dumps(row) + "\n")
            cache_f.flush()

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

            # Persist incrementally so an interrupted run is cleanly resumable.
            if (chunk_idx + 1) % SAVE_EVERY_CHUNKS == 0:
                save_index(index, all_ids)

            elapsed = time.perf_counter() - t0
            rate = len(all_ids) / elapsed if elapsed else 0.0
            logger.info(
                "Chunk %d — embedded=%d skipped=%d seen=%d (%.0f/s)",
                chunk_idx + 1, len(all_ids), skipped, total_seen, rate,
            )

    logger.info("Embedded %d candidates total (skipped %d duplicates/already-done)",
                len(all_ids), skipped)

    # ── Step 4: Save FAISS index ──────────────────────────────────────────────
    logger.info("=== Step 4/4: Saving FAISS index ===")
    # Invariant: one vector per unique candidate_id, index and id-list in lockstep.
    assert index.ntotal == len(all_ids) == len(set(all_ids)), (
        f"Coverage invariant broken: ntotal={index.ntotal}, "
        f"ids={len(all_ids)}, unique={len(set(all_ids))}"
    )
    # Parity: every indexed id must have a cache row (rank.py scores from the cache).
    # Guards against silent desync — e.g. an old index segment reused on resume.
    cache_ids: set[str] = set()
    with open(parsed_cache_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cache_ids.add(json.loads(line)["candidate_id"])
    index_ids = set(all_ids)
    assert cache_ids == index_ids, (
        f"Cache↔index desync: in_index_not_cache={len(index_ids - cache_ids)}, "
        f"in_cache_not_index={len(cache_ids - index_ids)}"
    )
    save_index(index, all_ids)
    if total_seen != len(all_ids):
        logger.warning(
            "Embedded %d unique of %d raw records seen (%d duplicates skipped).",
            len(all_ids), total_seen, total_seen - len(all_ids),
        )

    elapsed_total = time.perf_counter() - t0
    logger.info("✅ Pre-computation complete in %.1f min.", elapsed_total / 60)
    logger.info(
        "   Next: python rank.py --candidates %s --jd %s --out submission.csv",
        candidates_path, jd_path,
    )


if __name__ == "__main__":
    main()
