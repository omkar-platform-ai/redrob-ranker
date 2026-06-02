#!/usr/bin/env python3
"""Ranking step — MUST complete in < 5 minutes on 16GB CPU.

Constraints (submission_spec Section 3):
  ✗ No hosted LLM API calls
  ✗ No GPU usage
  ✗ No network access
  ✓ Pure CPU math on pre-built FAISS index

Usage:
  python rank.py --candidates candidates.jsonl --jd job_description.txt --out submission.csv

Pre-requisite:
  python precompute.py --candidates candidates.jsonl --jd job_description.txt

Output:
  CSV with exactly 100 rows, columns: candidate_id,rank,score,reasoning
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("rank")

_WALL_BUDGET_SECONDS = 270  # warn at 4m30s; hard budget is 5m


def main() -> None:
    t_start = time.perf_counter()

    parser = argparse.ArgumentParser(description="Rank candidates — CPU only, no LLM")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl[.gz]")
    parser.add_argument("--jd", required=True, help="Path to job_description.txt")
    parser.add_argument("--out", default="submission.csv", help="Output CSV path")
    args = parser.parse_args()

    candidates_path = Path(args.candidates)
    jd_path = Path(args.jd)
    out_path = Path(args.out)

    # ── Load pre-computed index ───────────────────────────────────────────────
    logger.info("Loading FAISS index...")
    from src.index import load_index, query_index
    from src.config import PARSED_JD_PATH, INDEX_DIR, TOP_K_RETRIEVE

    index, candidate_ids = load_index()

    # ── Load parsed JD ────────────────────────────────────────────────────────
    from src.parsers.jd import load_parsed_jd
    parsed_jd = load_parsed_jd(PARSED_JD_PATH)
    logger.info(
        "JD: '%s' | required=%d | nice=%d",
        parsed_jd.title, len(parsed_jd.required_skills), len(parsed_jd.nice_to_have_skills),
    )

    # ── Embed JD (no LLM — just the sentence-transformer model) ──────────────
    logger.info("Embedding JD...")
    from src.embedder import get_embedder
    embedder = get_embedder()
    jd_vec = embedder.embed_text(parsed_jd.to_embedding_text())

    # ── ANN search ────────────────────────────────────────────────────────────
    logger.info("ANN search: top-%d candidates...", TOP_K_RETRIEVE)
    ann_results = query_index(index, candidate_ids, jd_vec, TOP_K_RETRIEVE)
    logger.info("ANN search returned %d candidates", len(ann_results))

    # ── Load parsed candidates (from pre-computed cache) ─────────────────────
    parsed_cache = INDEX_DIR / "parsed_candidates.jsonl"
    if not parsed_cache.exists():
        logger.error(
            "Parsed candidate cache not found at %s. Run precompute.py first.", parsed_cache
        )
        sys.exit(1)

    ann_ids = {cid for cid, _ in ann_results}
    candidates_by_id: dict[str, dict] = {}
    with open(parsed_cache, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            if c["candidate_id"] in ann_ids:
                candidates_by_id[c["candidate_id"]] = c

    logger.info("Loaded %d candidates from cache for scoring", len(candidates_by_id))

    # ── Rank ──────────────────────────────────────────────────────────────────
    logger.info("Running multi-signal ranking...")
    from src.ranker import RankingEngine
    engine = RankingEngine()
    ranked = engine.rank(candidates_by_id, ann_results, parsed_jd)

    # ── Write submission CSV ──────────────────────────────────────────────────
    RankingEngine.to_csv(ranked, str(out_path))
    logger.info("✅ Submission written → %s (%d rows)", out_path, len(ranked))

    elapsed = time.perf_counter() - t_start
    logger.info("Total elapsed: %.1fs", elapsed)

    if elapsed > _WALL_BUDGET_SECONDS:
        logger.warning(
            "⚠️  Elapsed %.1fs exceeds 4m30s warning threshold. "
            "Verify this runs within 5 min on a fresh 16GB CPU machine.",
            elapsed,
        )

    # Quick sanity checks
    _sanity_check(out_path)


def _sanity_check(path: Path) -> None:
    """Catch common submission errors before upload."""
    import csv
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    errors: list[str] = []
    if len(rows) != 100:
        errors.append(f"Expected 100 rows, got {len(rows)}")

    ranks = [int(r["rank"]) for r in rows]
    if sorted(ranks) != list(range(1, 101)):
        errors.append("Ranks are not exactly 1-100 each appearing once")

    scores = [float(r["score"]) for r in rows]
    for i in range(1, len(scores)):
        if scores[i] > scores[i - 1] + 1e-9:
            errors.append(f"Score not non-increasing at rank {i+1}: {scores[i-1]:.4f} → {scores[i]:.4f}")
            break

    ids = [r["candidate_id"] for r in rows]
    if len(ids) != len(set(ids)):
        errors.append("Duplicate candidate_ids in submission")

    empty_reasoning = sum(1 for r in rows if not r.get("reasoning", "").strip())
    if empty_reasoning > 0:
        errors.append(f"{empty_reasoning} rows have empty reasoning")

    if errors:
        logger.error("Sanity check FAILED:")
        for e in errors:
            logger.error("  • %s", e)
        sys.exit(1)
    else:
        logger.info("✅ Sanity checks passed (100 rows, non-increasing scores, no empty reasoning)")


if __name__ == "__main__":
    main()
