#!/usr/bin/env python3
"""Local submission validator — run before uploading.

Checks every rule from submission_spec Sections 3 and 6.
Usage:
  python validate_submission.py --submission submission.csv --candidates candidates.jsonl
"""

import argparse
import csv
import gzip
import json
import sys
from pathlib import Path


def load_valid_ids(candidates_path: Path) -> set[str]:
    opener = gzip.open if candidates_path.suffix == ".gz" else open
    ids: set[str] = set()
    with opener(candidates_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(json.loads(line)["candidate_id"])
    return ids


def validate(submission_path: Path, candidates_path: Path) -> bool:
    errors: list[str] = []
    warnings: list[str] = []

    # ── Load submission ───────────────────────────────────────────────────────
    with open(submission_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    # Column order
    expected_cols = ["candidate_id", "rank", "score", "reasoning"]
    if list(fieldnames) != expected_cols:
        errors.append(f"Column mismatch. Got {fieldnames}, expected {expected_cols}")

    # Row count
    if len(rows) != 100:
        errors.append(f"Expected 100 rows, got {len(rows)}")

    # Ranks 1-100 each exactly once
    try:
        ranks = [int(r["rank"]) for r in rows]
        if sorted(ranks) != list(range(1, 101)):
            errors.append(f"Ranks must be exactly 1-100, each once. Got: {sorted(set(ranks))[:5]}...")
    except (ValueError, KeyError) as e:
        errors.append(f"Invalid rank values: {e}")

    # Scores non-increasing
    try:
        scores = [float(r["score"]) for r in rows]
        for i in range(1, len(scores)):
            if scores[i] > scores[i - 1] + 1e-9:
                errors.append(
                    f"Score not non-increasing at rank {i+1}: {scores[i-1]:.6f} → {scores[i]:.6f}"
                )
                break
    except (ValueError, KeyError) as e:
        errors.append(f"Invalid score values: {e}")

    # Duplicate candidate_ids
    ids = [r.get("candidate_id", "") for r in rows]
    if len(ids) != len(set(ids)):
        dupes = [x for x in ids if ids.count(x) > 1]
        errors.append(f"Duplicate candidate_ids: {list(set(dupes))[:5]}")

    # candidate_ids exist in dataset
    print(f"Loading valid IDs from {candidates_path}...")
    valid_ids = load_valid_ids(candidates_path)
    invalid = [cid for cid in ids if cid not in valid_ids]
    if invalid:
        errors.append(f"{len(invalid)} candidate_ids not found in dataset: {invalid[:3]}")

    # Reasoning quality
    empty = [r["rank"] for r in rows if not r.get("reasoning", "").strip()]
    if empty:
        errors.append(f"Empty reasoning at ranks: {empty}")

    identical = {}
    for r in rows:
        t = r.get("reasoning", "")
        identical[t] = identical.get(t, 0) + 1
    most_common = max(identical.values()) if identical else 0
    if most_common > 5:
        warnings.append(f"Most common reasoning string repeated {most_common} times — diversify")

    # Score all-same check
    if scores and len(set(scores)) == 1:
        errors.append("All scores are identical — model is not differentiating candidates")

    # ── Report ────────────────────────────────────────────────────────────────
    print()
    if errors:
        print(f"❌ VALIDATION FAILED — {len(errors)} error(s):")
        for e in errors:
            print(f"   • {e}")
    else:
        print("✅ VALIDATION PASSED — submission looks good!")

    if warnings:
        print(f"\n⚠️  {len(warnings)} warning(s):")
        for w in warnings:
            print(f"   • {w}")

    print(f"\nStats: {len(rows)} rows | score range: {min(scores):.4f} – {max(scores):.4f}")
    return len(errors) == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate submission CSV")
    parser.add_argument("--submission", required=True)
    parser.add_argument("--candidates", required=True)
    args = parser.parse_args()

    ok = validate(Path(args.submission), Path(args.candidates))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
