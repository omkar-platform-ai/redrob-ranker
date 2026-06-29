"""End-to-end: rank → CSV → validate_submission, on synthetic candidates.

Exercises the "success-criteria" artifacts the unit suite otherwise never touches:
the output CSV (RankingEngine.to_csv), the spec validator
(validate_submission.validate), and rank._sanity_check. Model-free — no FAISS, no
embedder; rank() takes ANN results directly.
"""

from __future__ import annotations

import json

import pytest

import validate_submission
from rank import _sanity_check
from src.ranker import RankingEngine
from tests.conftest import make_candidate


def _build_100(with_honeypots: int = 3):
    """100 candidates with strictly-descending semantic scores; a few honeypots."""
    candidates: dict[str, dict] = {}
    ann: list[tuple[str, float]] = []
    for i in range(100):
        cid = f"CAND_{i:05d}"
        candidates[cid] = make_candidate(candidate_id=cid)
        ann.append((cid, 0.95 - i * 0.005))
    # Impossible profiles: 15yr claimed over 12 career-months + expert skill used
    # 1 month → trips 2 honeypot signals.
    for i in range(with_honeypots):
        cid = f"CAND_{i:05d}"
        candidates[cid] = make_candidate(
            candidate_id=cid,
            years_of_experience=15.0,
            total_career_months=12,
            skills_with_meta=[
                {"name": "PyTorch", "proficiency": "expert", "endorsements": 0, "duration_months": 1},
            ],
        )
    return candidates, ann


def _write_candidates_jsonl(path, ids) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for cid in ids:
            f.write(json.dumps({"candidate_id": cid}) + "\n")


def test_rank_csv_passes_validator(parsed_jd, tmp_path):
    candidates, ann = _build_100()
    ranked = RankingEngine().rank(candidates, ann, parsed_jd)
    assert len(ranked) == 100

    csv_path = tmp_path / "submission.csv"
    RankingEngine.to_csv(ranked, str(csv_path))

    cand_path = tmp_path / "candidates.jsonl"
    _write_candidates_jsonl(cand_path, list(candidates.keys()))

    assert validate_submission.validate(csv_path, cand_path) is True


def test_output_honeypot_rate_under_10pct(parsed_jd):
    candidates, ann = _build_100(with_honeypots=3)
    ranked = RankingEngine().rank(candidates, ann, parsed_jd)
    rate = sum(1 for sc in ranked if sc.is_honeypot) / len(ranked)
    assert rate < 0.10


def test_sanity_check_accepts_good_csv(parsed_jd, tmp_path):
    candidates, ann = _build_100()
    ranked = RankingEngine().rank(candidates, ann, parsed_jd)
    csv_path = tmp_path / "submission.csv"
    RankingEngine.to_csv(ranked, str(csv_path))
    _sanity_check(csv_path)  # must not raise / sys.exit


def test_sanity_check_rejects_bad_csv(tmp_path):
    bad = tmp_path / "bad.csv"
    # Only 2 rows AND score increases → violates row-count and non-increasing.
    bad.write_text(
        "candidate_id,rank,score,reasoning\n"
        "A,1,0.5,ok\n"
        "B,2,0.9,better\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit):
        _sanity_check(bad)
