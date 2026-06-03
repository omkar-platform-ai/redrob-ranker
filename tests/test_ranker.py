"""Ranking engine tests — fusion + output invariants the spec enforces.
No FAISS needed: rank() accepts ANN results directly.
"""

from __future__ import annotations

import csv

from tests.conftest import make_candidate

from src.ranker import RankingEngine


def _setup(n=5):
    candidates = {f"C{i}": make_candidate(candidate_id=f"C{i}") for i in range(n)}
    # Descending semantic scores so order is predictable.
    ann = [(f"C{i}", 0.9 - 0.1 * i) for i in range(n)]
    return candidates, ann


def test_rank_outputs_sequential_ranks(parsed_jd):
    candidates, ann = _setup(5)
    ranked = RankingEngine().rank(candidates, ann, parsed_jd)
    assert [sc.rank for sc in ranked] == [1, 2, 3, 4, 5]


def test_rank_scores_non_increasing(parsed_jd):
    candidates, ann = _setup(5)
    ranked = RankingEngine().rank(candidates, ann, parsed_jd)
    scores = [sc.score for sc in ranked]
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))


def test_rank_reasoning_non_empty(parsed_jd):
    candidates, ann = _setup(3)
    ranked = RankingEngine().rank(candidates, ann, parsed_jd)
    assert all(sc.reasoning.strip() for sc in ranked)


def test_honeypot_pushed_to_bottom(parsed_jd):
    candidates, ann = _setup(4)
    # Make the top-semantic candidate a honeypot (2 signals).
    candidates["C0"] = make_candidate(
        candidate_id="C0",
        years_of_experience=12.0,
        total_career_months=24,
        skills_with_meta=[
            {"name": "PyTorch", "proficiency": "expert", "endorsements": 0, "duration_months": 1},
        ],
    )
    ranked = RankingEngine().rank(candidates, ann, parsed_jd)
    honeypot = next(sc for sc in ranked if sc.candidate_id == "C0")
    assert honeypot.is_honeypot is True
    assert honeypot.score == 0.0
    assert honeypot.rank == len(ranked)  # zero score → last


def test_to_csv_format(parsed_jd, tmp_path):
    candidates, ann = _setup(3)
    ranked = RankingEngine().rank(candidates, ann, parsed_jd)
    out = tmp_path / "submission.csv"
    RankingEngine.to_csv(ranked, str(out))

    with open(out, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [*rows[0].keys()] == ["candidate_id", "rank", "score", "reasoning"]
    assert len(rows) == 3
    assert [int(r["rank"]) for r in rows] == [1, 2, 3]
