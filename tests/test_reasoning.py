"""Reasoning generator tests — reasoning must be non-empty, grounded in the
candidate's actual data, and not hallucinate skills the candidate lacks.
"""

from __future__ import annotations

from tests.conftest import make_candidate

from src.reasoning import ReasoningGenerator, ScoredCandidate


def _scored(candidate: dict, **overrides) -> ScoredCandidate:
    base = dict(
        candidate_id=candidate["candidate_id"],
        rank=1,
        score=0.8,
        semantic_score=0.8,
        role_fit_score=0.7,
        skill_score=0.7,
        behavioral_score=0.7,
        career_score=0.6,
        matched_skills=["Python", "Embeddings"],
        hidden_gem_reasons=["open_source"],
        is_honeypot=False,
        current_title=candidate["current_title"],
        years_of_experience=candidate["years_of_experience"],
        current_company=candidate["current_company"],
        current_industry=candidate["current_industry"],
        all_consulting=candidate["all_consulting"],
        notice_period_days=candidate["behavioral_signals"]["notice_period_days"],
        location=candidate["location"],
        country=candidate["country"],
    )
    base.update(overrides)
    return ScoredCandidate(**base)


def test_reasoning_non_empty(parsed_candidate, parsed_jd):
    sc = _scored(parsed_candidate)
    text = ReasoningGenerator().generate(parsed_candidate, sc, parsed_jd)
    assert text.strip()


def test_reasoning_mentions_matched_skill(parsed_candidate, parsed_jd):
    sc = _scored(parsed_candidate)
    text = ReasoningGenerator().generate(parsed_candidate, sc, parsed_jd)
    assert "Python" in text


def test_reasoning_capped_length(parsed_candidate, parsed_jd):
    sc = _scored(parsed_candidate)
    text = ReasoningGenerator().generate(parsed_candidate, sc, parsed_jd)
    assert len(text) <= 280


def test_reasoning_flags_consulting_concern(parsed_jd):
    c = make_candidate(all_consulting=True)
    sc = _scored(c, all_consulting=True, matched_skills=[])
    text = ReasoningGenerator().generate(c, sc, parsed_jd)
    assert "consulting" in text.lower()
