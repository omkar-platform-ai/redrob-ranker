"""Scorer tests — every scorer must return a value in [0, 1] and behave
directionally as documented. These run at rank time, so they must stay
dependency-light and deterministic.
"""

from __future__ import annotations

from tests.conftest import make_candidate

from src.scorers.behavioral import BehavioralScorer
from src.scorers.career import CareerScorer
from src.scorers.role_fit import RoleFitScorer
from src.scorers.skill import SkillScorer


# ── role_fit ────────────────────────────────────────────────────────────────

def test_role_fit_in_bounds(parsed_candidate, parsed_jd):
    s = RoleFitScorer().score(parsed_candidate, parsed_jd)
    assert 0.0 <= s <= 1.0


def test_role_fit_disqualifying_title_penalised(parsed_candidate, parsed_jd):
    good = RoleFitScorer().score(parsed_candidate, parsed_jd)
    hr = RoleFitScorer().score(
        make_candidate(current_title="HR Manager"), parsed_jd
    )
    assert hr < good


def test_role_fit_consulting_only_penalised(parsed_candidate, parsed_jd):
    good = RoleFitScorer().score(parsed_candidate, parsed_jd)
    consulting = RoleFitScorer().score(
        make_candidate(all_consulting=True), parsed_jd
    )
    assert consulting < good


# ── skill ───────────────────────────────────────────────────────────────────

def test_skill_matches_required(parsed_candidate, parsed_jd):
    score, matched = SkillScorer().score(
        parsed_candidate["skills_with_meta"],
        parsed_jd.required_skills,
        parsed_jd.nice_to_have_skills,
    )
    assert 0.0 <= score <= 1.0
    assert "Python" in matched


def test_skill_neutral_when_no_jd_skills():
    score, matched = SkillScorer().score([], [], [])
    assert score == 0.50
    assert matched == []


# ── behavioral ──────────────────────────────────────────────────────────────

def test_behavioral_in_bounds(parsed_candidate):
    s = BehavioralScorer().score(parsed_candidate["behavioral_signals"])
    assert 0.0 <= s <= 1.0


def test_behavioral_recency_decay():
    scorer = BehavioralScorer()
    fresh = scorer.score({"last_active_days": 0, "notice_period_days": 30})
    stale = scorer.score({"last_active_days": 365, "notice_period_days": 30})
    assert fresh > stale


# ── career ──────────────────────────────────────────────────────────────────

def test_career_in_bounds(parsed_candidate):
    total, bonus, _reasons = CareerScorer().score(parsed_candidate)
    assert 0.0 <= total <= 1.0
    assert 0.0 <= bonus <= 0.15


def test_career_hidden_gem_bonus(parsed_candidate):
    _, bonus, reasons = CareerScorer().score(parsed_candidate)
    # fixture has open-source + 2 promotions + high interview completion
    assert bonus > 0.0
    assert "open_source" in reasons
