"""Scorer tests — every scorer must return a value in [0, 1] and behave
directionally as documented. These run at rank time, so they must stay
dependency-light and deterministic.
"""

from __future__ import annotations

from tests.conftest import make_candidate

from src.parsers.jd import ParsedJD
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


# ── role_fit: title scoring (multi-word AI/ML titles) ────────────────────────

def test_title_score_multiword_ai_title_full_credit():
    """Multi-word AI/ML titles the exact set misses still earn full credit
    (role word + in-scope ML token), e.g. 'Recommendation Systems Engineer'."""
    scorer = RoleFitScorer()
    assert scorer._title_score("Recommendation Systems Engineer") == 1.0
    assert scorer._title_score("Search Ranking Engineer") == 1.0
    assert scorer._title_score("Retrieval Engineer") == 1.0


def test_title_score_exact_set_unchanged():
    scorer = RoleFitScorer()
    assert scorer._title_score("Senior ML Engineer") == 1.0
    assert scorer._title_score("Software Engineer") == 1.0


def test_title_score_non_ml_engineers_not_elevated():
    """Engineering titles with no ML token keep generic credit, not 1.0."""
    scorer = RoleFitScorer()
    assert scorer._title_score("Mechanical Engineer") == 0.5
    assert scorer._title_score("Frontend Engineer") == 0.5
    assert scorer._title_score("Data Analyst") == 0.3
    assert scorer._title_score("Marketing Manager") == 0.0


def test_title_contains_word_is_boundary_matched():
    """The ML-token check must not fire on substrings: 'search' inside
    'research', 'ml' inside 'html'."""
    scorer = RoleFitScorer()
    assert scorer._title_contains_word("research analyst", ["search"]) is False
    assert scorer._title_contains_word("html developer", ["ml"]) is False
    assert scorer._title_contains_word("recommendation engineer", ["recommendation"]) is True


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


# ── behavioral: graded conversion component ─────────────────────────────────

def test_behavioral_conversion_graded():
    """offer/github/interview now grade continuously: a strong converter
    outscores one carrying only sentinel (-1) data, all else equal."""
    scorer = BehavioralScorer()
    base = {"last_active_days": 0, "notice_period_days": 30, "recruiter_response_rate": 0.5}
    weak = dict(base)  # offer/github missing → neutral; interview defaults to 0
    strong = dict(base, offer_acceptance_rate=0.9,
                  interview_completion_rate=1.0, github_activity_score=55)
    assert scorer.score(strong) > scorer.score(weak)


def test_behavioral_missing_signal_is_neutral():
    """A -1 sentinel (no GitHub / no prior offers) maps to neutral, not 0 —
    absence is never punished, only presence rewarded."""
    scorer = BehavioralScorer()
    base = {"last_active_days": 0, "notice_period_days": 30}
    missing = dict(base, github_activity_score=-1, offer_acceptance_rate=-1)
    low = dict(base, github_activity_score=0, offer_acceptance_rate=0.0)
    assert scorer.score(missing) > scorer.score(low)


# ── role_fit: JD-stated disqualifiers ────────────────────────────────────────

def test_role_fit_jd_disqualifier_title_chasing():
    """A JD that disqualifies title-chasing penalises short-tenure candidates."""
    hopper = make_candidate(avg_tenure_months=12)
    with_dq = RoleFitScorer().score(
        hopper, ParsedJD(disqualifiers=["title-chasing every 1.5 years"]))
    without_dq = RoleFitScorer().score(hopper, ParsedJD(disqualifiers=[]))
    assert with_dq < without_dq


def test_role_fit_jd_disqualifier_cv_speech():
    """A pure CV/speech title is penalised when the JD disqualifies that domain."""
    cv = make_candidate(current_title="Computer Vision Engineer")
    with_dq = RoleFitScorer().score(
        cv, ParsedJD(disqualifiers=["cv/speech/robotics primary"]))
    without_dq = RoleFitScorer().score(cv, ParsedJD(disqualifiers=[]))
    assert with_dq < without_dq
