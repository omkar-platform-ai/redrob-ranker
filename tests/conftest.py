"""Shared fixtures.

These build records in the *internal parsed* schema (the dict produced by
`parse_redrob_candidate` and consumed by the scorers / ranker), plus one raw
redrob record for parser tests.
"""

from __future__ import annotations

import pytest

from src.parsers.jd import ParsedJD


@pytest.fixture
def parsed_jd() -> ParsedJD:
    """A typical senior-AI-engineer JD."""
    return ParsedJD(
        title="Senior AI Engineer",
        required_skills=["python", "embeddings", "retrieval"],
        nice_to_have_skills=["faiss", "learning to rank"],
        min_experience_years=5,
        max_experience_years=9,
        seniority_level="senior",
        domain="ml/ai retrieval and ranking",
        location_preferences=["Pune"],
        industry_preferences=["product companies"],
        disqualifiers=["consulting-only"],
        raw_summary="Senior AI Engineer for embeddings, retrieval, and ranking.",
    )


def make_candidate(**overrides) -> dict:
    """Build a parsed-candidate dict with sensible, internally consistent defaults.

    Override any field via kwargs to construct edge cases.
    """
    candidate = {
        "candidate_id": "C1",
        "name": "Anon",
        "headline": "AI Engineer",
        "summary": "Builds retrieval and ranking systems.",
        "current_title": "Senior ML Engineer",
        "current_company": "ProductCo",
        "current_company_size": "201-500",
        "current_industry": "technology",
        "location": "Pune",
        "country": "India",
        "years_of_experience": 6.0,
        "seniority_level": "senior",
        "skills": ["Python", "Embeddings", "Retrieval"],
        "skills_with_meta": [
            {"name": "Python", "proficiency": "expert", "endorsements": 40, "duration_months": 60},
            {"name": "Embeddings", "proficiency": "advanced", "endorsements": 10, "duration_months": 30},
            {"name": "Retrieval", "proficiency": "advanced", "endorsements": 5, "duration_months": 24},
        ],
        "work_experience": [],
        "avg_tenure_months": 24.0,
        "distinct_title_count": 3,
        "promotions_count": 2,
        "total_career_months": 72,
        "product_company_months": 60,
        "services_company_months": 0,
        "all_consulting": False,
        "has_open_source_contributions": True,
        "github_activity_score": 45.0,
        "education": [],
        "behavioral_signals": {
            "last_active_days": 5,
            "open_to_work": True,
            "recruiter_response_rate": 0.7,
            "applications_count": 4,
            "profile_views_last_30d": 50,
            "saved_by_recruiters_30d": 3,
            "notice_period_days": 30,
            "interview_completion_rate": 0.9,
            "willing_to_relocate": False,
        },
    }
    candidate.update(overrides)
    return candidate


@pytest.fixture
def parsed_candidate() -> dict:
    return make_candidate()


@pytest.fixture
def raw_candidate() -> dict:
    """A raw redrob record in the on-disk JSONL schema."""
    return {
        "candidate_id": "R1",
        "profile": {
            "anonymized_name": "Candidate R1",
            "headline": "Senior ML Engineer",
            "summary": "Embeddings and ranking.",
            "current_title": "Senior ML Engineer",
            "current_company": "ProductCo",
            "current_company_size": "201-500",
            "current_industry": "technology",
            "location": "Bangalore",
            "country": "India",
            "years_of_experience": 7,
        },
        "redrob_signals": {
            "last_active_date": "2026-06-01",
            "open_to_work_flag": True,
            "recruiter_response_rate": 0.6,
            "applications_submitted_30d": 3,
            "notice_period_days": 30,
            "github_activity_score": 40,
            "interview_completion_rate": 0.85,
        },
        "career_history": [
            {"title": "Junior ML Engineer", "company": "ProductCo", "industry": "technology",
             "duration_months": 36, "is_current": False, "start_date": "2019-01-01"},
            {"title": "Senior ML Engineer", "company": "ProductCo", "industry": "technology",
             "duration_months": 48, "is_current": True, "start_date": "2022-01-01"},
        ],
        "skills": [
            {"name": "Python", "proficiency": "expert", "endorsements": 30, "duration_months": 84},
            {"name": "Embeddings", "proficiency": "advanced", "endorsements": 8, "duration_months": 36},
        ],
        "education": [
            {"degree": "B.Tech", "field_of_study": "CS", "institution": "IIT",
             "tier": "tier1", "end_year": 2018},
        ],
    }
