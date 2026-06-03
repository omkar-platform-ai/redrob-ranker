"""Candidate parser tests — raw redrob schema → internal flat dict.
The parser must never crash on missing fields (all fields default).
"""

from __future__ import annotations

from src.parsers.candidate import build_embedding_text, parse_redrob_candidate


def test_parse_maps_core_fields(raw_candidate):
    c = parse_redrob_candidate(raw_candidate)
    assert c["candidate_id"] == "R1"
    assert c["current_title"] == "Senior ML Engineer"
    assert c["years_of_experience"] == 7
    assert c["country"] == "India"


def test_parse_derives_career_stats(raw_candidate):
    c = parse_redrob_candidate(raw_candidate)
    assert c["total_career_months"] == 84          # 36 + 48
    assert c["product_company_months"] == 84        # both roles in tech
    assert c["promotions_count"] >= 1               # ML → Senior ML
    assert c["all_consulting"] is False


def test_parse_skills_with_meta(raw_candidate):
    c = parse_redrob_candidate(raw_candidate)
    names = {s["name"] for s in c["skills_with_meta"]}
    assert "Python" in names
    assert all("proficiency" in s for s in c["skills_with_meta"])


def test_parse_empty_record_does_not_crash():
    c = parse_redrob_candidate({})
    assert c["candidate_id"] == ""
    assert c["years_of_experience"] == 0
    assert c["skills"] == []
    assert "embedding_text" in c


def test_embedding_text_includes_title_and_skills():
    text = build_embedding_text({
        "current_title": "ML Engineer",
        "skills": ["Python", "FAISS"],
        "years_of_experience": 5,
    })
    assert "ML Engineer" in text
    assert "Python" in text
