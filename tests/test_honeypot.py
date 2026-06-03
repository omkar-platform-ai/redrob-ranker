"""Honeypot detector tests — a profile is flagged only when 2+ signals fire.
Keeping the top-100 honeypot rate below 10% is a hard spec requirement.
"""

from __future__ import annotations

from tests.conftest import make_candidate

from src.honeypot import HoneypotDetector


def test_clean_profile_not_flagged(parsed_candidate):
    is_honeypot, reasons = HoneypotDetector().detect(parsed_candidate)
    assert is_honeypot is False
    assert reasons == []


def test_single_signal_not_enough():
    # Only YoE/career mismatch (10yr claimed vs ~2yr career) — one signal.
    c = make_candidate(years_of_experience=10.0, total_career_months=24)
    is_honeypot, reasons = HoneypotDetector().detect(c)
    assert len(reasons) >= 1
    assert is_honeypot is False


def test_two_signals_flag_honeypot():
    # Signal 1: YoE (10) >> career (~2yr). Signal 2: expert skill, ~0 months use.
    c = make_candidate(
        years_of_experience=10.0,
        total_career_months=24,
        skills_with_meta=[
            {"name": "PyTorch", "proficiency": "expert", "endorsements": 0, "duration_months": 1},
        ],
    )
    is_honeypot, reasons = HoneypotDetector().detect(c)
    assert is_honeypot is True
    assert len(reasons) >= 2


def test_title_skill_mismatch_signal():
    # Marketing title + many advanced AI skills triggers the mismatch check.
    ai = [
        {"name": n, "proficiency": "expert", "endorsements": 0, "duration_months": 12}
        for n in ["machine learning", "deep learning", "nlp", "pytorch", "transformers"]
    ]
    c = make_candidate(
        current_title="Marketing Manager",
        skills=["machine learning", "deep learning", "nlp", "pytorch", "transformers"],
        skills_with_meta=ai,
    )
    _, reasons = HoneypotDetector().detect(c)
    assert any("title_skill_mismatch" in r for r in reasons)
