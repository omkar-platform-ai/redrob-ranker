"""Tests for the no-LLM JD keyword fallback (`src/parsers/jd.py`).

This path runs in the sandbox demo and whenever no LLM key is set, so it must be
deterministic and rank-time safe (no network). The judged pipeline uses the LLM
parse in pre-compute; these tests cover the offline equivalent.
"""

from __future__ import annotations

from pathlib import Path

from src.parsers.jd import _find_skills, _keyword_fallback

JD_PATH = Path(__file__).resolve().parent.parent / "data" / "job_description.txt"


# ── skill extraction: vocab + aliases + casing ───────────────────────────────

def test_find_skills_canonical_casing():
    """Acronyms/brands come back in canonical display casing, not lowercase."""
    got = _find_skills("we use python, faiss, pytorch, llm and rag in production")
    assert set(got) == {"Python", "FAISS", "PyTorch", "LLM", "RAG"}


def test_find_skills_alias_folding():
    """Spelling variants fold to the canonical skill."""
    assert _find_skills("strong k8s and golang experience") == ["Kubernetes", "Go"]
    assert _find_skills("built a vector db on postgresql") == [
        "vector database", "Postgres"
    ]


def test_find_skills_new_vocab_detected():
    """Previously out-of-vocab skills are no longer silently dropped."""
    got = set(_find_skills("experience with ray, vllm, triton, databricks and react"))
    assert {"Ray", "vLLM", "Triton", "Databricks", "React"} <= got


def test_find_skills_dedupes_canonical():
    """A skill and its alias collapse to one entry."""
    got = _find_skills("we run go, golang, kubernetes and k8s")
    assert got.count("Go") == 1
    assert got.count("Kubernetes") == 1


def test_find_skills_boundary_no_false_positives():
    """Boundary match: 'go' not inside 'google', 'ml' not inside 'html'."""
    assert _find_skills("we use google docs and write html") == []


# ── full keyword fallback on the real JD ─────────────────────────────────────

def test_keyword_fallback_on_real_jd():
    parsed = _keyword_fallback(JD_PATH.read_text(encoding="utf-8"))
    skills = set(parsed.required_skills) | set(parsed.nice_to_have_skills)
    # Core skills the JD genuinely calls for must be present and display-cased.
    assert {"Python", "Embeddings", "Retrieval", "Ranking", "FAISS"} <= skills
    # No lowercase acronym leaked through (casing applied everywhere).
    assert "llm" not in skills and "faiss" not in skills
    # The JD is a senior 5–9 AI role.
    assert (parsed.min_experience_years, parsed.max_experience_years) == (5, 9)
    assert parsed.required_skills, "required skills must not be empty"


# ── title & experience locality ──────────────────────────────────────────────

def test_extract_title_skips_non_title_first_line():
    """A JD opening with 'About Us' still finds the real title further down."""
    jd = "About Us\nWe are a great company.\n\nSenior Machine Learning Engineer\nDo ML."
    assert _keyword_fallback(jd).title == "Senior Machine Learning Engineer"


def test_extract_title_first_line_fallback():
    """No title-looking line anywhere → fall back to the first meaningful line."""
    jd = "Growth Hacker Extraordinaire\nWe do growth."
    assert _keyword_fallback(jd).title == "Growth Hacker Extraordinaire"


def test_extract_experience_prefers_cue_locality():
    """An open band by an experience cue wins over a stray earlier 'N years ago'."""
    jd = ("AI Engineer\n"
          "We raised our seed round 5 years ago and never looked back.\n"
          "Minimum experience: 8+ years shipping ML systems.")
    parsed = _keyword_fallback(jd)
    assert (parsed.min_experience_years, parsed.max_experience_years) == (8, 12)


def test_extract_experience_default_band():
    jd = "AI Engineer\nWe build cool stuff with no stated band."
    parsed = _keyword_fallback(jd)
    assert (parsed.min_experience_years, parsed.max_experience_years) == (5, 9)


# ── section-aware exclusion ───────────────────────────────────────────────────

def test_excluded_block_skill_dropped_from_nice():
    """A skill mentioned only inside a 'we do NOT want' block is dropped."""
    jd = ("Senior AI Engineer\n"
          "Requirements: strong Python and embeddings experience.\n"
          "Nice to have: FAISS experience.\n"
          "Things we explicitly do NOT want: React-only frontend folks.\n")
    parsed = _keyword_fallback(jd)
    skills = set(parsed.required_skills) | set(parsed.nice_to_have_skills)
    assert "React" not in skills           # only in the do-NOT-want block
    assert "Python" in parsed.required_skills
    assert "FAISS" in parsed.nice_to_have_skills


def test_required_skill_survives_being_disparaged_later():
    """A skill that is required AND later disparaged stays required."""
    jd = ("AI Engineer\n"
          "Requirements: production Python and FAISS.\n"
          "Things we do NOT want: people who only know FAISS from a tutorial.\n")
    parsed = _keyword_fallback(jd)
    assert "FAISS" in parsed.required_skills


def test_exclude_window_keeps_trailing_positive_skills():
    """The exclude window must not swallow a later positive section: the real JD's
    'recommendation' (in 'how to read between the lines') stays a skill."""
    parsed = _keyword_fallback(JD_PATH.read_text(encoding="utf-8"))
    skills = set(parsed.required_skills) | set(parsed.nice_to_have_skills)
    assert "Recommendation" in skills
