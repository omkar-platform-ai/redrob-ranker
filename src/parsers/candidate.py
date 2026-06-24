"""Candidate parser — maps the real redrob candidates.jsonl schema to the internal dict.

The internal schema is designed for the scoring pipeline; it normalises
all field names, computes derived features (seniority, tenure, promotions),
and pre-calculates signals the scorers need.

This module has ZERO external dependencies beyond stdlib + dateutil so it
can be used in the compute-constrained ranking step.
"""

from __future__ import annotations

import gzip
import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

from dateutil.parser import parse as parse_date

logger = logging.getLogger(__name__)

# ── Seniority inference ───────────────────────────────────────────────────────

_SENIORITY_BANDS: list[tuple[float, float, str]] = [
    (0, 2, "junior"),
    (2, 5, "mid"),
    (5, 8, "senior"),
    (8, 12, "lead"),
    (12, 99, "principal"),
]


def _derive_seniority(yoe: float) -> str:
    for lo, hi, label in _SENIORITY_BANDS:
        if lo <= yoe < hi:
            return label
    return "principal"


# ── Career stats ──────────────────────────────────────────────────────────────

_IC_LADDER = ("junior", "senior", "lead", "staff", "principal", "distinguished")
_MGMT_LADDER = ("manager", "head", "director", "vp", "cto", "ceo")


def _title_rank(title: str) -> tuple[int, int]:
    t = title.lower()
    ic = next((i for i, kw in enumerate(_IC_LADDER) if kw in t), -1)
    mg = next((i for i, kw in enumerate(_MGMT_LADDER) if kw in t), -1)
    return ic, mg


def _derive_career_stats(career_history: list[dict]) -> dict[str, Any]:
    """Compute avg_tenure, distinct_titles, promotion_count from career_history."""
    if not career_history:
        return {"avg_tenure_months": 0.0, "distinct_title_count": 0, "promotions_count": 0,
                "total_career_months": 0, "product_company_months": 0,
                "services_company_months": 0, "all_consulting": False}

    from src.config import CONSULTING_FIRMS, PRODUCT_INDUSTRIES, SERVICES_INDUSTRIES

    months_list = [int(e.get("duration_months", 0)) for e in career_history]
    titles = {(e.get("title") or "").strip().lower() for e in career_history}
    total_months = sum(months_list)

    # Promotion count
    sorted_hist = sorted(career_history, key=lambda e: e.get("start_date", "2000-01-01"))
    prev = (-1, -1)
    promotions = 0
    for e in sorted_hist:
        r = _title_rank(e.get("title", ""))
        if r[0] > prev[0] or r[1] > prev[1]:
            promotions += 1
        prev = (max(prev[0], r[0]), max(prev[1], r[1]))
    promotions = max(0, promotions - 1)

    # Product vs services company months
    product_months = 0
    services_months = 0
    company_names: list[str] = []
    for e in career_history:
        ind = (e.get("industry") or "").lower()
        dm = int(e.get("duration_months", 0))
        company_names.append((e.get("company") or "").lower())
        if any(p in ind for p in PRODUCT_INDUSTRIES):
            product_months += dm
        elif any(s in ind for s in SERVICES_INDUSTRIES):
            services_months += dm

    # All-consulting check — company name based
    all_consulting = all(
        any(cf in cn for cf in CONSULTING_FIRMS)
        for cn in company_names if cn
    )

    return {
        "avg_tenure_months": total_months / len(months_list) if months_list else 0.0,
        "distinct_title_count": len(titles),
        "promotions_count": promotions,
        "total_career_months": total_months,
        "product_company_months": product_months,
        "services_company_months": services_months,
        "all_consulting": all_consulting,
    }


# ── Skill normalisation ───────────────────────────────────────────────────────

def _match_assessment(skill_name: str, assessments: dict) -> float:
    """Measured Redrob assessment score (0-100) for a skill, or -1 if unassessed.

    `skill_assessment_scores` is the only OBJECTIVE proficiency signal in the
    dataset (everything else is self-declared). Match conservatively on a
    normalised key (lowercase, alphanumerics only): a false attribution would
    corrupt the skill score, so we prefer a miss over a wrong match. Stdlib only
    (no rapidfuzz) to keep this module rank-time-safe.
    """
    if not assessments:
        return -1.0
    target = "".join(ch for ch in skill_name.lower() if ch.isalnum())
    if not target:
        return -1.0
    for k, v in assessments.items():
        norm_k = "".join(ch for ch in str(k).lower() if ch.isalnum())
        if norm_k and norm_k == target:
            return float(v)
    return -1.0


def _parse_skills(
    raw_skills: list[dict], assessments: dict | None = None
) -> tuple[list[str], list[dict]]:
    """Returns (skill_names_list, skills_with_meta_list).

    Each skill meta carries an `assessment_score` (0-100, or -1 when Redrob has no
    measured score for it) so the skill scorer can blend objective proficiency.
    """
    assessments = assessments or {}
    names: list[str] = []
    meta: list[dict] = []
    for s in raw_skills:
        name = (s.get("name") or "").strip()
        if not name:
            continue
        names.append(name)
        meta.append({
            "name": name,
            "proficiency": (s.get("proficiency") or "intermediate").lower(),
            "endorsements": int(s.get("endorsements", 0)),
            "duration_months": int(s.get("duration_months", 0)),
            "assessment_score": _match_assessment(name, assessments),
        })
    return names, meta


# ── Behavioral signals ────────────────────────────────────────────────────────

def _parse_behavioral(signals: dict) -> dict[str, Any]:
    """Map redrob_signals fields to the internal behavioral dict."""
    today = date.today()

    # last_active_days: compute from date string
    last_active_days = 365
    raw_date = signals.get("last_active_date")
    if raw_date:
        try:
            la = parse_date(str(raw_date)).date()
            last_active_days = max(0, (today - la).days)
        except Exception:
            pass

    apps = int(signals.get("applications_submitted_30d", 0))
    return {
        "last_active_days": last_active_days,
        "open_to_work": bool(signals.get("open_to_work_flag", False)),
        "recruiter_response_rate": float(signals.get("recruiter_response_rate", 0.0)),
        "applications_count": apps,
        "profile_views_last_30d": int(signals.get("profile_views_received_30d", 0)),
        "saved_by_recruiters_30d": int(signals.get("saved_by_recruiters_30d", 0)),
        "search_appearances_last_30d": int(signals.get("search_appearance_30d", 0)),
        "notice_period_days": int(signals.get("notice_period_days", 90)),
        "github_activity_score": float(signals.get("github_activity_score", -1)),
        "interview_completion_rate": float(signals.get("interview_completion_rate", 0.0)),
        "offer_acceptance_rate": float(signals.get("offer_acceptance_rate", -1)),
        "willing_to_relocate": bool(signals.get("willing_to_relocate", False)),
        "actively_applying": apps > 1,
        "preferred_work_mode": signals.get("preferred_work_mode", "flexible"),
    }


# ── Build embedding text ──────────────────────────────────────────────────────

def build_embedding_text(record: dict) -> str:
    """Rich text representation of a candidate for embedding.

    Prioritises career descriptions over raw skill lists because the JD
    explicitly says "the right answer is career history, not keyword matching".
    """
    parts: list[str] = []

    title = record.get("current_title", "")
    if title:
        parts.append(f"Role: {title}")

    summary = record.get("summary", "")
    if summary:
        parts.append(summary[:500])  # truncate long summaries

    # Career descriptions — most signal-rich text
    for job in (record.get("work_experience") or [])[:4]:
        desc = (job.get("description") or "").strip()[:300]
        role = job.get("title", "")
        co = job.get("company", "")
        if desc:
            parts.append(f"{role} at {co}: {desc}")

    # Skills — secondary, after career text
    skills = record.get("skills", [])
    if skills:
        parts.append("Skills: " + ", ".join(skills[:20]))

    yoe = record.get("years_of_experience", 0)
    parts.append(f"Experience: {yoe} years")

    return " | ".join(p for p in parts if p)


# ── Main parser ───────────────────────────────────────────────────────────────

def parse_redrob_candidate(raw: dict) -> dict:
    """Normalise a single raw redrob candidate record to the internal schema.

    Returns a flat dict consumed by scorers and the embedding engine.
    All fields have sensible defaults — scorers must never crash on missing data.
    """
    profile = raw.get("profile") or {}
    signals = raw.get("redrob_signals") or {}
    career_history = raw.get("career_history") or []
    raw_skills = raw.get("skills") or []
    education = raw.get("education") or []

    skill_names, skills_with_meta = _parse_skills(
        raw_skills, signals.get("skill_assessment_scores")
    )
    behavioral = _parse_behavioral(signals)
    career_stats = _derive_career_stats(career_history)

    yoe = float(profile.get("years_of_experience", 0))
    current_title = (profile.get("current_title") or "").strip()
    country = (profile.get("country") or "").strip()
    location = (profile.get("location") or "").strip()

    # work_experience in a shape the scorers understand
    work_experience = [
        {
            "title": (e.get("title") or ""),
            "company": (e.get("company") or ""),
            "industry": (e.get("industry") or ""),
            "company_size": (e.get("company_size") or ""),
            "months": int(e.get("duration_months", 0)),
            "is_current": bool(e.get("is_current", False)),
            "description": (e.get("description") or ""),
            "start_date": (e.get("start_date") or ""),
        }
        for e in career_history
    ]

    # GitHub activity → open-source proxy
    gh_score = float(signals.get("github_activity_score", -1))
    has_open_source = gh_score >= 30 if gh_score >= 0 else False

    candidate = {
        # Identity
        "candidate_id": str(raw.get("candidate_id", "")),
        "name": (profile.get("anonymized_name") or ""),
        # Profile
        "headline": (profile.get("headline") or ""),
        "summary": (profile.get("summary") or ""),
        "current_title": current_title,
        "current_company": (profile.get("current_company") or ""),
        "current_company_size": (profile.get("current_company_size") or ""),
        "current_industry": (profile.get("current_industry") or ""),
        "location": location,
        "country": country,
        "years_of_experience": yoe,
        "seniority_level": _derive_seniority(yoe),
        # Skills
        "skills": skill_names,
        "skills_with_meta": skills_with_meta,
        # Career
        "work_experience": work_experience,
        "avg_tenure_months": career_stats["avg_tenure_months"],
        "distinct_title_count": career_stats["distinct_title_count"],
        "promotions_count": career_stats["promotions_count"],
        "total_career_months": career_stats["total_career_months"],
        "product_company_months": career_stats["product_company_months"],
        "services_company_months": career_stats["services_company_months"],
        "all_consulting": career_stats["all_consulting"],
        "has_open_source_contributions": has_open_source,
        "github_activity_score": gh_score,
        # Education
        "education": [
            {
                "degree": (e.get("degree") or ""),
                "field": (e.get("field_of_study") or ""),
                "institution": (e.get("institution") or ""),
                "tier": (e.get("tier") or "unknown"),
                "end_year": int(e.get("end_year", 0)),
            }
            for e in education
        ],
        # Behavioral (pre-normalised)
        "behavioral_signals": behavioral,
    }

    # Build embedding text and store it
    candidate["embedding_text"] = build_embedding_text(candidate)
    return candidate


# ── Dataset loader ────────────────────────────────────────────────────────────

def load_candidates(path: str | Path) -> list[dict]:
    """Load candidates.jsonl or candidates.jsonl.gz and parse each record."""
    path = Path(path)
    logger.info("Loading candidates from %s", path)

    opener = gzip.open if path.suffix == ".gz" else open
    raw_records: list[dict] = []
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    raw_records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping malformed line: %s", exc)

    logger.info("Loaded %d raw records, parsing...", len(raw_records))
    candidates = [parse_redrob_candidate(r) for r in raw_records]
    logger.info("Parsed %d candidates", len(candidates))
    return candidates
