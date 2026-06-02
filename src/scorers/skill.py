"""Skill Scorer — proficiency-weighted fuzzy skill matching.

Uses RapidFuzz for substring/token-set matching so aliases and abbreviations
("NLP" ↔ "Natural Language Processing") are handled without a hardcoded map.

Scoring per skill:
  proficiency_weight × 0.70
  + duration_bonus   × 0.20  (caps at SKILL_DURATION_CAP_MONTHS)
  + endorsement_bonus× 0.10  (caps at 0.10 extra)

No external dependencies beyond rapidfuzz (CPU-only, fast).
"""

from __future__ import annotations

from rapidfuzz import fuzz

from src.config import (
    SKILL_DURATION_CAP_MONTHS,
    SKILL_OPTIONAL_WEIGHT,
    SKILL_PROFICIENCY_WEIGHTS,
    SKILL_REQUIRED_WEIGHT,
)

_MATCH_THRESHOLD = 80  # RapidFuzz score 0-100


class SkillScorer:
    """Fuzzy, proficiency-aware skill matching."""

    def score(
        self,
        skills_with_meta: list[dict],
        required_skills: list[str],
        nice_to_have_skills: list[str],
    ) -> tuple[float, list[str]]:
        """Returns (score 0-1, list of matched skill names)."""
        if not required_skills and not nice_to_have_skills:
            return 0.50, []  # No JD skills listed — neutral score

        req_scores, req_matched = self._match_group(skills_with_meta, required_skills)
        opt_scores, opt_matched = self._match_group(skills_with_meta, nice_to_have_skills)

        if not nice_to_have_skills:
            total = sum(req_scores) / max(1, len(required_skills))
        elif not required_skills:
            total = sum(opt_scores) / max(1, len(nice_to_have_skills))
        else:
            r = sum(req_scores) / max(1, len(required_skills))
            o = sum(opt_scores) / max(1, len(nice_to_have_skills))
            total = SKILL_REQUIRED_WEIGHT * r + SKILL_OPTIONAL_WEIGHT * o

        matched = list(dict.fromkeys(req_matched + opt_matched))  # dedupe, preserve order
        return min(1.0, total), matched

    # ── Internals ─────────────────────────────────────────────────────────────

    def _match_group(
        self, skills_with_meta: list[dict], jd_skills: list[str]
    ) -> tuple[list[float], list[str]]:
        scores: list[float] = []
        matched_names: list[str] = []
        for jd_skill in jd_skills:
            best_score, best_name = self._best_match(skills_with_meta, jd_skill)
            scores.append(best_score)
            if best_score > 0:
                matched_names.append(best_name)
        return scores, matched_names

    def _best_match(
        self, skills_with_meta: list[dict], jd_skill: str
    ) -> tuple[float, str]:
        """Find the best-matching candidate skill for a JD skill."""
        best = 0.0
        best_name = ""
        jd_lower = jd_skill.lower()
        for s in skills_with_meta:
            candidate_skill = s["name"].lower()
            # Use token_set_ratio: handles word order and partial matches
            fuzz_score = fuzz.token_set_ratio(jd_lower, candidate_skill)
            if fuzz_score >= _MATCH_THRESHOLD:
                weight = self._skill_weight(s)
                if weight > best:
                    best = weight
                    best_name = s["name"]
        return best, best_name

    def _skill_weight(self, skill_meta: dict) -> float:
        """Compute weighted match score for a single candidate skill."""
        prof_w = SKILL_PROFICIENCY_WEIGHTS.get(
            skill_meta.get("proficiency", "intermediate"), 0.6
        )
        duration = min(skill_meta.get("duration_months", 0), SKILL_DURATION_CAP_MONTHS)
        duration_bonus = duration / SKILL_DURATION_CAP_MONTHS

        endorsements = min(skill_meta.get("endorsements", 0), 50)
        endorsement_bonus = endorsements / 50 * 0.10  # max 0.10

        return min(1.0, prof_w * 0.70 + duration_bonus * 0.20 + endorsement_bonus)
