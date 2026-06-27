"""Role-Fit Scorer — structural fit beyond skill keywords.

This is what separates a good ranking system from a keyword-matcher.
A candidate's title, industry, and company type are stronger proxies
for genuine fit than skill list membership.

Signals:
  1. Title match     (40%) — is the title in the AI/ML engineering family?
  2. Company type    (35%) — product company vs. services/consulting
  3. Location        (15%) — hub tier + domestic relocation intent, scaled by work-mode fit
  4. YoE band fit    (10%) — is experience within the JD's stated range?

Hard penalties (multipliers, not additive):
  - Consulting-only career: ×0.25
  - Disqualifying title (HR, Marketing, etc.): ×0.10
  - JD-stated disqualifiers (title-chasing, CV/speech/robotics primary): ×0.50
"""

from __future__ import annotations

import re

from src.config import (
    AI_ENGINEER_TITLES,
    CONSULTING_ONLY_MULTIPLIER,
    CV_SPEECH_ROBOTICS_TOKENS,
    DISQUALIFYING_TITLES,
    INDIA_TIER1_CITIES,
    INDIA_TIER2_CITIES,
    JD_DISQUALIFIER_MULTIPLIER,
    LOCATION_ABROAD,
    LOCATION_ABROAD_RELOCATE,
    LOCATION_INDIA_OTHER,
    LOCATION_INDIA_OTHER_RELOCATE,
    LOCATION_TIER1,
    LOCATION_TIER2,
    LOCATION_TIER2_RELOCATE,
    ML_INSCOPE_TOKENS,
    ROLE_FIT_COMPANY_WEIGHT,
    ROLE_FIT_LOCATION_WEIGHT,
    ROLE_FIT_TITLE_WEIGHT,
    ROLE_FIT_YOE_WEIGHT,
    TITLE_CHASING_FULL_PENALTY_MONTHS,
    TITLE_CHASING_MAX_TENURE_MONTHS,
    WORK_MODE_FIT,
    YOE_OVER_PENALTY_FLOOR,
    YOE_OVER_PENALTY_PER_YEAR,
    YOE_UNDER_PENALTY_PER_YEAR,
)


# Engineering role nouns used for title scoring (word-boundary matched).
_ROLE_WORDS: tuple[str, ...] = ("engineer", "scientist", "developer", "architect")


class RoleFitScorer:
    """Computes structural fit: title + company-type + location + YoE band."""

    def score(self, candidate: dict, parsed_jd) -> float:
        title_s = self._title_score(candidate.get("current_title", ""))
        company_s = self._company_type_score(candidate)
        location_s = self._location_score(candidate)
        yoe_s = self._yoe_band_score(
            candidate.get("years_of_experience", 0),
            parsed_jd.min_experience_years,
            parsed_jd.max_experience_years,
        )

        base = (
            ROLE_FIT_TITLE_WEIGHT * title_s
            + ROLE_FIT_COMPANY_WEIGHT * company_s
            + ROLE_FIT_LOCATION_WEIGHT * location_s
            + ROLE_FIT_YOE_WEIGHT * yoe_s
        )

        # Hard penalties
        if self._is_disqualifying_title(candidate.get("current_title", "")):
            base *= 0.10
        if candidate.get("all_consulting", False):
            base *= CONSULTING_ONLY_MULTIPLIER
        # JD-stated disqualifiers (was a latent bug: parsed but no scorer read them)
        base *= self._jd_disqualifier_multiplier(candidate, parsed_jd)

        return min(1.0, base)

    # ── Sub-scorers ───────────────────────────────────────────────────────────

    def _title_score(self, title: str) -> float:
        """Is the current title in the AI/ML engineering family?"""
        t = title.lower()
        for eng_title in AI_ENGINEER_TITLES:
            if eng_title in t:
                return 1.0
        # Multi-word AI/ML titles the exact set misses (e.g. "Recommendation
        # Systems Engineer"): an engineering role word plus an in-scope ML token.
        # Word-boundary matched so "search" can't fire inside "research".
        if (self._title_contains_word(t, _ROLE_WORDS)
                and self._title_contains_word(t, ML_INSCOPE_TOKENS)):
            return 1.0
        # Partial credit for adjacent titles
        if any(kw in t for kw in _ROLE_WORDS):
            return 0.5
        if any(kw in t for kw in ["analyst", "researcher", "lead"]):
            return 0.3
        return 0.0

    def _is_disqualifying_title(self, title: str) -> bool:
        t = title.lower()
        return any(dt in t for dt in DISQUALIFYING_TITLES)

    def _jd_disqualifier_multiplier(self, candidate: dict, parsed_jd) -> float:
        """Multiplier from JD-STATED disqualifiers (previously parsed but unused).

        Only reliably-computable predicates are wired; fuzzy JD disqualifiers
        ("LLM-only <12mo", "pure research") stay as JD text only by design.
        """
        dqs = " ".join(getattr(parsed_jd, "disqualifiers", []) or []).lower()
        if not dqs:
            return 1.0
        mult = 1.0
        # Title-chasing / short tenure ("title-chasing every 1.5 years", "job-hopping").
        # Penalty ramps with tenure instead of a hard cliff, so a 17.5mo candidate
        # isn't punished like a 6mo hopper.
        if any(tok in dqs for tok in ("chas", "hopping", "1.5", "short tenure", "job hop")):
            mult *= self._tenure_chasing_multiplier(candidate.get("avg_tenure_months", 99))
        # CV / speech / robotics as the PRIMARY domain (out of scope for a
        # text/retrieval/ranking ML role). Suppressed when the title also carries
        # an in-scope ML token (an "ML Engineer" who touches vision stays clean).
        if any(tok in dqs for tok in ("cv", "speech", "robotics", "vision", "autonomous")):
            title = candidate.get("current_title", "")
            if (self._title_contains_word(title, CV_SPEECH_ROBOTICS_TOKENS)
                    and not self._title_contains_word(title, ML_INSCOPE_TOKENS)):
                mult *= JD_DISQUALIFIER_MULTIPLIER
        return mult

    @staticmethod
    def _title_contains_word(title: str, tokens) -> bool:
        """Word-boundary match so 'vision' doesn't hit 'division', 'ml' not 'html'."""
        t = (title or "").lower()
        return any(re.search(r"\b" + re.escape(tok) + r"\b", t) for tok in tokens)

    @staticmethod
    def _tenure_chasing_multiplier(avg_tenure_months: float) -> float:
        """Title-chasing penalty as a gradient, not a cliff.

        Full JD_DISQUALIFIER_MULTIPLIER at/below TITLE_CHASING_FULL_PENALTY_MONTHS,
        easing linearly to 1.0 (no penalty) at TITLE_CHASING_MAX_TENURE_MONTHS. This
        avoids a knife-edge where a 17.5mo tenure is punished like a 6mo one.
        """
        full = TITLE_CHASING_FULL_PENALTY_MONTHS
        ceiling = TITLE_CHASING_MAX_TENURE_MONTHS
        if avg_tenure_months >= ceiling:
            return 1.0
        if avg_tenure_months <= full:
            return JD_DISQUALIFIER_MULTIPLIER
        frac = (avg_tenure_months - full) / (ceiling - full)
        return JD_DISQUALIFIER_MULTIPLIER + frac * (1.0 - JD_DISQUALIFIER_MULTIPLIER)

    def _company_type_score(self, candidate: dict) -> float:
        """Product company background scores higher than services/consulting."""
        total = max(1, candidate.get("total_career_months", 1))
        product_ratio = candidate.get("product_company_months", 0) / total
        services_ratio = candidate.get("services_company_months", 0) / total

        if product_ratio >= 0.60:
            return 1.0
        if product_ratio >= 0.40:
            return 0.75
        if product_ratio >= 0.20:
            return 0.50
        if services_ratio >= 0.80:
            return 0.20  # mostly services
        return 0.35  # mixed / unclear industry

    def _location_score(self, candidate: dict) -> float:
        """India hub proximity scores highest; relocation willingness lifts non-Tier-1
        domestic candidates toward the hub; abroad scored by relocation intent. The
        result is scaled by work-mode fit for the hybrid role (remote-only softened)."""
        bsig = candidate.get("behavioral_signals", {})
        country = (candidate.get("country") or "").lower()
        location = (candidate.get("location") or "").lower()
        willing = bsig.get("willing_to_relocate", False)

        if country in ("india", "in"):
            if any(city in location for city in INDIA_TIER1_CITIES):
                base = LOCATION_TIER1
            elif any(city in location for city in INDIA_TIER2_CITIES):
                base = LOCATION_TIER2_RELOCATE if willing else LOCATION_TIER2
            else:
                base = LOCATION_INDIA_OTHER_RELOCATE if willing else LOCATION_INDIA_OTHER
        elif willing:
            base = LOCATION_ABROAD_RELOCATE
        else:
            base = LOCATION_ABROAD

        mode = (bsig.get("preferred_work_mode") or "flexible").lower()
        return base * WORK_MODE_FIT.get(mode, 1.0)

    def _yoe_band_score(self, yoe: float, min_yoe: int, max_yoe: int) -> float:
        """Score based on how well YoE fits the JD range."""
        if min_yoe <= yoe <= max_yoe:
            return 1.0
        if yoe < min_yoe:
            gap = min_yoe - yoe
            return max(0.0, 1.0 - gap * YOE_UNDER_PENALTY_PER_YEAR)
        # Over-qualified
        gap = yoe - max_yoe
        return max(YOE_OVER_PENALTY_FLOOR, 1.0 - gap * YOE_OVER_PENALTY_PER_YEAR)
