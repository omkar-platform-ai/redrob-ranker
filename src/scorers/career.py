"""Career Scorer — trajectory quality and hidden-gem detection.

Signals:
  1. Velocity     (40%) — reached current level in fewer years than average?
  2. Stability    (25%) — avg tenure per role (job-hopping penalty)
  3. Progression  (35%) — title/responsibility growth over time

Hidden-gem bonuses (capped at HIDDEN_GEM_CAP):
  - Open-source contributions (GitHub activity score ≥ 30)
  - Multi-promotion (promoted 2+ times)
  - High interview completion rate (≥ 0.8)
"""

from __future__ import annotations

from src.config import (
    CAREER_PROGRESSION_WEIGHT,
    CAREER_STABILITY_WEIGHT,
    CAREER_VELOCITY_WEIGHT,
    HIDDEN_GEM_BONUSES,
    HIDDEN_GEM_CAP,
)

_SENIORITY_EXPECTED_YOE = {
    "junior": 1.5,
    "mid": 3.5,
    "senior": 6.0,
    "lead": 9.0,
    "principal": 12.0,
}


class CareerScorer:
    """Scores career trajectory, stability, and hidden-gem signals."""

    def score(self, candidate: dict) -> tuple[float, float, list[str]]:
        """Returns (career_score, hidden_gem_bonus, gem_reasons)."""
        velocity = self._velocity(candidate)
        stability = self._stability(candidate.get("avg_tenure_months", 24))
        progression = self._progression(candidate)

        base = (
            CAREER_VELOCITY_WEIGHT * velocity
            + CAREER_STABILITY_WEIGHT * stability
            + CAREER_PROGRESSION_WEIGHT * progression
        )

        gem_bonus, gem_reasons = self._hidden_gem(candidate)
        total = min(1.0, base + gem_bonus)
        return total, gem_bonus, gem_reasons

    # ── Sub-scorers ───────────────────────────────────────────────────────────

    def _velocity(self, c: dict) -> float:
        """Fast-tracker: reached current level in fewer years than average."""
        yoe = c.get("years_of_experience", 0)
        level = c.get("seniority_level", "mid").lower()
        expected = _SENIORITY_EXPECTED_YOE.get(level, 4.0)
        if yoe <= 0:
            return 0.50
        if yoe <= expected:
            return min(1.0, 1.0 + 0.05 * (expected - yoe))
        return max(0.20, 1.0 - (yoe - expected) * 0.03)

    def _stability(self, avg_tenure_months: float) -> float:
        """Penalise extreme job-hopping; reward reasonable tenure."""
        if avg_tenure_months >= 18:
            return 1.0
        if avg_tenure_months >= 12:
            return 0.80
        if avg_tenure_months >= 6:
            return 0.50
        return 0.20

    def _progression(self, c: dict) -> float:
        """Did their title / responsibilities grow over time?"""
        promotions = c.get("promotions_count", 0)
        title_changes = c.get("distinct_title_count", 1)
        return min(1.0, 0.5 + promotions * 0.15 + (title_changes - 1) * 0.10)

    def _hidden_gem(self, c: dict) -> tuple[float, list[str]]:
        bonus = 0.0
        reasons: list[str] = []

        if c.get("has_open_source_contributions", False):
            bonus += HIDDEN_GEM_BONUSES["open_source"]
            reasons.append("open_source")

        if c.get("promotions_count", 0) >= 2:
            bonus += HIDDEN_GEM_BONUSES["multi_promotion"]
            reasons.append("multi_promotion")

        beh = c.get("behavioral_signals", {})
        if float(beh.get("interview_completion_rate", 0)) >= 0.80:
            bonus += 0.03
            reasons.append("high_interview_completion")

        return min(HIDDEN_GEM_CAP, bonus), reasons
