"""Behavioral Scorer — maps redrob platform signals to a 0-1 score.

Signal groups (weighted):
  1. Recency     (40%) — exponential decay on days since last active
  2. Engagement  (25%) — applications, profile views, saved-by-recruiters, search appearances
  3. Response    (15%) — recruiter_response_rate (strongest hiring proxy)
  4. Conversion  (12%) — graded offer-acceptance / interview-completion / github activity
  5. Notice      ( 8%) — notice period (lower = easier to hire)

No external dependencies. Pure Python math.
"""

from __future__ import annotations

import math

from src.config import (
    CONVERSION_GITHUB_WEIGHT,
    CONVERSION_INTERVIEW_WEIGHT,
    CONVERSION_NEUTRAL,
    CONVERSION_OFFER_WEIGHT,
    CONVERSION_WEIGHT,
    ENGAGEMENT_APPS_WEIGHT,
    ENGAGEMENT_OPEN_TO_WORK_WEIGHT,
    ENGAGEMENT_SAVED_WEIGHT,
    ENGAGEMENT_SEARCH_WEIGHT,
    ENGAGEMENT_VIEWS_WEIGHT,
    ENGAGEMENT_WEIGHT,
    GITHUB_SCORE_CAP,
    NOTICE_WEIGHT,
    RECENCY_DECAY_LAMBDA,
    RECENCY_WEIGHT,
    RESPONSE_RATE_WEIGHT,
)


class BehavioralScorer:
    """Converts redrob_signals behavioral dict (pre-parsed) to 0-1 score."""

    def score(self, signals: dict) -> float:
        recency = self._recency(signals.get("last_active_days", 365))
        engagement = self._engagement(signals)
        response = float(signals.get("recruiter_response_rate", 0.0))
        conversion = self._conversion(signals)
        notice = self._notice(signals.get("notice_period_days", 90))

        return min(
            1.0,
            RECENCY_WEIGHT * recency
            + ENGAGEMENT_WEIGHT * engagement
            + RESPONSE_RATE_WEIGHT * response
            + CONVERSION_WEIGHT * conversion
            + NOTICE_WEIGHT * notice,
        )

    # ── Sub-scorers ───────────────────────────────────────────────────────────

    def _recency(self, days: int) -> float:
        """Exponential decay: active today → 1.0; 30 days → ~0.50; 90 days → ~0.13."""
        return math.exp(-RECENCY_DECAY_LAMBDA * max(0, int(days)))

    def _engagement(self, s: dict) -> float:
        apps = min(s.get("applications_count", 0), 15) / 15
        views = min(s.get("profile_views_last_30d", 0), 120) / 120
        saved = min(s.get("saved_by_recruiters_30d", 0), 10) / 10
        # Third recruiter-demand signal alongside views + saved.
        search = min(s.get("search_appearances_last_30d", 0), 200) / 200
        open_to_work = 1.0 if s.get("open_to_work", False) else 0.0
        return (
            ENGAGEMENT_APPS_WEIGHT * apps
            + ENGAGEMENT_VIEWS_WEIGHT * views
            + ENGAGEMENT_SAVED_WEIGHT * saved
            + ENGAGEMENT_SEARCH_WEIGHT * search
            + ENGAGEMENT_OPEN_TO_WORK_WEIGHT * open_to_work
        )

    def _conversion(self, s: dict) -> float:
        """Graded hireability signals: offer-acceptance, interview-completion,
        github activity. Previously binarised (github/interview) or dropped
        (offer_acceptance_rate). Now continuous.

        offer_acceptance_rate and github_activity_score carry a -1 sentinel for
        ~60-65% of candidates (no prior offers / no GitHub). Missing data maps to
        a NEUTRAL value so absence is never punished — only differentiated when
        the signal actually exists.
        """
        offer = float(s.get("offer_acceptance_rate", -1))
        offer = CONVERSION_NEUTRAL if offer < 0 else min(1.0, max(0.0, offer))

        interview = min(1.0, max(0.0, float(s.get("interview_completion_rate", 0.0))))

        github = float(s.get("github_activity_score", -1))
        github = CONVERSION_NEUTRAL if github < 0 else min(github, GITHUB_SCORE_CAP) / GITHUB_SCORE_CAP

        return (
            CONVERSION_OFFER_WEIGHT * offer
            + CONVERSION_INTERVIEW_WEIGHT * interview
            + CONVERSION_GITHUB_WEIGHT * github
        )

    def _notice(self, days: int) -> float:
        """Lower notice period → easier to hire → higher score."""
        if days <= 0:
            return 1.0
        if days <= 30:
            return 1.0
        if days <= 60:
            return 0.75
        if days <= 90:
            return 0.50
        if days <= 120:
            return 0.30
        return 0.10
