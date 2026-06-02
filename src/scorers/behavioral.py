"""Behavioral Scorer — maps redrob platform signals to a 0-1 score.

Signal groups (weighted):
  1. Recency     (45%) — exponential decay on days since last active
  2. Engagement  (30%) — applications, profile views, saved by recruiters
  3. Response    (15%) — recruiter_response_rate (strongest hiring proxy)
  4. Notice      (10%) — notice period (lower = easier to hire)

No external dependencies. Pure Python math.
"""

from __future__ import annotations

import math

from src.config import (
    ENGAGEMENT_WEIGHT,
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
        notice = self._notice(signals.get("notice_period_days", 90))

        return min(
            1.0,
            RECENCY_WEIGHT * recency
            + ENGAGEMENT_WEIGHT * engagement
            + RESPONSE_RATE_WEIGHT * response
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
        open_to_work = 1.0 if s.get("open_to_work", False) else 0.0
        return 0.30 * apps + 0.25 * views + 0.20 * saved + 0.25 * open_to_work

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
