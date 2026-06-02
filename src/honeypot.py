"""Honeypot Detector — identifies candidates with subtly impossible profiles.

The hackathon dataset contains ~80 honeypots. Submissions with >10% honeypots
in top 100 are auto-disqualified (submission_spec Section 7).

Detection heuristics:
  1. YoE vs career history mismatch
  2. Expert skill with near-zero usage time
  3. Implausibly high expert skill count in short career
  4. Title-skill mismatch (Marketing Manager with 8 AI expert skills)
"""

from __future__ import annotations

import logging

from src.config import (
    HONEYPOT_EXPERT_MIN_MONTHS,
    HONEYPOT_EXPERT_SKILL_COUNT_LIMIT,
    HONEYPOT_YOE_CAREER_RATIO_THRESHOLD,
)

logger = logging.getLogger(__name__)


class HoneypotDetector:
    def detect(self, candidate: dict) -> tuple[bool, list[str]]:
        """Returns (is_honeypot, list_of_reasons). Pure computation, no I/O."""
        reasons: list[str] = []

        reasons += self._check_yoe_career_mismatch(candidate)
        reasons += self._check_expert_no_time(candidate)
        reasons += self._check_too_many_experts_short_career(candidate)
        reasons += self._check_title_skill_mismatch(candidate)

        is_honeypot = len(reasons) >= 2  # require 2+ signals to flag
        if is_honeypot:
            logger.debug(
                "Honeypot detected: %s — %s",
                candidate.get("candidate_id"), reasons,
            )
        return is_honeypot, reasons

    # ── Checks ────────────────────────────────────────────────────────────────

    def _check_yoe_career_mismatch(self, c: dict) -> list[str]:
        yoe = c.get("years_of_experience", 0)
        total_months = c.get("total_career_months", 0)
        career_years = total_months / 12.0 if total_months else 0

        if yoe > 0 and career_years > 0:
            if yoe > career_years * HONEYPOT_YOE_CAREER_RATIO_THRESHOLD:
                return [f"yoe_career_mismatch:{yoe:.1f}yr_claimed_vs_{career_years:.1f}yr_career"]
        return []

    def _check_expert_no_time(self, c: dict) -> list[str]:
        flags: list[str] = []
        for s in c.get("skills_with_meta", []):
            if s.get("proficiency") == "expert" and int(s.get("duration_months", 0)) < HONEYPOT_EXPERT_MIN_MONTHS:
                flags.append(f"expert_no_time:{s['name']}")
        return flags[:3]  # cap at 3 to avoid log spam

    def _check_too_many_experts_short_career(self, c: dict) -> list[str]:
        total_months = c.get("total_career_months", 0)
        if total_months >= 60:
            return []
        expert_count = sum(
            1 for s in c.get("skills_with_meta", [])
            if s.get("proficiency") == "expert"
        )
        if expert_count > HONEYPOT_EXPERT_SKILL_COUNT_LIMIT:
            return [f"too_many_experts:{expert_count}_in_{total_months}mo_career"]
        return []

    def _check_title_skill_mismatch(self, c: dict) -> list[str]:
        """Flag candidates with non-technical titles but many advanced AI skills."""
        from src.config import DISQUALIFYING_TITLES
        title = (c.get("current_title") or "").lower()
        is_non_technical = any(dt in title for dt in DISQUALIFYING_TITLES)
        if not is_non_technical:
            return []

        adv_ai_skills = {
            "machine learning", "deep learning", "nlp", "natural language processing",
            "pytorch", "tensorflow", "transformers", "llm", "rag", "fine-tuning",
            "embeddings", "vector database", "bert", "gpt",
        }
        candidate_skills = {s.lower() for s in c.get("skills", [])}
        advanced_count = sum(
            1 for s in c.get("skills_with_meta", [])
            if s["name"].lower() in adv_ai_skills
            and s.get("proficiency") in ("advanced", "expert")
        )
        if advanced_count >= 5:
            return [f"title_skill_mismatch:{title}_with_{advanced_count}_advanced_ai_skills"]
        return []
