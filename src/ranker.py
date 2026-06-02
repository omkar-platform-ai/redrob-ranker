"""Ranking Engine — orchestrates all scorers into one final ranked list.

This is the ONLY module that runs during the constraint-bound ranking step.
Zero LLM calls. Zero network I/O. Pure CPU math.

Pipeline:
  1. Load pre-built FAISS index from disk
  2. Embed JD (cached) → ANN search → top-500 candidate IDs + cosine scores
  3. For each of 500: compute role_fit, skill, behavioral, career scores
  4. Weighted fusion → composite score
  5. Honeypot detection → zero-score honeypots
  6. Sort → take top 100
  7. Generate reasoning (template-based, no LLM)
  8. Write CSV (4 columns: candidate_id, rank, score, reasoning)
"""

from __future__ import annotations

import csv
import logging
import time
from dataclasses import dataclass, field

from src.config import (
    CONFIDENCE_HIGH,
    CONFIDENCE_MEDIUM,
    OUTPUT_COLUMNS,
    TOP_K_FINAL,
    TOP_K_RETRIEVE,
    WEIGHTS,
)
from src.honeypot import HoneypotDetector
from src.reasoning import ReasoningGenerator, ScoredCandidate
from src.scorers.behavioral import BehavioralScorer
from src.scorers.career import CareerScorer
from src.scorers.role_fit import RoleFitScorer
from src.scorers.skill import SkillScorer

logger = logging.getLogger(__name__)


class RankingEngine:
    """Stateless scorer composition. One instance per process."""

    def __init__(self) -> None:
        self.behavioral = BehavioralScorer()
        self.career = CareerScorer()
        self.role_fit = RoleFitScorer()
        self.skill = SkillScorer()
        self.honeypot = HoneypotDetector()
        self.reasoning = ReasoningGenerator()

    def rank(
        self,
        candidates_by_id: dict[str, dict],    # {candidate_id: parsed_candidate}
        ann_results: list[tuple[str, float]],  # [(candidate_id, cosine_score), ...]
        parsed_jd,
    ) -> list[ScoredCandidate]:
        """Score, sort, deduplicate, and return the top TOP_K_FINAL candidates."""
        t0 = time.perf_counter()
        scored: list[ScoredCandidate] = []

        for cid, sem_score in ann_results:
            c = candidates_by_id.get(cid)
            if c is None:
                continue

            # ── Scorers ──────────────────────────────────────────────────────
            role_s = self.role_fit.score(c, parsed_jd)
            skill_s, matched = self.skill.score(
                c.get("skills_with_meta", []),
                parsed_jd.required_skills,
                parsed_jd.nice_to_have_skills,
            )
            beh_s = self.behavioral.score(c.get("behavioral_signals", {}))
            car_s, gem_bonus, gem_reasons = self.career.score(c)

            # ── Weighted fusion ───────────────────────────────────────────────
            composite = (
                sem_score    * WEIGHTS["semantic"]
                + role_s     * WEIGHTS["role_fit"]
                + skill_s    * WEIGHTS["skill"]
                + beh_s      * WEIGHTS["behavioral"]
                + car_s      * WEIGHTS["career"]
            )

            # ── Honeypot check ────────────────────────────────────────────────
            is_honeypot, _ = self.honeypot.detect(c)
            if is_honeypot:
                composite = 0.0  # ensure it never reaches top 100

            sc = ScoredCandidate(
                candidate_id=cid,
                rank=0,
                score=round(composite, 6),
                semantic_score=round(sem_score, 4),
                role_fit_score=round(role_s, 4),
                skill_score=round(skill_s, 4),
                behavioral_score=round(beh_s, 4),
                career_score=round(car_s, 4),
                matched_skills=matched,
                hidden_gem_reasons=gem_reasons,
                is_honeypot=is_honeypot,
                current_title=c.get("current_title", ""),
                years_of_experience=c.get("years_of_experience", 0),
                current_company=c.get("current_company", ""),
                current_industry=c.get("current_industry", ""),
                all_consulting=c.get("all_consulting", False),
                notice_period_days=c.get("behavioral_signals", {}).get("notice_period_days", 90),
                location=c.get("location", ""),
                country=c.get("country", ""),
                confidence=self._confidence(composite),
            )
            scored.append(sc)

        # ── Sort + rank ───────────────────────────────────────────────────────
        scored.sort(key=lambda x: x.score, reverse=True)
        top = scored[:TOP_K_FINAL]

        # Enforce non-increasing scores (break ties by candidate_id ascending)
        prev_score = top[0].score if top else 1.0
        for i, sc in enumerate(top):
            if sc.score > prev_score:
                sc = ScoredCandidate(**{**sc.__dict__, "score": prev_score})
                top[i] = sc
            prev_score = sc.score
            sc.rank = i + 1

        # ── Generate reasoning ────────────────────────────────────────────────
        for sc in top:
            c = candidates_by_id[sc.candidate_id]
            sc.reasoning = self.reasoning.generate(c, sc, parsed_jd)

        elapsed = time.perf_counter() - t0
        logger.info(
            "Ranked %d candidates in %.2fs (from %d ANN results)",
            len(top), elapsed, len(ann_results),
        )
        if elapsed > 240:
            logger.warning("Ranking took %.1fs — approaching 5-minute budget!", elapsed)

        return top

    def _confidence(self, score: float) -> str:
        if score >= CONFIDENCE_HIGH:
            return "HIGH"
        if score >= CONFIDENCE_MEDIUM:
            return "MEDIUM"
        return "LOW"

    @staticmethod
    def to_csv(ranked: list[ScoredCandidate], path: str) -> None:
        """Write submission CSV in the exact format required by the spec."""
        if not ranked:
            raise ValueError("Ranked list is empty — nothing to write")

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()
            for sc in ranked:
                writer.writerow({
                    "candidate_id": sc.candidate_id,
                    "rank": sc.rank,
                    "score": sc.score,
                    "reasoning": sc.reasoning,
                })
        logger.info("Wrote %d rows → %s", len(ranked), path)
