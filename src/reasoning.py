"""Reasoning Generator — produces per-candidate reasoning without any LLM call.

The submission spec (Section 3) requires 1-2 sentence reasoning per candidate.
It penalises: empty, identical, hallucinated (skills not in profile), or
templated (just inserts name).

This module generates reasoning from the candidate's ACTUAL data only.
Every field referenced must come from the parsed candidate dict.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ScoredCandidate:
    candidate_id: str
    rank: int
    score: float
    semantic_score: float
    role_fit_score: float
    skill_score: float
    behavioral_score: float
    career_score: float
    matched_skills: list[str]
    hidden_gem_reasons: list[str]
    is_honeypot: bool
    # From parsed candidate
    current_title: str
    years_of_experience: float
    current_company: str
    current_industry: str
    all_consulting: bool
    notice_period_days: int
    location: str
    country: str
    reasoning: str = ""
    confidence: str = "LOW"


class ReasoningGenerator:
    """Template-based reasoning generator. Zero LLM calls."""

    def generate(self, candidate: dict, sc: ScoredCandidate, parsed_jd) -> str:
        """Build a 1-2 sentence reasoning string grounded in candidate data."""
        yoe = candidate.get("years_of_experience", 0)
        title = candidate.get("current_title", "engineer")
        company = candidate.get("current_company", "")
        matched = sc.matched_skills[:3]
        notice = candidate.get("behavioral_signals", {}).get("notice_period_days", 90)
        beh_signals = candidate.get("behavioral_signals", {})
        open_to_work = beh_signals.get("open_to_work", False)
        response_rate = float(beh_signals.get("recruiter_response_rate", 0))
        loc = candidate.get("location", "")
        country = candidate.get("country", "")
        all_consulting = candidate.get("all_consulting", False)
        product_months = candidate.get("product_company_months", 0)
        gems = sc.hidden_gem_reasons

        # Build sentence 1: strongest positive signal
        s1 = self._sentence1(yoe, title, company, matched, product_months, parsed_jd)

        # Build sentence 2: availability / concern / secondary positive
        s2 = self._sentence2(
            notice, open_to_work, response_rate, loc, country,
            all_consulting, gems, sc.rank, yoe, parsed_jd
        )

        return f"{s1}; {s2}".strip()[:280]

    # ── Sentence builders ─────────────────────────────────────────────────────

    def _sentence1(
        self, yoe: float, title: str, company: str,
        matched: list[str], product_months: int, parsed_jd
    ) -> str:
        skills_str = " + ".join(matched) if matched else "adjacent skills"
        co_str = f" at {company}" if company else ""
        # Phrase prior product-company experience as a career-history signal, never
        # as a property of the *current* company — the current employer may be a
        # consulting firm even when the candidate has prior product stints
        # (e.g. "at Infosys (product-stage company)" was a misattribution).
        product_ctx = " with prior product-company experience" if product_months > 24 else ""

        if matched and yoe >= parsed_jd.min_experience_years:
            return (
                f"{yoe:.0f}yr {title}{co_str}{product_ctx}; "
                f"{skills_str} match JD requirements directly"
            )
        elif parsed_jd.min_experience_years <= yoe <= parsed_jd.max_experience_years:
            return (
                f"{yoe:.0f}yr {title}{co_str}; "
                f"experience band aligns ({parsed_jd.min_experience_years}-{parsed_jd.max_experience_years}yr range)"
            )
        elif matched:
            return (
                f"{title}{co_str} with {skills_str}; "
                f"YoE ({yoe:.0f}yr) below target but skill alignment is strong"
            )
        else:
            return (
                f"{yoe:.0f}yr {title} — adjacent profile included for long-tail coverage"
            )

    def _sentence2(
        self, notice: int, open_to_work: bool, response_rate: float,
        loc: str, country: str, all_consulting: bool,
        gems: list[str], rank: int, yoe: float, parsed_jd
    ) -> str:
        concerns: list[str] = []
        positives: list[str] = []

        if notice <= 30:
            positives.append(f"notice {notice}d")
        elif notice > 90:
            concerns.append(f"long notice ({notice}d)")

        if open_to_work:
            positives.append("actively open to roles")

        if response_rate >= 0.5:
            positives.append(f"high response rate ({response_rate:.0%})")
        elif response_rate < 0.10 and response_rate >= 0:
            concerns.append(f"low response rate ({response_rate:.0%})")

        if country and country.lower() in ("india", "in"):
            if loc:
                positives.append(f"{loc}-based")
        elif country:
            concerns.append(f"outside India ({country})")

        if all_consulting:
            concerns.append("consulting-only career history")

        if "open_source" in gems:
            positives.append("open-source contributor")
        if "multi_promotion" in gems:
            positives.append("multi-promotion trajectory")

        if yoe < parsed_jd.min_experience_years:
            concerns.append(f"under target YoE ({yoe:.0f} vs {parsed_jd.min_experience_years}+)")

        if rank >= 80:
            concerns.append("borderline fit — included as long-tail filler")

        if positives and not concerns:
            return ", ".join(positives[:3]) + "."
        if concerns and not positives:
            return "Concern: " + "; ".join(concerns[:2]) + "."
        if positives and concerns:
            return ", ".join(positives[:2]) + f"; note: {concerns[0]}."
        return "included based on composite signal score."
