"""Central config — all tunable constants in one place.

No secrets here. Keys live in .env only.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
INDEX_DIR = DATA_DIR / "index"

FAISS_INDEX_PATH = INDEX_DIR / "candidates.faiss"
CANDIDATE_IDS_PATH = INDEX_DIR / "candidate_ids.json"
PARSED_JD_PATH = INDEX_DIR / "parsed_jd.json"

# ── LLM (pre-computation only — NEVER called during ranking step) ─────────────
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "anthropic")
ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")
GEMINI_MODEL: str = "gemini-1.5-flash"

# ── Embeddings ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIM: int = 768
EMBED_BATCH_SIZE: int = 256

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K_RETRIEVE: int = 500   # ANN candidates before scoring
TOP_K_FINAL: int = 100      # Final submission rows (spec: exactly 100)

# ── Ranking weights (must sum to 1.0) ─────────────────────────────────────────
WEIGHTS: dict[str, float] = {
    "semantic":   0.40,   # embedding cosine similarity
    "role_fit":   0.20,   # title/industry/company-type match
    "skill":      0.15,   # proficiency-weighted skill overlap
    "behavioral": 0.15,   # recency + engagement + response rate
    "career":     0.10,   # trajectory, stability, hidden-gem bonus
}

# ── Behavioral scorer ─────────────────────────────────────────────────────────
RECENCY_DECAY_LAMBDA: float = 0.023   # exp(-λ × days): 30d → ~0.50, 90d → ~0.13
RECENCY_WEIGHT: float = 0.45
ENGAGEMENT_WEIGHT: float = 0.30
RESPONSE_RATE_WEIGHT: float = 0.15
NOTICE_WEIGHT: float = 0.10

# ── Career scorer ─────────────────────────────────────────────────────────────
CAREER_VELOCITY_WEIGHT: float = 0.40
CAREER_STABILITY_WEIGHT: float = 0.25
CAREER_PROGRESSION_WEIGHT: float = 0.35

HIDDEN_GEM_BONUSES: dict[str, float] = {
    "open_source":     0.06,
    "side_projects":   0.04,
    "publications":    0.05,
    "multi_promotion": 0.05,
}
HIDDEN_GEM_CAP: float = 0.15

# ── Consulting-firm penalty ───────────────────────────────────────────────────
# Candidates whose ENTIRE career is at these firms get a career multiplier.
CONSULTING_FIRMS: frozenset[str] = frozenset({
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra", "mphasis",
    "hexaware", "niit technologies", "ltimindtree", "l&t infotech",
})
CONSULTING_ONLY_MULTIPLIER: float = 0.25

# ── Skill scorer ─────────────────────────────────────────────────────────────
SKILL_PROFICIENCY_WEIGHTS: dict[str, float] = {
    "beginner": 0.3,
    "intermediate": 0.6,
    "advanced": 0.85,
    "expert": 1.0,
}
SKILL_REQUIRED_WEIGHT: float = 0.75
SKILL_OPTIONAL_WEIGHT: float = 0.25
SKILL_DURATION_CAP_MONTHS: int = 24   # score maxes at 2 years of use
# Weight on Redrob's measured `assessment_score` when a skill has one (0 = ignore
# the objective score, 1 = use it exclusively). 0.5 gives objective proficiency
# equal say with self-declared proficiency + duration + endorsements.
SKILL_ASSESSMENT_BLEND: float = 0.50

# ── Role-fit scorer ───────────────────────────────────────────────────────────
# Titles that signal genuine ML/AI engineering background
AI_ENGINEER_TITLES: frozenset[str] = frozenset({
    "ml engineer", "machine learning engineer", "ai engineer", "applied scientist",
    "nlp engineer", "data scientist", "research engineer", "software engineer",
    "backend engineer", "platform engineer", "search engineer", "ranking engineer",
    "recommendation engineer", "mlops engineer", "data engineer",
})
# Titles that are strong negative signals for this JD
DISQUALIFYING_TITLES: frozenset[str] = frozenset({
    "hr manager", "hr executive", "recruiter", "talent acquisition",
    "marketing manager", "content writer", "graphic designer",
    "accountant", "civil engineer", "mechanical engineer",
    "sales executive", "business development",
})
# Industries that map to product-company context (positive signal)
PRODUCT_INDUSTRIES: frozenset[str] = frozenset({
    "technology", "saas", "fintech", "edtech", "healthtech", "e-commerce",
    "internet", "software", "ai", "ml", "data", "startup",
})
# Industries that are pure services (negative signal)
SERVICES_INDUSTRIES: frozenset[str] = frozenset({
    "it services", "consulting", "staffing", "outsourcing", "bpo",
})

# YoE-band penalties (role-fit sub-score). The JD band is min..max years.
# Under-floor candidates are penalised steeply — a sub-5yr profile is a weaker
# fit for a 5-9yr senior role and should not sit in the top-10 on skills alone.
YOE_UNDER_PENALTY_PER_YEAR: float = 0.35   # slope below min band (was 0.15 — too soft)
YOE_OVER_PENALTY_PER_YEAR: float = 0.08    # slope above max band
YOE_OVER_PENALTY_FLOOR: float = 0.40       # over-qualified never scores below this

# ── Location scorer ───────────────────────────────────────────────────────────
INDIA_TIER1_CITIES: frozenset[str] = frozenset({
    "pune", "noida", "hyderabad", "mumbai", "bangalore", "bengaluru",
    "delhi", "gurugram", "gurgaon", "chennai", "kolkata",
})
# Tier-2 Indian tech hubs — score between Tier-1 and the rest. Variants included
# for substring matching (kochi/cochin, vizag/visakhapatnam, etc.).
INDIA_TIER2_CITIES: frozenset[str] = frozenset({
    "jaipur", "indore", "coimbatore", "chandigarh", "mohali", "kochi", "cochin",
    "ahmedabad", "nagpur", "lucknow", "bhubaneswar", "trivandrum",
    "thiruvananthapuram", "vizag", "visakhapatnam", "vadodara", "surat",
    "mysore", "mysuru", "bhopal", "nashik", "mangalore", "mangaluru",
    "vijayawada", "madurai",
})

# Location sub-score values (role-fit). India hub proximity scores highest;
# willingness to relocate lifts non-Tier-1 domestic candidates toward the hub
# (previously their relocation flag was ignored entirely). The tier/relocation
# tier is monotonic — relocation only ever raises it.
LOCATION_TIER1: float = 1.00
LOCATION_TIER2_RELOCATE: float = 0.90
LOCATION_TIER2: float = 0.85
LOCATION_INDIA_OTHER_RELOCATE: float = 0.85
LOCATION_INDIA_OTHER: float = 0.80      # unchanged baseline
LOCATION_ABROAD_RELOCATE: float = 0.60  # unchanged
LOCATION_ABROAD: float = 0.30           # unchanged
# Composite-level suppression for abroad candidates UNWILLING to relocate: they are
# effectively unhireable for this India-hybrid role, and the role-fit location tier
# above (LOCATION_ABROAD × the 20% role_fit weight) alone cannot keep them out of the
# top ranks. Abroad + willing-to-relocate and all India tiers are untouched.
LOCATION_INELIGIBLE_MULTIPLIER: float = 0.50

# Work-mode fit for a HYBRID Pune/Noida role. hybrid/onsite/flexible all accept
# hub presence; remote-only is a softer fit. Applied as a multiplier on the
# location sub-score (default 1.0 for unknown values — never penalise missing data).
WORK_MODE_FIT: dict[str, float] = {
    "hybrid": 1.00,
    "onsite": 1.00,
    "flexible": 1.00,
    "remote": 0.85,
}

# ── Honeypot detection ────────────────────────────────────────────────────────
HONEYPOT_YOE_CAREER_RATIO_THRESHOLD: float = 1.35  # if claimed YoE > 1.35× sum(career months/12)
HONEYPOT_EXPERT_MIN_MONTHS: int = 6              # "expert" with < 6 months → flag
HONEYPOT_EXPERT_SKILL_COUNT_LIMIT: int = 8       # >8 "expert" skills in <60 months total career

# ── Confidence thresholds ─────────────────────────────────────────────────────
CONFIDENCE_HIGH: float = 0.72
CONFIDENCE_MEDIUM: float = 0.50

# ── Output ────────────────────────────────────────────────────────────────────
OUTPUT_COLUMNS: list[str] = ["candidate_id", "rank", "score", "reasoning"]
