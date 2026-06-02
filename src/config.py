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

# ── Location scorer ───────────────────────────────────────────────────────────
INDIA_TIER1_CITIES: frozenset[str] = frozenset({
    "pune", "noida", "hyderabad", "mumbai", "bangalore", "bengaluru",
    "delhi", "gurugram", "gurgaon", "chennai", "kolkata",
})

# ── Honeypot detection ────────────────────────────────────────────────────────
HONEYPOT_YOE_CAREER_RATIO_THRESHOLD: float = 1.35  # if claimed YoE > 1.35× sum(career months/12)
HONEYPOT_EXPERT_MIN_MONTHS: int = 6              # "expert" with < 6 months → flag
HONEYPOT_EXPERT_SKILL_COUNT_LIMIT: int = 8       # >8 "expert" skills in <60 months total career

# ── Confidence thresholds ─────────────────────────────────────────────────────
CONFIDENCE_HIGH: float = 0.72
CONFIDENCE_MEDIUM: float = 0.50

# ── Output ────────────────────────────────────────────────────────────────────
OUTPUT_COLUMNS: list[str] = ["candidate_id", "rank", "score", "reasoning"]
