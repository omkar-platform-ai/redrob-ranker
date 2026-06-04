# Codebase Map — redrob-ranker

Repo root: ~/Library/CloudStorage/OneDrive-Henkel/Personal/Hackathon/INDIA.RUNS/redrob-ranker/

## Entry points

| Script | Purpose | Constraints |
|---|---|---|
| precompute.py | Build FAISS index + parse JD | No time limit; LLM + network OK |
| rank.py | Produce submission.csv | <=5 min, CPU, no LLM, no network |
| validate_submission.py | Check CSV against spec | Run before every upload |

## Source modules (src/)

| File | Role |
|---|---|
| config.py | All weights, thresholds, constants |
| embedder.py | SentenceTransformer wrapper (BAAI/bge-base-en-v1.5) |
| index.py | FAISS IndexFlatIP build/load/query |
| ranker.py | Orchestrates all scorers → ranked list → CSV |
| honeypot.py | Profile consistency checks → score=0 |
| reasoning.py | Template-based reasoning (no LLM) |
| parsers/candidate.py | Maps redrob schema → internal dict |
| parsers/jd.py | LLM JD extraction (pre-compute only) |
| scorers/behavioral.py | Recency decay + engagement + response_rate + notice |
| scorers/career.py | Velocity + stability + progression + hidden-gem |
| scorers/role_fit.py | Title + company-type + location + YoE band |
| scorers/skill.py | Proficiency × duration fuzzy match (RapidFuzz) |

## Data files

| Path | Description |
|---|---|
| data/candidates.jsonl | 100K candidate records (symlink to bundle) |
| data/job_description.txt | Raw JD text (extracted from bundle .docx) |
| data/index/candidates.faiss | Pre-built FAISS index (gitignored) |
| data/index/candidate_ids.json | ID list mapping FAISS row → candidate_id |
| data/index/parsed_candidates.jsonl | Parsed + normalised candidates (gitignored) |
| data/index/parsed_jd.json | Parsed JD from LLM (gitignored) |

## Candidate schema (redrob format)

Top-level keys: candidate_id, profile, career_history, education, skills, redrob_signals

profile keys: anonymized_name, headline, summary, location, country,
              years_of_experience, current_title, current_company,
              current_company_size, current_industry

career_history[]: company, title, start_date, end_date, duration_months,
                  is_current, industry, company_size, description

skills[]: name, proficiency (beginner/intermediate/advanced/expert),
          endorsements, duration_months

redrob_signals keys: profile_completeness_score, signup_date, last_active_date,
  open_to_work_flag, profile_views_received_30d, applications_submitted_30d,
  recruiter_response_rate, avg_response_time_hours, skill_assessment_scores,
  connection_count, endorsements_received, notice_period_days,
  expected_salary_range_inr_lpa, preferred_work_mode, willing_to_relocate,
  github_activity_score, search_appearance_30d, saved_by_recruiters_30d,
  interview_completion_rate, offer_acceptance_rate, verified_email,
  verified_phone, linkedin_connected

## Submission CSV format

Columns (exact order): candidate_id, rank, score, reasoning
100 rows, UTF-8, non-increasing score, ranks 1-100 each exactly once

## Key config constants (src/config.py)

WEIGHTS = {semantic:0.40, role_fit:0.20, skill:0.15, behavioral:0.15, career:0.10}
TOP_K_RETRIEVE = 500   # ANN retrieval candidates
TOP_K_FINAL    = 100   # submission rows
RECENCY_DECAY_LAMBDA = 0.023
OUTPUT_COLUMNS = ["candidate_id", "rank", "score", "reasoning"]
