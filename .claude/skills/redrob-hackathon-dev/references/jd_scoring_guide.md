# JD Scoring Guide — What Makes a Good Candidate

## The role: Senior AI Engineer, Redrob AI

Location: Pune/Noida, India (hybrid). Open to Hyderabad, Mumbai, Delhi NCR.
Experience: 5-9 years (sweet spot: 6-8 years applied ML/AI at product companies).

## MUST-HAVE signals (true fit)

- Production embeddings/retrieval system deployed to real users
- Product company background (NOT consulting-only entire career)
- Strong Python (code quality matters)
- Hands-on evaluation framework experience (NDCG, MRR, MAP, A/B)
- Vector database / hybrid search experience in production
- Active on platform: open_to_work_flag=true, last_active < 60 days, recruiter_response_rate > 0.2

## NICE-TO-HAVE signals

- LLM fine-tuning (LoRA, QLoRA, PEFT)
- Learning-to-rank (XGBoost-based or neural)
- HR-tech / marketplace / recommendation system background
- Open-source contributions (github_activity_score >= 30)

## HARD DISQUALIFIERS (score these near 0)

- Consulting-only career (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini, HCL,
  Tech Mahindra, Mphasis, Hexaware) with NO product company experience
- Pure researcher (academic labs, no production deployment)
- AI experience = LangChain tutorials only, <12 months, no prior ML
- Senior engineer who hasn't written code in 18+ months
- Computer vision / speech / robotics only (no NLP/IR)
- Current title is HR, Marketing, Content Writer, Accountant, Sales, Civil/Mech Engineer
  (regardless of skill list)

## LOCATION scoring

Tier/relocation base (relocation never lowers a score; it lifts non-Tier-1
domestic candidates toward the Pune/Noida hub):

| Segment | not willing | willing_to_relocate |
|---|---|---|
| India, Tier-1 (Pune/Noida/Bengaluru/Hyderabad/Mumbai/Delhi NCR/Chennai/Kolkata) | 1.00 | 1.00 |
| India, Tier-2 (Jaipur/Indore/Coimbatore/Chandigarh/Kochi/Ahmedabad/Nagpur/Lucknow/Bhubaneswar/…) | 0.85 | 0.90 |
| India, other / Tier-3 | 0.80 | 0.85 |
| Outside India | 0.30 | 0.60 |

The base is then multiplied by the work-mode fit below.

## WORK MODE scoring

The role is **hybrid** (Pune/Noida). `preferred_work_mode` scales the location score:

- hybrid / onsite / flexible: ×1.00 (accept hub presence)
- remote: ×0.85 (softer fit for a hybrid role)
- unknown/missing: ×1.00 (never penalise missing data)

## NOTICE PERIOD scoring

- <=30 days: 1.0 (strong preference)
- <=60 days: 0.75
- <=90 days: 0.50
- >90 days: 0.30

## BEHAVIORAL SIGNALS (from redrob_signals)

Key fields and their meaning:
- last_active_date: convert to days, apply exp(-0.023*days) decay
  (today=1.0, 30d=0.50, 90d=0.13, 180d=0.017)
- open_to_work_flag: direct intent signal
- recruiter_response_rate: strongest proxy for actual hirability
- applications_submitted_30d: engagement signal
- notice_period_days: availability signal
- github_activity_score: -1=no GitHub; >=30=open-source contributor
- interview_completion_rate: reliability signal

## HONEYPOT detection

A candidate is a honeypot if 2+ of:
- profile.years_of_experience > 1.35 * sum(career_history[].duration_months) / 12
- Any skill has proficiency="expert" and duration_months < 6
- Count of proficiency="expert" skills > 8 AND total_career_months < 60
- current_title in disqualifying list AND count of advanced AI skills >= 5

Set score = 0.0 for confirmed honeypots.

## KEYWORD TRAP (the main evaluation trap)

The ground truth DOES NOT reward keyword-matching. A Marketing Manager who lists
"RAG, Pinecone, FAISS" as skills is NOT a good match. The JD explicitly warns:
"The right answer involves reasoning about the gap between what the JD says and
what the JD means."

Use career_history descriptions and title/industry/company_type as primary signals.
Skill list is secondary and must be validated against proficiency + duration.

## SCORING WEIGHTS (current config)

Semantic:   0.40  (embedding cosine - BAAI/bge-base-en-v1.5)
Role-fit:   0.20  (title + company_type + location + YoE band)
Skill:      0.15  (proficiency-weighted fuzzy match via RapidFuzz)
Behavioral: 0.15  (recency decay + response_rate + notice + engagement)
Career:     0.10  (velocity + stability + progression + hidden-gem bonus)
