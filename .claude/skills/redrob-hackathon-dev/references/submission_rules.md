# Submission Rules — Redrob Hackathon v4

## Required output file

**Columns (exact order):** `candidate_id,rank,score,reasoning`
**Rows:** exactly 100 data rows + 1 header = 101 lines total
**Encoding:** UTF-8
**Filename:** `<participant_id>.csv` (e.g. `team_velocitylabs.csv`)

## Hard format rules (auto-rejected if violated)

| Rule | Detail |
|---|---|
| Exactly 100 rows | No 99, no 101 |
| Ranks 1-100, each exactly once | No duplicates, no 0-based ranks |
| score non-increasing | score[rank=1] >= score[rank=2] >= ... >= score[rank=100] |
| Ties: unique ranks required | Break ties by candidate_id ascending |
| All candidate_ids exist in candidates.jsonl | No typos, no invented IDs |
| No duplicate candidate_ids | Each CAND_XXXXXXX once only |
| Score column not all-same | Model must differentiate candidates |

## Common auto-rejections

- 99 or 101 rows
- Ranks starting at 0
- Duplicate candidate_ids
- candidate_id not in candidates.jsonl
- All scores identical
- Scores increasing with rank (rank 1 has lowest score)
- File submitted as .xlsx or .json

## Compute constraints (ranking step ONLY)

rank.py must complete in <=5 min on 16 GB CPU machine with:
- No GPU
- No hosted LLM API calls (Anthropic, OpenAI, Gemini, etc.)
- No network access

precompute.py has NO time/RAM/network constraints. LLM calls allowed there.

Enforced at Stage 3: Docker reproduction of rank.py exactly.

## Reasoning column

Penalised: empty, all-identical, templated (name-only), hallucinated skills, contradicts rank.
Good: specific, grounded in actual candidate data, 1-2 sentences, varied.

## Scoring formula

Final = 0.50*NDCG@10 + 0.30*NDCG@50 + 0.15*MAP + 0.05*P@10

NDCG@10 = 50% of score. Top-10 quality matters most.
Tiebreaks: P@5 -> P@10 -> earlier submission timestamp.

## Honeypot rule

~80 honeypots with impossible profiles in dataset. If >10% of top-100 are honeypots: DISQUALIFIED.

Honeypot signals:
- YoE > 1.35x sum(career months/12)
- "expert" skill with <6 months usage
- >8 expert skills in <60-month total career
- Disqualifying title (HR/Marketing/etc.) with 5+ advanced AI skills

## Evaluation stages

1. Format validation + sandbox check
2. Automated NDCG/MAP/P@10 scoring
3. Top-N code reproduction in Docker (rank.py only, CPU-only)
4. Manual review of 10 sampled reasoning rows
5. Defend-your-work interview

## Submission cap

3 submissions max. Last valid counts. No live leaderboard.

## Required artefacts

1. <participant_id>.csv (ranked output)
2. GitHub repo: README.md, full source, requirements.txt, submission_metadata.yaml
3. Sandbox link (HuggingFace Spaces / Streamlit Cloud / Colab / Docker)
4. Portal metadata

Single reproduction command in README:
  python rank.py --candidates ./candidates.jsonl --out ./submission.csv
