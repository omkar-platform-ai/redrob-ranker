---
name: redrob-hackathon-dev
description: >
  Full development workflow for the Redrob INDIA.RUNS hackathon challenge
  "Intelligent Candidate Discovery & Ranking". Use this skill whenever working
  on the redrob-ranker codebase to: (1) BUILD new scoring logic or pipeline
  components, (2) TEST and validate the submission CSV against the hackathon
  spec, (3) REVIEW code for rule compliance (compute constraints, output format,
  honeypot handling), (4) FIX bugs in ranking, parsing, or scoring modules,
  (5) VERIFY the full pipeline runs end-to-end within constraints.
  Triggers on: "build ranker", "fix scoring", "validate submission", "check
  compliance", "review ranking code", "test pipeline", "debug rank.py",
  "improve NDCG", "tune weights", "check honeypots", "submission format",
  "precompute error", "ranking error", or any task touching the redrob-ranker
  codebase files.
---

# Redrob Hackathon Dev Skill

## Context

**Challenge:** Intelligent Candidate Discovery & Ranking (INDIA.RUNS by Redrob AI)  
**Team:** Velocity Labs  
**Repo root:** `~/...OneDrive-Henkel/Personal/Hackathon/INDIA.RUNS/redrob-ranker/`  
**Bundle:** `~/...INDIA.RUNS/India_runs_data_and_ai_challenge/`

Read `references/codebase_map.md` for the full file/module map and candidate schema.  
Read `references/submission_rules.md` for every spec rule and disqualification condition.  
Read `references/jd_scoring_guide.md` for what makes a good candidate for this JD.

---

## Two-step pipeline (never mix the two)

```
precompute.py   →  LLM + embeddings + FAISS build  (no time limit, network OK)
rank.py         →  pure CPU math, no LLM, no network, <=5 min
```

Any LLM call, network call, or GPU use inside `rank.py` = **Stage 3 disqualification**.

---

## BUILD — adding or changing features

1. Scoring weights live in `src/config.py` `WEIGHTS` dict. All changes go there.
2. One class per signal: `scorers/behavioral.py`, `scorers/career.py`, `scorers/role_fit.py`, `scorers/skill.py`.
3. `src/ranker.py` orchestrates — do not add business logic there.
4. New signals must return a float in [0.0, 1.0].
5. After any change to scoring, re-run `rank.py` and spot-check top-10 manually — they must be ML/AI engineers at product companies, active on platform, India-based.

**Critical JD insight:** Career descriptions + title + company_type beat skill keywords. A "Marketing Manager" with AI skills is NOT a fit. A "Backend Engineer" who built recommendation systems IS. See `references/jd_scoring_guide.md`.

---

## TEST — verifying the submission CSV

Always run the local validator before uploading:
```bash
python .claude/skills/redrob-hackathon-dev/scripts/validate_local.py \
    --submission submission.csv \
    --candidates data/candidates.jsonl
```

Or use the repo-level validator:
```bash
python validate_submission.py --submission submission.csv --candidates data/candidates.jsonl
```

**Manual top-10 check** — open submission.csv and verify ranks 1-10:
- All should be ML/AI/data engineers (NOT HR, marketing, accountants, civil/mech engineers)
- All should be at product companies (NOT TCS/Infosys/Wipro/Accenture/consulting-only)
- Years of experience should be in the 4-10 year range
- `reasoning` column must mention skills/roles actually in the candidate's profile

**Honeypot rate check:** grep the top-100 candidate_ids against known impossible profiles.
If >10 of the top-100 have impossible YoE-vs-career math → the ranker is broken.

---

## REVIEW — compliance checklist

Before any commit or submission, verify all of:

### Output format
- [ ] Columns exactly: `candidate_id,rank,score,reasoning` in that order
- [ ] Exactly 100 data rows
- [ ] Ranks 1–100, each exactly once
- [ ] Scores non-increasing (rank 1 highest, rank 100 lowest)
- [ ] No duplicate candidate_ids
- [ ] All candidate_ids exist in candidates.jsonl
- [ ] No empty reasoning strings
- [ ] Reasoning strings varied (not templated)

### Compute constraints (rank.py)
- [ ] No `import anthropic` / `import openai` / `import google.generativeai` at rank time
- [ ] No `requests.get` / `httpx` / any HTTP call in ranking path
- [ ] No `torch.cuda` / `device='cuda'` / GPU usage
- [ ] FAISS index loaded from disk (not rebuilt at rank time)
- [ ] Embedding model loaded once, not per-candidate
- [ ] Full run completes in <5 min locally

### GitHub repo
- [ ] `.env` is gitignored (no API keys committed)
- [ ] `data/index/` is gitignored (no large artifacts committed)
- [ ] `data/candidates.jsonl` is gitignored
- [ ] `submission_metadata.yaml` at repo root with participant_id filled
- [ ] Single reproduction command in README

---

## FIX — common bugs and their solutions

### "score non-increasing violated"
Cause: score computation has randomness or floating-point inconsistency.
Fix: sort by score descending, then by candidate_id ascending as tiebreak.
In `src/ranker.py`: `scored.sort(key=lambda x: (-x.score, x.candidate_id))`

### "candidate_id not in dataset"
Cause: parsing or ID munging error.
Fix: ensure `candidate_id` is taken directly from `raw["candidate_id"]` without transformation.
Valid format: `CAND_XXXXXXX` (7 digits).

### "all behavioral scores are 0 / identical"
Cause: field name mismatch — old code uses `last_active_days` but schema has `last_active_date`.
Fix: in `src/parsers/candidate.py`, convert `redrob_signals["last_active_date"]` string to days:
```python
from datetime import date
from dateutil.parser import parse as parse_date
last_active_days = (date.today() - parse_date(signals["last_active_date"]).date()).days
```

### "JD parsed with only 2-3 required skills"
Cause: LLM returned markdown-fenced JSON (```json ... ```); parser fails.
Fix: strip fences in `src/parsers/jd.py` `_parse_llm_response()`:
```python
if cleaned.startswith("```"):
    cleaned = cleaned.split("\n", 1)[1]
    if cleaned.rstrip().endswith("```"):
        cleaned = cleaned.rstrip()[:-3].rstrip()
```

### "precompute.py runs out of memory / crashes Mac"
Use chunked streaming: `--chunk-size 200` (default 500). Each chunk is parsed,
embedded, added to FAISS, then freed. Peak RAM ~500 MB at chunk-size=200.
Use `--resume` to continue from last completed chunk if interrupted.

### "rank.py hangs or exceeds 5 minutes"
Check: is the embedding model being reloaded per candidate? Should be loaded once.
Check: is `TOP_K_RETRIEVE` set too high? Default 500 is fine; 5000+ will be slow.
Check: is there a network call hiding somewhere? Search for `requests`, `httpx`, `urllib`.

### "too many honeypots in top-100"
The honeypot detector in `src/honeypot.py` requires 2+ signals to flag.
If honeypots are slipping through, lower threshold to 1 signal:
```python
is_honeypot = len(reasons) >= 1  # was >= 2
```
Or strengthen role_fit scorer — if disqualifying title check works correctly,
most honeypots never reach the top 100.

---

## TUNE — improving NDCG@10

NDCG@10 = 50% of score. Optimise the top-10, not the full 100.

**Levers (in order of impact):**

1. **Role-fit weight**: increase from 0.20 → 0.25 if too many non-engineers in top-10.
   Decrease semantic weight to compensate.
2. **Consulting-firm multiplier**: tighten from 0.25 → 0.10 if consulting candidates appear.
3. **Recency decay lambda**: increase from 0.023 → 0.035 if inactive candidates rank high.
4. **Skill proficiency weights**: "expert" = 1.0 is correct; verify "advanced" = 0.85.
5. **Location score**: bump India tier-1 from 1.0 → 1.0 + 0.05 bonus in composite.

Always spot-check after any weight change:
- Does top-10 still look like ML engineers at product companies?
- Are scores still non-increasing?
- Is reasoning still accurate (no hallucinated skills)?

---

## SANDBOX — HuggingFace Spaces deployment

The sandbox is required for Stage 1. It must:
- Accept ≤100 candidates as input (upload or pre-loaded sample)
- Run the full ranking pipeline end-to-end
- Produce a downloadable ranked CSV
- Complete in <5 min on CPU

Demo app: `scripts/demo_app.py` (Streamlit)
Dockerfile: `Dockerfile` at repo root

Deploy:
```bash
# HuggingFace Spaces (free tier)
# Create a new Space, choose Streamlit, push repo
# Or use the Dockerfile directly:
docker build -t redrob-ranker .
docker run -p 8501:8501 redrob-ranker
```
