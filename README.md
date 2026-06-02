# Redrob Ranker — Velocity Labs

**INDIA.RUNS Hackathon · Track 01 · Intelligent Candidate Discovery & Ranking**

A multi-signal AI ranking engine that finds the right candidates — not just the keyword-matching ones.

---

## Architecture

```
PRE-COMPUTATION (no time limit, run once)
─────────────────────────────────────────
candidates.jsonl ──► CandidateParser ──► 100K parsed dicts
                                              │
job_description.txt ──► LLM JDParser ──►  ParsedJD ──► parsed_jd.json
                                              │
                      Embedder (bge-base) ──► 100K × 768 float32 embeddings
                                              │
                             FAISS IndexFlatIP ──► candidates.faiss
                                                   candidate_ids.json

RANKING STEP (<5 min, CPU only, no LLM, no network)
────────────────────────────────────────────────────
FAISS index + parsed_jd.json (from disk)
       │
       ├─► Embed JD ──► ANN search ──► top-500 candidates
       │
       └─► MultiSignalRanker (for each of 500):
              ├── Semantic     40%  cosine similarity (FAISS score)
              ├── Role-Fit     20%  title + company-type + location + YoE band
              ├── Skill        15%  proficiency-weighted fuzzy match (RapidFuzz)
              ├── Behavioral   15%  recency decay + response rate + notice period
              └── Career       10%  velocity + stability + progression + hidden-gem
              │
              ├── HoneypotDetector ──► zero-score impossible profiles
              └── ReasoningGenerator ──► template-based 1-2 sentence reasoning
              │
              └──► top-100 ranked CSV
```

**Final composite = 0.40 × NDCG@10 + ... see submission_spec**

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # add your LLM API key (for pre-computation only)
```

---

## Step 1 — Pre-compute (run once, no time limit)

Pre-computation has **no time or resource constraints** per the hackathon spec (submission_spec §3 and §10.3). Only the ranking step is constrained.

### Default run

```bash
python precompute.py --candidates data/candidates.jsonl --jd data/job_description.txt
```

Outputs to `data/index/`: `candidates.faiss`, `candidate_ids.json`, `parsed_candidates.jsonl`, `parsed_jd.json`

### Tuning for lower RAM usage

The script streams candidates in chunks so peak RAM stays manageable (~600 MB at the default chunk size). Use `--chunk-size` if you need to reduce memory pressure further:

| `--chunk-size` | Peak RAM | Approx. time (MacBook CPU) |
|---|---|---|
| 500 (default) | ~700 MB | ~20–25 min |
| 200 | ~500 MB | ~25–30 min |
| 100 | ~450 MB | ~30–35 min |

```bash
# Lower memory footprint
python precompute.py --candidates data/candidates.jsonl --jd data/job_description.txt \
  --chunk-size 200

# Minimum footprint (slowest)
python precompute.py --candidates data/candidates.jsonl --jd data/job_description.txt \
  --chunk-size 100
```

> **Tip:** Close unused browser tabs and apps before running. The embedding model (`BAAI/bge-base-en-v1.5`) downloads ~430 MB on first run and is cached in `~/.cache/huggingface/` afterwards.

### Resuming an interrupted run

If the process is killed mid-way, resume exactly where it left off — no re-embedding:

```bash
python precompute.py --candidates data/candidates.jsonl --jd data/job_description.txt \
  --chunk-size 200 --resume
```

---

## Step 2 — Rank (< 5 min, CPU only)

```bash
python rank.py --candidates data/candidates.jsonl --jd data/job_description.txt --out submission.csv
```

No LLM calls. No network. Loads the pre-built FAISS index from disk and runs in under 5 minutes on CPU.

---

## Step 3 — Validate

```bash
python validate_submission.py --submission submission.csv --candidates data/candidates.jsonl
```

---

## Key design decisions

**Why FAISS over ChromaDB?** FAISS is a single binary with no server process — it loads from disk in under 1 second and runs fully in-process. Critical for the sandboxed Docker reproduction at Stage 3.

**Why no LLM during ranking?** The spec forbids hosted API calls in the ranking step. Reasoning is generated from candidate data via templates — specific, non-hallucinated, and varied across ranks.

**Why role_fit over pure semantic?** The JD explicitly warns against keyword-matching. A `Marketing Manager` listing AI skills scores 0 on role_fit and never reaches the top 100, even with high semantic similarity.

**Honeypot detection:** Two or more consistency signals (YoE vs career timeline, expert skills with < 6 months usage, etc.) → composite score set to 0. This keeps the honeypot rate well below the 10% disqualification threshold.

---

## Scoring weights rationale

| Signal | Weight | Why |
|--------|--------|-----|
| Semantic similarity | 40% | Deep JD-profile understanding; captures implicit fit |
| Role-fit | 20% | Hard structural filter; prevents keyword-stuffer inflation |
| Skill depth | 15% | Proficiency + duration beats binary presence/absence |
| Behavioral | 15% | Active candidates with low notice period actually hire |
| Career trajectory | 10% | Hidden-gem detection; fast-trackers undervalued by keyword search |

---

## Repo structure

```
redrob-ranker/
├── precompute.py          # Step 1: build index (no time limit)
├── rank.py                # Step 2: ranking (<5 min, CPU, no LLM)
├── validate_submission.py # Step 3: local validation
├── submission_metadata.yaml
├── requirements.txt
├── Dockerfile             # Sandbox (Streamlit demo)
├── src/
│   ├── config.py          # All weights and constants
│   ├── embedder.py        # SentenceTransformer wrapper
│   ├── index.py           # FAISS build/load/query
│   ├── ranker.py          # Orchestration engine
│   ├── honeypot.py        # Profile consistency checks
│   ├── reasoning.py       # Template reasoning (no LLM)
│   ├── parsers/
│   │   ├── candidate.py   # redrob schema → internal dict
│   │   └── jd.py          # LLM JD extraction (pre-compute only)
│   └── scorers/
│       ├── behavioral.py  # Recency decay + engagement + notice
│       ├── career.py      # Velocity + stability + hidden-gem
│       ├── role_fit.py    # Title + company-type + location + YoE
│       └── skill.py       # Proficiency-weighted fuzzy match
├── scripts/
│   └── demo_app.py        # Streamlit sandbox
└── data/
    └── index/             # Pre-computed artifacts (gitignored)
```
