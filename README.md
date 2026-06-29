---
sdk: docker
app_port: 7860
title: Redrob Ranker
emoji: 🎯
colorFrom: blue
colorTo: indigo
pinned: false
---

# Redrob Ranker - VelocityLabs

**INDIA.RUNS Hackathon · Track 01 · Intelligent Candidate Discovery & Ranking**

A multi-signal AI ranking engine that finds the right candidates — not just the keyword-matching ones.

---

## Architecture

![Redrob Ranker system architecture: an unconstrained pre-computation pipeline (LLM JD parsing, embedding, FAISS index build) feeding a constraint-bound CPU-only ranking step (multi-signal fusion, honeypot zero-out, template reasoning) that emits a 100-row submission CSV](architecture.svg)

*Two phases: unconstrained pre-compute (LLM + embeddings + FAISS, run once) → constrained rank step (≤5 min, CPU-only, no LLM, no network) → 100-row CSV.*

<details>
<summary>Same architecture as a text diagram</summary>

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
       └─► RankingEngine (for each of 500):
              ├── Semantic     40%  cosine similarity (FAISS score)
              ├── Role-Fit     20%  title + company-type + location + YoE band + JD disqualifiers
              ├── Skill        15%  proficiency-weighted fuzzy match (RapidFuzz)
              ├── Behavioral   15%  recency + engagement + response + conversion + notice
              └── Career       10%  velocity + stability + progression + hidden-gem
              │
              ├── HoneypotDetector ──► zero-score impossible profiles
              └── ReasoningGenerator ──► template-based 1-2 sentence reasoning
              │
              └──► top-100 ranked CSV
```

</details>

**Final composite = 0.50 × NDCG@10 + 0.30 × NDCG@50 + 0.15 × MAP + 0.05 × P@10 — see submission_spec**

---

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate   # recommended (Python 3.11); Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env   # add your LLM API key (for pre-computation only)
```

> **Platform note:** dependencies install the same way on all OSes (pinned wheels, no build step).
> The command snippets use bash and are tested on **macOS / Linux**. On **Windows**, run them
> under **WSL2**, or skip local setup and use the **Docker** path (Option C) — it's fully
> OS-agnostic and mirrors how the Docker reproduction runs. Only venv activation differs by OS
> (shown inline above).

---

## Dataset

The full candidate dataset (`candidates.jsonl`) is **provided by the hackathon and is not
committed to this repo** — it is not ours to publish (see `.gitignore`). Before running
Step 1, place the provided file at `data/candidates.jsonl` (or pass any path via
`--candidates`). The spec's reproduction command supplies the same file the same way:
`python rank.py --candidates ./candidates.jsonl …`. Only `data/job_description.txt` and a
100-row `data/sample_candidates.json` (for the sandbox demo) ship in the repo.

---

## Step 1 — Pre-compute (run once, no time limit)

Pre-computation has **no time or resource constraints** per the hackathon spec (submission_spec §3 and §10.3). Only the ranking step is constrained.

### Default run

```bash
python precompute.py --candidates data/candidates.jsonl --jd data/job_description.txt
```

Outputs to `data/index/` (all gitignored, except `parsed_jd.json` which is pinned for reproducible ranking — see Step 2):

| File | What it holds / how it's used |
|---|---|
| `candidates.faiss` | The FAISS `IndexFlatIP` — 100K × 768-dim, L2-normalized `bge-base` candidate embeddings (~295 MB). Inner product on normalized vectors = cosine similarity. `rank.py` loads this and runs the ANN search to shortlist the top-500 candidates. |
| `candidate_ids.json` | JSON array of `candidate_id` strings, **row-aligned** with the FAISS index (row *i* ↔ `candidate_ids[i]`). Maps each ANN hit back to its candidate id. |
| `parsed_candidates.jsonl` | Parsed-candidate cache — one internal flat dict per line (redrob schema → normalized fields), exactly one row per id, aligned with the index. `rank.py` loads only the **top-500** records from here, so it never re-parses the raw dataset at rank time. |
| `parsed_jd.json` | The LLM-parsed job description: title, required / nice-to-have skills, min/max experience years, disqualifiers, and a raw summary. Built once during pre-compute (the only LLM call in the pipeline); `rank.py` reads it but never calls the LLM. |

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

> The embedding model (`BAAI/bge-base-en-v1.5`) downloads ~430 MB on first run and is cached in `~/.cache/huggingface/` afterwards.

### Resuming an interrupted run

If the process is killed mid-way, resume exactly where it left off — no re-embedding:

```bash
python precompute.py --candidates data/candidates.jsonl --jd data/job_description.txt \
  --chunk-size 200 --resume
```

---

## Step 2 — Rank (< 5 min, CPU only)

> **Prerequisite:** run **Step 1 (`precompute.py`) first.** `rank.py` loads the pre-built
> FAISS index + `parsed_jd.json` from `data/index/` — it does **not** re-read `--candidates`
> (that flag is accepted for spec-command compatibility only). Pre-computation is
> unconstrained per spec §3 / §10.3; only this step is time/CPU/network limited.

```bash
python rank.py --candidates data/candidates.jsonl --out submission.csv
```

No LLM calls. No network. Loads the pre-built FAISS index from disk and runs in under 5 minutes on CPU.

---

## Step 3 — Validate

```bash
python validate_submission.py --submission submission.csv --candidates data/candidates.jsonl
```

---

## Step 4 — Test in the sandbox (Stage 1)

The sandbox is a self-contained Streamlit app (`scripts/demo_app.py`). It parses the JD with a fast keyword fallback (**no LLM**) and ranks CPU-only via one of two paths:

- **Upload** ≤100 candidates (JSONL or JSON array, redrob schema) → parsed and embedded **in-memory** at runtime (the slow step on free-tier CPU).
- **No upload** → loads a **pre-built 100-candidate sample index** (`sample_index/`), baked into the Docker image at build time by `scripts/build_sample_index.py`, so only the JD is embedded (fast). This mirrors how production `rank.py` loads a pre-built index.

Either way it offers a downloadable CSV with columns `candidate_id,rank,score,reasoning`.

### Prepare a sample input (≤100 candidates)

A `data/sample_candidates.json` (100 candidates) is already committed and is what the Dockerfile bakes into the image. Regenerate it only if you want a different sample (the full dataset is gitignored — not ours to publish):

```bash
head -100 data/candidates.jsonl > data/sample_candidates.json
```

### Option A — Run Streamlit locally (fastest)

```bash
pip install -r requirements.txt
python scripts/build_sample_index.py   # builds sample_index/ for the no-upload path
streamlit run scripts/demo_app.py
```

Open <http://localhost:8501>, paste the JD (contents of `data/job_description.txt`), then either upload `data/sample_candidates.json` or leave the uploader empty to rank the built-in sample. Click **🚀 Run Ranking**, then download `submission.csv`.

### Option B — Deploy to HuggingFace Spaces

1. Create a new Space → **SDK: Docker**.
2. Push this repo; the root `Dockerfile` is used automatically.
3. Commit a `data/sample_candidates.json` (or upload candidates at runtime) — the full `data/candidates.jsonl` is gitignored and must not be published.

### Option C — Docker (mirrors the HuggingFace Spaces runtime)

```bash
docker build -t redrob-ranker .      # bakes the embedding model into the image
docker run -p 7860:7860 redrob-ranker
```

Open <http://localhost:7860>. The image sets `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` and pre-downloads `BAAI/bge-base-en-v1.5` at **build** time, so ranking runs with **no network** — matching the Stage-3 constraints. Port **7860** is the HuggingFace Spaces default, so the same image deploys to a Space unchanged.

### What to verify in the sandbox

- App loads, accepts a JD + ≤100 candidates (or no upload → built-in sample)
- Produces a downloadable CSV with columns `candidate_id,rank,score,reasoning`
- Top-20 preview shows ML/AI engineers with **non-increasing** scores
- Full run completes in **< 5 min on CPU**

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
| Behavioral | 15% | Active, hireable candidates: recency + engagement + response + conversion (offer/interview/GitHub) + notice period |
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
├── redrob_results_view.py # Client-side results view (rendered by demo_app.py)
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
│       ├── behavioral.py  # Recency + engagement + response + conversion + notice
│       ├── career.py      # Velocity + stability + progression + hidden-gem
│       ├── role_fit.py    # Title + company-type + location + YoE + JD disqualifiers
│       └── skill.py       # Proficiency-weighted fuzzy match
├── scripts/
│   ├── build_sample_index.py  # Build sample_index/ for the demo (Docker build / local)
│   └── demo_app.py            # Streamlit sandbox
├── tests/                 # pytest suite: parser, scorers, honeypot, ranker, reasoning, jd
│   ├── conftest.py        # shared fixtures
│   ├── test_candidate_parser.py
│   ├── test_honeypot.py
│   ├── test_jd_parser.py
│   ├── test_ranker.py
│   ├── test_reasoning.py
│   └── test_scorers.py
├── sample_index/          # Pre-built demo index (gitignored; built at image-build time)
└── data/
    ├── job_description.txt     # Provided JD (sample input for the demo)
    ├── sample_candidates.json  # Committed 100-candidate demo sample (baked into image)
    └── index/                  # Pre-computed artifacts (gitignored)
```
