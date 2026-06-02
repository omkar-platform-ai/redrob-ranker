# CLAUDE.md

Multi-signal AI candidate ranking engine for the **INDIA.RUNS Hackathon (Track 01)**.
Ranks 100K candidates against a job description and emits the top 100 as CSV.

## The one constraint that shapes everything

The pipeline is split into two phases by the hackathon spec (`submission_spec` §3, §10.3):

- **Pre-computation** (`precompute.py`) — *no* time/RAM/network limit. Runs the LLM, the embedding model, and builds the FAISS index. Run once.
- **Ranking** (`rank.py`) — must finish in **< 5 min on a 16GB CPU box**, with **no LLM API calls, no GPU, no network**. Pure CPU math over the pre-built index.

> Anything imported by `rank.py` (i.e. everything under `src/` except `parsers/jd.py` and `embedder.py`'s download path) must obey the rank-time constraints. The LLM (`src/parsers/jd.py`) is **pre-compute only** — never call it from the rank path. `src/config.py` documents this on the relevant constants.

## Commands

```bash
# Setup
pip install -r requirements.txt
cp .env.example .env            # add LLM key (pre-compute only)

# Step 1 — pre-compute (once, no limits). --resume continues an interrupted run.
python precompute.py --candidates data/candidates.jsonl --jd data/job_description.txt
python precompute.py ... --chunk-size 200 --resume   # lower RAM / resume

# Step 2 — rank (< 5 min, CPU only)
python rank.py --candidates data/candidates.jsonl --jd data/job_description.txt --out submission.csv

# Step 3 — validate before upload
python validate_submission.py --submission submission.csv --candidates data/candidates.jsonl

# Lint
ruff check .
```

There are **no tests yet** (`tests/` is empty). `rank.py` self-validates output via `_sanity_check`; `validate_submission.py` checks every spec rule. Treat those as the success criteria.

## Pipeline flow

```
precompute:  candidates.jsonl ─┬─ parse_redrob_candidate ─► parsed_candidates.jsonl (cache)
                               └─ Embedder(bge-base) ─► FAISS IndexFlatIP ─► candidates.faiss + candidate_ids.json
             job_description.txt ─ parse_jd_with_llm ─► parsed_jd.json

rank:        load index → embed JD → ANN top-500 → load those 500 from cache
             → RankingEngine.rank(): per-candidate weighted fusion + honeypot zero-out
             → sort → top-100 → template reasoning → CSV
```

## Architecture map

- `src/config.py` — **single source of truth** for all weights, thresholds, paths, and keyword sets. Tune scoring here, not in scorers. `WEIGHTS` must sum to 1.0.
- `src/ranker.py` — `RankingEngine`, the only orchestrator at rank time. Weighted fusion of 5 signals, honeypot zero-out, enforces non-increasing scores, writes CSV.
- `src/index.py` — FAISS build/save/load/query. `IndexFlatIP` = cosine on L2-normalized vectors.
- `src/embedder.py` — `SentenceTransformer` singleton (`BAAI/bge-base-en-v1.5`, 768-dim). `get_embedder()` is the only entry point.
- `src/honeypot.py` — flags impossible profiles; **2+ signals → honeypot → composite score forced to 0** (keeps top-100 under the 10% honeypot disqualification threshold).
- `src/reasoning.py` — template-based, no LLM. Must be specific and non-hallucinated.
- `src/parsers/candidate.py` — raw redrob schema → internal flat dict. **stdlib + dateutil only** (rank-time safe). All fields have defaults; scorers must never crash on missing data.
- `src/parsers/jd.py` — LLM JD extraction (Anthropic or Gemini), keyword fallback if no key. **Pre-compute only.** Treats JD contents as untrusted (prompt-injection guard).
- `src/scorers/` — `role_fit` (20%), `skill` (15%), `behavioral` (15%), `career` (10%); semantic (40%) comes straight from the FAISS score. Each scorer returns a float in [0,1].

## Conventions

- Python 3.11, `from __future__ import annotations`, type hints throughout.
- Scorers are stateless classes with a `score()` method returning `[0,1]`; pull constants from `config.py`.
- `logging` (not `print`) — module-level `logger = logging.getLogger(__name__)`.
- Section dividers use `# ── label ──...` comment banners — match this style.
- Defensive parsing: `(x.get(...) or default)` everywhere; never assume a field exists.
- Output CSV is exactly `candidate_id,rank,score,reasoning`, 100 rows, ranks 1–100, non-increasing scores, no empty reasoning.

## Gotchas

- The role-fit scorer applies **hard multiplicative penalties** (disqualifying title ×0.10, consulting-only career ×0.25) — this is intentional anti-keyword-stuffing logic, not a bug.
- `data/index/`, `.env`, `data/candidates.jsonl*`, and `submission.csv` are gitignored. The embedding model caches to `~/.cache/huggingface/` (~430MB first run).
- When changing scoring, verify `WEIGHTS` still sums to 1.0 and re-run `validate_submission.py`.
