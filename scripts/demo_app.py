"""Streamlit sandbox demo — accepts ≤100 candidates, runs full ranking pipeline.

Results are rendered by ``redrob_results_view.render_results_view`` (a self-
contained client-side component). The ranking pipeline below is unchanged.
"""

import html
import json
import sys
from pathlib import Path

import streamlit as st

# Make the repo root importable so the results view (repo-root module) resolves
# whether the app is launched from the repo root or from scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from redrob_results_view import render_results_view


def _load_candidates(raw: str) -> list[dict]:
    """Parse an uploaded candidate file as either a JSON array or JSONL.

    Uploads arrive in either shape: a pretty/compact JSON array
    (``[ { ... }, { ... } ]``) or newline-delimited JSON (one object per line).
    A leading BOM is stripped upstream by decoding as ``utf-8-sig``.
    """
    stripped = raw.strip()
    if not stripped:
        return []
    if stripped[0] == "[":  # JSON array — a single loads() covers pretty & compact
        try:
            data = json.loads(stripped)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass  # not a valid array as a whole — fall through to line-by-line JSONL
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


# ── DEMO-ONLY ranking polish (NOT in the production ranker) ────────────────────
# The production `rank.py` is the judged artifact (NDCG over 100K candidates with
# the LLM JD parse, done in PRE-COMPUTE). This sandbox demo parses the JD with the
# JD-aware KEYWORD extractor (no LLM, no network) so it adapts to any pasted JD
# while staying provably within the "no LLM during ranking" constraint. Some
# JD-excluded profiles — consulting-only careers, non-engineering titles, abroad
# candidates — can still leak into the visible top-20, so we demote them below.
#
# To make the live demo convincing WITHOUT touching `src/` (production), we
# re-score those excluded candidates with a stronger multiplier AFTER
# engine.rank() and re-sort. Scores stay non-increasing (the CSV still passes the
# validator); only the in-list ordering changes. Nothing here runs in rank.py.
_DEMO_EXTRA_DISQUALIFYING = frozenset({
    "business analyst", "project manager", "operations manager",
    "customer support",   # demo-only addition: not ML-adjacent, keeps the lead clean
})
_DEMO_EXCLUDE_MULTIPLIER = 0.40   # demoted below every clean engineer (0.45–0.65)


def _demo_is_excluded(sc, disqualifying: frozenset[str]) -> bool:
    """Demo-only JD-exclusion predicate. True → score is demoted below the fits."""
    title = (sc.current_title or "").lower()
    if any(dt in title for dt in disqualifying):
        return True
    if getattr(sc, "all_consulting", False):            # entire career at a services firm
        return True
    country = (sc.country or "").lower()
    if country and country not in ("india", "in"):       # abroad — India-hybrid role
        return True
    return False


def _demo_demote_excluded(ranked: list) -> list:
    """Re-score JD-excluded candidates and re-sort the list in place.

    Keeps the row count and reasoning unchanged; only `score`/`rank` move. After
    demotion, clean ML engineers lead and the excluded profiles sink to the tail.
    """
    if not ranked:
        return ranked
    # Production disqualifying titles + the demo-only extras (BA / PM / Ops).
    from src.config import DISQUALIFYING_TITLES
    disqualifying = DISQUALIFYING_TITLES | _DEMO_EXTRA_DISQUALIFYING
    for sc in ranked:
        if _demo_is_excluded(sc, disqualifying):
            sc.score = round(sc.score * _DEMO_EXCLUDE_MULTIPLIER, 6)
    ranked.sort(key=lambda x: (-x.score, x.candidate_id))
    # Re-enforce non-increasing scores and assign fresh ranks (mirrors ranker.py).
    prev = ranked[0].score
    for i, sc in enumerate(ranked):
        if sc.score > prev:
            sc.score = prev
        prev = sc.score
        sc.rank = i + 1
    return ranked


# Skill chips. Streamlit 1.40 has no markdown badge syntax, so we render real HTML
# spans via unsafe_allow_html. Colors live in the CSS (.rr-chip--*) injected by
# _inject_css(); here we only pick the class so markup stays clean.
_CHIP_CLASS = {
    "blue": "rr-chip--blue", "green": "rr-chip--green",
    "red": "rr-chip--red", "gold": "rr-chip--gold",
}


def _badges(items: list[str], color: str) -> str:
    """Render a skill list as styled HTML chips, or '—' if empty."""
    if not items:
        return "—"
    cls = _CHIP_CLASS[color]
    return "".join(
        f'<span class="rr-chip {cls}">{html.escape(str(s))}</span>' for s in items
    )


# ── Honeypot reason humaniser (used when building the results payload) ──────────
# The detector returns machine-readable reason strings (e.g. "expert_no_time:LoRA");
# we humanize them and re-derive them demo-side via HoneypotDetector().detect()
# (read-only, rank-time-safe — no src/ change, no effect on the ranking).

def _humanize_honeypot_reason(reason: str) -> str:
    """Turn a HoneypotDetector reason code into a readable phrase."""
    kind, _, payload = reason.partition(":")
    if kind == "expert_no_time":
        return f"Claims expert “{payload}” with ~0 months of hands-on time"
    if kind == "yoe_career_mismatch":
        claimed, _, rest = payload.partition("_claimed_vs_")          # 14.1yr_claimed_vs_4.7yr_career
        career = rest.replace("_career", "").replace("yr", " yrs")
        claimed = claimed.replace("yr", " yrs")
        return f"Claims {claimed} of experience but career history totals only {career}"
    if kind == "too_many_experts":
        count, _, rest = payload.partition("_in_")                    # 9_in_56mo_career
        months = rest.replace("_career", "").replace("mo", "")
        return f"{count} expert-level skills in just a {months}-month career"
    if kind == "title_skill_mismatch":
        title, _, rest = payload.partition("_with_")                  # <title>_with_<n>_advanced_ai_skills
        n = rest.split("_", 1)[0] if rest else "several"
        return f"Non-technical title ({title}) paired with {n} advanced AI skills"
    return reason.replace("_", " ")


# All visual styling for the hero + input card. Injected once after
# set_page_config. The Inter @import is browser-side (this is a public web page);
# the system-ui fallback keeps it readable if the font can't load.
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stApp { font-family: 'Instrument Sans', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif; }
.stApp { background: #f6f7f9; }

/* tidy chrome + a centered, comfortable column (matches the results component) */
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stHeader"] { visibility: hidden; height: 0; }
.block-container { max-width: 100%; padding-top: 2.4rem; padding-bottom: 4rem; padding-left: 3rem; padding-right: 3rem; }

/* top bar — mirrors the results component header */
.rrx-top { display: flex; align-items: center; gap: 13px; padding: 0 4px 16px; }
.rrx-logo { width: 15px; height: 15px; border-radius: 4px; background: #4f46e5; transform: rotate(45deg); display: inline-block; }
.rrx-word { font-weight: 700; font-size: 21px; letter-spacing: -0.015em; color: #0f172a; }
.rrx-tagpill { font-size: 12px; font-weight: 600; color: #6b7280; background: #f1f2f6; padding: 4px 10px; border-radius: 7px; letter-spacing: 0.02em; }
.rrx-spacer { flex: 1; }
.rrx-mono { font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #94a3b8; }

/* hero — white card + hairline, same surface language as the ledger */
.rr-hero { background: #fff; border: 1px solid #e7e9ef; border-radius: 16px; padding: 26px 30px; margin-bottom: 16px; box-shadow: 0 1px 2px rgba(15,23,42,0.04); position: relative; overflow: hidden; }
.rr-hero:before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px; background: linear-gradient(#6366f1, #8b5cf6); }
.rr-hero__badge { display: inline-block; font-family: 'JetBrains Mono', monospace; background: #f7f8fa; border: 1px solid #e7e9ef; color: #4f46e5; font-weight: 600; font-size: 0.7rem; letter-spacing: 0.02em; padding: 5px 11px; border-radius: 7px; margin-bottom: 14px; }
.rr-hero__title { font-size: 2.0rem; font-weight: 700; color: #0f172a; line-height: 1.12; margin: 0; letter-spacing: -0.025em; }
.rr-hero__tag { font-size: 0.95rem; color: #64748b; margin-top: 9px; max-width: 64ch; }
.rr-pills { margin-top: 15px; display: flex; flex-wrap: wrap; gap: 7px; }
.rr-pill { display: inline-block; background: #f1f2f6; border: 1px solid #e7e9ef; color: #475569; font-weight: 600; font-size: 0.76rem; padding: 5px 11px; border-radius: 999px; }
.rr-pill--accent { color: #4f46e5; border-color: #dcd9fb; background: #eef0fe; }

/* section labels in the input card */
.rr-section { font-size: 0.95rem; font-weight: 700; color: #0f172a; margin: 4px 0 6px; letter-spacing: -0.01em; }
.rr-section .rr-num { font-family: 'JetBrains Mono', monospace; color: #94a3b8; font-weight: 700; margin-right: 5px; }
.rr-subtle { color: #94a3b8; font-weight: 500; }
.rr-hint { color: #94a3b8; font-size: 0.9rem; text-align: center; margin: 22px 0; }

/* chips (still used elsewhere) */
.rr-chip { display: inline-block; padding: 3px 11px; border-radius: 999px; margin: 3px 5px 3px 0; font-size: 0.82rem; font-weight: 500; white-space: nowrap; border: 1px solid transparent; }
.rr-chip--blue  { background: #eef2ff; color: #3730a3; border-color: #dfe3fb; }
.rr-chip--green { background: #ecfdf3; color: #1b7a32; border-color: #d6f5df; }
.rr-chip--red   { background: #fef2f2; color: #b42318; border-color: #fbdcdc; }
.rr-chip--gold  { background: #fef6e0; color: #92600a; border-color: #f3dc97; }

/* Streamlit widgets → component look */
[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 16px; border-color: #e7e9ef; background: #fff; }
.stTextArea textarea { font-family: 'Instrument Sans', sans-serif; font-size: 14px; background: #f7f8fa; border: 1px solid #e7e9ef; border-radius: 10px; color: #0f172a; }
.stTextArea textarea:focus { border-color: #c7c3f3; box-shadow: 0 0 0 3px rgba(79,70,229,0.12); }
[data-testid="stFileUploaderDropzone"] { background: #f7f8fa; border: 1px dashed #d6dae2; border-radius: 12px; }
.stButton > button, .stDownloadButton > button { border-radius: 10px; font-weight: 600; font-family: 'Instrument Sans', sans-serif; }
.stButton > button[kind="secondary"] { background: #fff; border: 1px solid #e2e4ec; color: #475569; }
.stButton > button[kind="secondary"]:hover { border-color: #cfd3df; color: #0f172a; }
.stButton > button[kind="primary"] { background: #4f46e5; border-color: #4f46e5; }
.stButton > button[kind="primary"]:hover { background: #4338ca; border-color: #4338ca; }
[data-testid="stAlert"], [data-testid="stNotification"] { border-radius: 12px; font-family: 'Instrument Sans', sans-serif; }
/* input expander styled like the component's cards */
[data-testid="stExpander"] { border: 1px solid #e7e9ef; border-radius: 16px; background: #fff; box-shadow: 0 1px 2px rgba(15,23,42,0.04); }
[data-testid="stExpander"] details > summary { padding: 14px 20px; font-size: 0.95rem; font-weight: 700; color: #0f172a; font-family: 'Instrument Sans', sans-serif; }
[data-testid="stExpander"] details > summary:hover { color: #4f46e5; }
[data-testid="stExpander"] details[open] > summary { border-bottom: 1px solid #f1f2f6; }
</style>
"""


def _inject_css() -> None:
    """Inject the custom stylesheet once (call right after set_page_config)."""
    st.markdown(_CSS, unsafe_allow_html=True)


# ── Sample JD (powers the "Load sample JD" button) ─────────────────────────────
# Loads the real data/job_description.txt when present (it ships in the image);
# otherwise falls back to a condensed but faithful copy so the button always works.
_SAMPLE_JD_PATH = Path(__file__).resolve().parent.parent / "data" / "job_description.txt"

_SAMPLE_JD_FALLBACK = """\
Job Description: Senior AI Engineer — Founding Team
Company: Redrob AI (Series A AI-native talent intelligence platform)
Location: Pune/Noida, India (Hybrid) | Open to relocation from Tier-1 Indian cities
Experience Required: 5–9 years

We're building a new AI Engineering org from scratch. We need someone with deep
technical depth in modern ML systems — embeddings, retrieval, ranking, LLMs,
fine-tuning — AND a scrappy product-engineering attitude who will ship a working
ranker in a week and learn from real users.

You'd own the intelligence layer: the ranking, retrieval, and matching systems
behind candidate and role search. First 90 days: audit the current BM25 + rule
stack, ship a v2 ranking system (embeddings, hybrid retrieval, LLM re-ranking),
and stand up evaluation infrastructure (offline NDCG/MAP benchmarks, online A/B
testing, recruiter-feedback loops).

Things you absolutely need:
- Production embeddings-based retrieval (sentence-transformers, BGE, E5, OpenAI) at real-user scale
- Vector DB / hybrid search infra (Pinecone, Weaviate, Qdrant, Milvus, FAISS, Elasticsearch)
- Strong Python and code quality
- Designing evaluation frameworks for ranking (NDCG, MRR, MAP, offline-to-online correlation, A/B tests)

Nice to have: LoRA/QLoRA/PEFT fine-tuning, learning-to-rank, HR-tech exposure,
distributed systems, open-source AI/ML contributions.

Explicit disqualifiers:
- Pure-research-only careers with no production deployment
- "AI experience" that is only recent (<12 months) LangChain-calls-OpenAI work
- Senior engineers who haven't written production code in 18+ months
- Title-chasers switching companies every ~1.5 years
- Entirely consulting-firm careers (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini)
- Primary expertise in computer vision, speech, or robotics without NLP/IR

The trap: do NOT just match the most AI keywords. A "Marketing Manager" with a
perfect AI skill list is not a fit; a Tier-5 candidate who built a recommendation
system at a product company is. Weigh behavioral signals — an inactive, low-
response candidate is, for hiring purposes, not available. Down-weight them.
"""


def _sample_jd_text() -> str:
    """Return the real JD if it ships in the image, else the condensed fallback."""
    try:
        text = _SAMPLE_JD_PATH.read_text(encoding="utf-8").strip()
        return text or _SAMPLE_JD_FALLBACK
    except Exception:
        return _SAMPLE_JD_FALLBACK


def _load_sample_jd() -> None:
    """on_click callback — pre-fills the JD textarea before widgets instantiate."""
    st.session_state["jd_text"] = _sample_jd_text()


st.set_page_config(page_title="Redrob Ranker — Velocity Labs", page_icon="🎯", layout="wide")
_inject_css()
st.session_state.setdefault("jd_text", "")

# ── Hero ────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="rrx-top">
      <span class="rrx-logo"></span>
      <span class="rrx-word">Redrob Ranker</span>
      <span class="rrx-tagpill">EVIDENCE LEDGER</span>
      <span class="rrx-spacer"></span>
      <span class="rrx-mono">CPU · No LLM at rank time · &lt;5 min</span>
    </div>
    <div class="rr-hero">
      <div class="rr-hero__badge">VELOCITY LABS · INDIA.RUNS HACKATHON — TRACK 01</div>
      <div class="rr-hero__title">Redrob Intelligent Candidate Ranker</div>
      <div class="rr-hero__tag">An evidence ledger for hiring — every rank shows the five signals that earned it. 100K candidates → top 100, CPU-only with no LLM at rank time.</div>
      <div class="rr-pills">
        <span class="rr-pill rr-pill--accent">Explainable scoring</span>
        <span class="rr-pill">CPU · No network · &lt;5 min</span>
        <span class="rr-pill">Honeypot filtering</span>
        <span class="rr-pill">Hidden gems</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Input card (collapses once results exist, so the ledger surfaces) ──────────
_has_results = st.session_state.get("results") is not None
with st.expander("Job description & candidates  ·  edit & run", expanded=not _has_results):
    hcol, bcol = st.columns([3, 1])
    with hcol:
        st.markdown('<div class="rr-section"><span class="rr-num">1</span> Job description</div>', unsafe_allow_html=True)
    with bcol:
        st.button(
            "Load sample JD",
            on_click=_load_sample_jd,
            use_container_width=True,
            help="Fill the box with the Redrob Senior AI Engineer JD so you can run the ranker in one click.",
        )
    jd_text = st.text_area(
        "Job description",
        key="jd_text",
        height=180,
        placeholder="Paste the full job description here…",
        label_visibility="collapsed",
    )
    st.markdown(
        '<div class="rr-section"><span class="rr-num">2</span> Candidates <span class="rr-subtle">· optional</span></div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Upload ≤100 candidates (JSONL / JSON array, redrob schema), or leave empty "
        "for the built-in demo sample."
    )
    uploaded = st.file_uploader(
        "Candidate file",
        type=["jsonl", "json"],
        label_visibility="collapsed",
    )
    if uploaded is None:
        st.info(
            "⚡ **Ranking the built-in 100-candidate demo sample** — pre-indexed at "
            "build time, so only the job description is embedded now (fast)."
        )
    else:
        st.success(
            "📤 **Ranking your uploaded candidates** - parsed & embedded live on CPU "
            "(adds a few seconds)."
        )
    run = st.button(
        "Run ranking", type="primary", disabled=not jd_text, use_container_width=True
    )

# ── Pipeline (runs on click; results persisted to session_state) ───────────────
if run:
    sys.path.insert(0, str(Path(__file__).parent.parent))

    # With no upload we load a pre-built sample index (built at Docker image
    # time), so the demo only embeds the JD — seconds, not the ~1 min that
    # embedding 100 candidates costs on the HF free-tier CPU. Ranking logic is
    # unchanged either way.
    with st.status("Running ranking pipeline...", expanded=True) as status:
        try:
            from src.parsers.candidate import parse_redrob_candidate
            from src.parsers.jd import _keyword_fallback
            from src.embedder import get_embedder
            from src.index import build_index, query_index
            from src.ranker import RankingEngine
            from src.honeypot import HoneypotDetector
            from src.config import EMBED_BATCH_SIZE, TOP_K_RETRIEVE
            import faiss

            SAMPLE_INDEX_DIR = Path(__file__).resolve().parent.parent / "sample_index"

            st.write("Parsing JD (keyword extraction — no LLM, no network)...")
            parsed_jd = _keyword_fallback(jd_text)  # JD-aware, no LLM (rank-time safe)
            embedder = get_embedder()

            if uploaded is not None:
                # ── Upload path: parse + embed the provided candidates at runtime ──
                st.write("Parsing upload...")
                raw = uploaded.read().decode("utf-8-sig")  # strips a leading BOM
                candidates_raw = _load_candidates(raw)
                if not candidates_raw:
                    raise ValueError(
                        "No candidates found in the upload. Expected JSONL (one JSON "
                        "object per line) or a JSON array of candidate objects."
                    )
                candidates = [parse_redrob_candidate(r) for r in candidates_raw]
                texts = [c["embedding_text"] for c in candidates]
                n_batches = max(1, (len(texts) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE)
                st.write(
                    f"Embedding {len(texts)} candidates on CPU "
                    f"(~{n_batches} batch{'es' if n_batches != 1 else ''}; the slow step, "
                    f"please wait)..."
                )
                embeddings = embedder.embed_batch(texts)
                candidate_ids = [c["candidate_id"] for c in candidates]
                index = build_index(embeddings, candidate_ids)
                cands_by_id = {c["candidate_id"]: c for c in candidates}
            else:
                # ── Fast path: load the pre-built sample index (no embedding) ──
                idx_file = SAMPLE_INDEX_DIR / "candidates.faiss"
                if not idx_file.exists():
                    raise FileNotFoundError(
                        "No candidate upload and no built-in sample index found. "
                        "Either upload a candidate file, or build the sample index "
                        "first: python scripts/build_sample_index.py"
                    )
                st.write("Loading pre-built sample index (100 candidates, no embedding)...")
                index = faiss.read_index(str(idx_file))
                candidate_ids = json.loads(
                    (SAMPLE_INDEX_DIR / "candidate_ids.json").read_text()
                )
                cache = []
                with open(SAMPLE_INDEX_DIR / "parsed_candidates.jsonl", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            cache.append(json.loads(line))
                candidates = cache
                cands_by_id = {c["candidate_id"]: c for c in candidates}

            st.write("Embedding job description (cached on repeat runs)...")
            jd_vec = embedder.embed_text(parsed_jd.to_embedding_text())

            st.write("Ranking...")
            k = min(len(candidates), TOP_K_RETRIEVE)
            ann_results = query_index(index, candidate_ids, jd_vec, k)
            engine = RankingEngine()
            ranked = engine.rank(cands_by_id, ann_results, parsed_jd)
            ranked = _demo_demote_excluded(ranked)   # demo-only polish, NOT in production

            status.update(label="✅ Ranking complete", state="complete", expanded=False)
        except Exception as exc:
            status.update(label="Ranking failed", state="error", expanded=True)
            st.error(f"Error: {exc}")
            raise

    # Build the CSV (exact spec format) and a candidate_id → display-name map, then
    # persist everything so results survive download/widget reruns (st.button is
    # only True on the click run; st.download_button triggers a rerun).
    import io
    import csv
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["candidate_id", "rank", "score", "reasoning"])
    writer.writeheader()
    for sc in ranked:
        writer.writerow({
            "candidate_id": sc.candidate_id,
            "rank": sc.rank,
            "score": sc.score,
            "reasoning": sc.reasoning,
        })
    names = {
        sc.candidate_id: (cands_by_id.get(sc.candidate_id, {}).get("name") or sc.candidate_id)
        for sc in ranked
    }
    # Re-derive honeypot reasons for the flagged candidates (the engine keeps only
    # the boolean). Read-only — does not touch the scores already computed.
    _detector = HoneypotDetector()
    honeypots = []
    for sc in ranked:
        if not getattr(sc, "is_honeypot", False):
            continue
        _, reasons = _detector.detect(cands_by_id.get(sc.candidate_id, {}))
        honeypots.append({
            "id": sc.candidate_id,
            "name": names.get(sc.candidate_id, sc.candidate_id),
            "title": sc.current_title,
            "company": sc.current_company,
            "reasons": [_humanize_honeypot_reason(r) for r in reasons],
        })
    st.session_state["results"] = {
        "ranked": ranked,
        "parsed_jd": parsed_jd,
        "csv": buf.getvalue(),
        "names": names,
        "honeypots": honeypots,
    }
    # Rerun immediately so the page re-renders with results present: the input
    # expander collapses (it reads session_state["results"]) and the ledger
    # surfaces right under the hero instead of below the open input card.
    st.rerun()

# ── Results: redesigned evidence ledger (rendered from session_state) ──────────
# All presentation lives in redrob_results_view.render_results_view — a self-
# contained client-side component. It reads only fields already on the
# ScoredCandidate objects; no src/ import, no scoring, no LLM, no network.
results = st.session_state.get("results")
if not results:
    st.markdown(
        '<div class="rr-hint">Paste a job description (or hit <b>Load sample JD</b>) and run '
        "the ranker to see the evidence ledger — each candidate's score, broken down by signal.</div>",
        unsafe_allow_html=True,
    )
else:
    render_results_view(
        ranked=results["ranked"],
        parsed_jd=results["parsed_jd"],
        names=results["names"],
        honeypots=results["honeypots"],
        csv_data=results["csv"],
    )
