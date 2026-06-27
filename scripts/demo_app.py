"""Streamlit sandbox demo — accepts ≤100 candidates, runs full ranking pipeline."""

import html
import json
from pathlib import Path

import streamlit as st


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
_CHIP_CLASS = {"blue": "rr-chip--blue", "green": "rr-chip--green", "red": "rr-chip--red"}


def _badges(items: list[str], color: str) -> str:
    """Render a skill list as styled HTML chips, or '—' if empty."""
    if not items:
        return "—"
    cls = _CHIP_CLASS[color]
    return "".join(
        f'<span class="rr-chip {cls}">{html.escape(str(s))}</span>' for s in items
    )


def _render_jd_keywords(parsed_jd) -> None:
    """Show the keyword extraction (no LLM) on screen, grouped by category.

    Surfaces what the engine pulled from the pasted JD, so it's clear the role was
    understood. Pure display of the existing `ParsedJD` fields — no scoring side
    effects.
    """
    st.caption("Keyword extraction — no LLM, no network (rank-time safe)")

    col1, col2, col3 = st.columns(3)
    col1.metric("Role", parsed_jd.title)
    col2.metric("Seniority", parsed_jd.seniority_level.title())
    col3.metric(
        "Experience",
        f"{parsed_jd.min_experience_years}–{parsed_jd.max_experience_years} yrs",
    )
    st.markdown(f"**Domain:** {parsed_jd.domain}")
    st.markdown(
        f"**Required skills:** {_badges(parsed_jd.required_skills, 'blue')}",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"**Nice-to-have skills:** {_badges(parsed_jd.nice_to_have_skills, 'green')}",
        unsafe_allow_html=True,
    )
    if parsed_jd.disqualifiers:
        # "JD-specific flags" not "Disqualifiers": the detectors are intentionally
        # narrow/JD-tuned, so this label reads more honestly than a hard verdict.
        st.markdown(
            f"**JD-specific flags:** {_badges(parsed_jd.disqualifiers, 'red')}",
            unsafe_allow_html=True,
        )


# ── Evidence-ledger config ──────────────────────────────────────────────────────
# The five fusion signals, in weight order. Weights MIRROR src/config.WEIGHTS
# (kept as a literal so this render layer needs no rank-time import). Each ledger
# row draws a stacked bar whose segment widths = sub_score × weight, so the bar
# literally shows how the composite was built — our explainability angle.
_SIGNAL_META = [
    ("semantic_score",   "Semantic",   "#6366f1", 0.40),
    ("role_fit_score",   "Role-fit",   "#0ea5e9", 0.20),
    ("skill_score",      "Skill",      "#10b981", 0.15),
    ("behavioral_score", "Behavioral", "#f59e0b", 0.15),
    ("career_score",     "Career",     "#8b5cf6", 0.10),
]

_LEDGER_ROWS = 12    # rich ledger cards (full ranking lives in the table + CSV)


# All visual styling for the custom components lives here. Injected once after
# set_page_config. The Inter @import is browser-side (this is a public web page);
# the system-ui fallback keeps it readable if the font can't load.
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp { font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif; }

/* tidy chrome + a centered, comfortable column */
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; height: 0; }
.block-container { max-width: 960px; padding-top: 2.0rem; padding-bottom: 4rem; }

/* hero */
.rr-hero {
  background: linear-gradient(135deg, #eef2ff 0%, #f5f3ff 50%, #eff6ff 100%);
  border: 1px solid #e5e7f5; border-radius: 18px; padding: 24px 30px; margin-bottom: 18px;
  box-shadow: 0 1px 3px rgba(15,23,42,0.04);
}
.rr-hero__badge {
  display: inline-block; background: #fff; border: 1px solid #e2e4f0; color: #4f46e5;
  font-weight: 600; font-size: 0.76rem; letter-spacing: .03em;
  padding: 5px 12px; border-radius: 999px; margin-bottom: 12px;
}
.rr-hero__title { font-size: 2.0rem; font-weight: 800; color: #0f172a; line-height: 1.15; margin: 0; letter-spacing: -0.02em; }
.rr-hero__tag { font-size: 0.94rem; color: #64748b; margin-top: 8px; }
.rr-pills { margin-top: 14px; display: flex; flex-wrap: wrap; gap: 8px; }
.rr-pill {
  display: inline-block; background: rgba(255,255,255,0.75); border: 1px solid #e2e4f0; color: #475569;
  font-weight: 600; font-size: 0.78rem; padding: 5px 12px; border-radius: 999px;
}
.rr-pill--accent { color: #4f46e5; border-color: #d7dbfb; background: #eef2ff; }

/* section + helpers */
.rr-section { font-size: 1.0rem; font-weight: 700; color: #0f172a; margin: 6px 0 6px; letter-spacing: -0.01em; }
.rr-subtle { color: #94a3b8; font-weight: 500; }
.rr-hint { color: #94a3b8; font-size: 0.9rem; text-align: center; margin: 22px 0; }

/* chips */
.rr-chip {
  display: inline-block; padding: 3px 11px; border-radius: 999px; margin: 3px 5px 3px 0;
  font-size: 0.82rem; font-weight: 500; white-space: nowrap; border: 1px solid transparent;
}
.rr-chip--blue  { background: #eef2ff; color: #3730a3; border-color: #dfe3fb; }
.rr-chip--green { background: #ecfdf3; color: #1b7a32; border-color: #d6f5df; }
.rr-chip--red   { background: #fef2f2; color: #b42318; border-color: #fbdcdc; }

/* scan recap */
.rr-recap { font-size: 0.95rem; color: #334155; }
.rr-recap b { color: #0f172a; }

/* signal legend */
.rr-legend { display: flex; flex-wrap: wrap; gap: 16px; margin: 4px 0 14px; font-size: 0.78rem; color: #64748b; }
.rr-legend__item { white-space: nowrap; }
.rr-legend__dot { display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 6px; vertical-align: middle; }

/* ledger card head/body (the card border comes from st.container(border=True)) */
.rr-led__head { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; }
.rr-led__id { min-width: 0; }
.rr-led__rank { font-weight: 800; color: #cbd5e1; font-size: 0.92rem; margin-right: 8px; }
.rr-led__rank--top { color: #4f46e5; }
.rr-led__name { font-weight: 700; color: #0f172a; font-size: 1.0rem; }
.rr-led__score { font-weight: 800; color: #0f172a; font-size: 1.18rem; white-space: nowrap; letter-spacing: -0.01em; }
.rr-led__score small { font-size: 0.6rem; color: #94a3b8; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; margin-left: 4px; }
.rr-led__meta { font-size: 0.8rem; color: #94a3b8; margin: 3px 0 0; }
.rr-led__reason { font-size: 0.86rem; color: #475569; line-height: 1.5; margin-top: 8px; }

/* the contribution bar — the signature visual */
.rr-bar { display: flex; height: 12px; background: #eef0f6; border-radius: 999px; overflow: hidden; margin: 11px 0 2px; }
.rr-seg { height: 100%; }
.rr-seg:hover { filter: brightness(1.08); }

/* honeypot card (inside the "Honeypots filtered" panel) */
.rr-hp__head { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; }
.rr-hp__name { font-weight: 700; color: #0f172a; font-size: 0.96rem; }
.rr-hp__zero { font-size: 0.7rem; font-weight: 700; letter-spacing: .04em; text-transform: uppercase; color: #b42318; background: #fef2f2; border: 1px solid #fbdcdc; padding: 2px 9px; border-radius: 999px; white-space: nowrap; }
.rr-hp__chips { margin-top: 8px; }

/* button polish */
.stButton > button, .stDownloadButton > button { border-radius: 10px; font-weight: 600; }
</style>
"""


def _inject_css() -> None:
    """Inject the custom stylesheet once (call right after set_page_config)."""
    st.markdown(_CSS, unsafe_allow_html=True)


# ── Ledger renderers (pure presentation of existing ScoredCandidate fields) ─────

def _meta_line(sc) -> str:
    """`title @ company · Xyr · location · Nd notice` — HTML-escaped."""
    parts: list[str] = []
    title = sc.current_title or "—"
    company = sc.current_company or ""
    parts.append(f"{title} @ {company}" if company else title)
    try:
        parts.append(f"{float(sc.years_of_experience):g}yr")
    except (TypeError, ValueError):
        pass
    if sc.location:
        parts.append(sc.location)
    notice = getattr(sc, "notice_period_days", None)
    if notice is not None:
        parts.append(f"{int(notice)}d notice")
    return html.escape(" · ".join(parts))


def _render_legend() -> str:
    """Color key for the contribution bar (signal → weight)."""
    items = "".join(
        f'<span class="rr-legend__item">'
        f'<span class="rr-legend__dot" style="background:{color}"></span>'
        f"{label} {int(weight * 100)}%</span>"
        for _key, label, color, weight in _SIGNAL_META
    )
    return f'<div class="rr-legend">{items}</div>'


def _contribution_bar(sc) -> str:
    """Stacked bar: each segment width = sub_score × weight (scale 0–1 = full bar)."""
    segs = []
    for key, label, color, weight in _SIGNAL_META:
        val = max(0.0, float(getattr(sc, key, 0.0) or 0.0))
        contrib = val * weight
        if contrib <= 0:
            continue
        segs.append(
            f'<div class="rr-seg" style="width:{contrib * 100:.2f}%;background:{color}" '
            f'title="{label}: {contrib:.3f}  ({val:.2f} × {weight:.2f})"></div>'
        )
    return f'<div class="rr-bar">{"".join(segs)}</div>'


def _ledger_head_html(sc, name: str) -> str:
    """Headline + meta + contribution bar + reasoning for one ledger card."""
    rank_cls = " rr-led__rank--top" if sc.rank <= 3 else ""
    return (
        '<div class="rr-led__head">'
        f'<div class="rr-led__id"><span class="rr-led__rank{rank_cls}">#{sc.rank}</span>'
        f'<span class="rr-led__name">{html.escape(str(name))}</span></div>'
        f'<div class="rr-led__score">{float(sc.score or 0.0):.4f}<small>score</small></div>'
        "</div>"
        f'<div class="rr-led__meta">{_meta_line(sc)}</div>'
        f"{_contribution_bar(sc)}"
        f'<div class="rr-led__reason">{html.escape(sc.reasoning or "")}</div>'
    )


def _render_evidence(sc) -> None:
    """Inside-the-expander drill-down: matched skills, hidden-gem signals, sub-scores."""
    st.markdown(
        f"**Matched skills:** {_badges(sc.matched_skills, 'green')}",
        unsafe_allow_html=True,
    )
    if sc.hidden_gem_reasons:
        gems = [r.replace("_", " ") for r in sc.hidden_gem_reasons]
        st.markdown(
            f"**Hidden-gem signals:** {_badges(gems, 'blue')}", unsafe_allow_html=True
        )
    if getattr(sc, "is_honeypot", False):
        st.markdown("**Flag:** ⚠️ honeypot — composite forced to 0")
    breakdown = " · ".join(
        f"{label} {float(getattr(sc, key, 0.0) or 0.0):.2f}"
        for key, label, _color, _w in _SIGNAL_META
    )
    st.markdown(f"**Signal scores:** {breakdown}")
    st.caption(f"Confidence: {getattr(sc, 'confidence', '—')}")


# ── Honeypot panel (anti-gaming evidence) ───────────────────────────────────────
# Honeypots are forced to score 0 by RankingEngine, so they sink to the tail and
# never reach the top-12 ledger. This panel surfaces WHY each was caught. The
# detector returns machine-readable reason strings (e.g. "expert_no_time:LoRA");
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


def _honeypot_card_html(hp: dict) -> str:
    """One honeypot row: name + 'score 0' badge, title@company·id meta, red reason chips."""
    title = hp.get("title") or "—"
    company = hp.get("company") or ""
    meta = f"{title} @ {company}" if company else title
    meta = f"{meta} · {hp.get('id', '')}"
    return (
        '<div class="rr-hp__head">'
        f'<span class="rr-hp__name">{html.escape(str(hp.get("name", "")))}</span>'
        '<span class="rr-hp__zero">score 0 · excluded</span>'
        "</div>"
        f'<div class="rr-led__meta">{html.escape(meta)}</div>'
        f'<div class="rr-hp__chips">{_badges(hp.get("reasons", []), "red")}</div>'
    )


st.set_page_config(page_title="Redrob Ranker — Velocity Labs", page_icon="🎯", layout="centered")
_inject_css()

# ── Hero ────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="rr-hero">
      <div class="rr-hero__badge">🎯 Velocity Labs · INDIA.RUNS Hackathon — Track 01</div>
      <div class="rr-hero__title">Redrob Intelligent Candidate Ranker</div>
      <div class="rr-hero__tag">An evidence ledger for hiring — every rank shows the five signals that earned it. 100K candidates → top 100, CPU-only with no LLM at rank time.</div>
      <div class="rr-pills">
        <span class="rr-pill rr-pill--accent">Explainable scoring</span>
        <span class="rr-pill">CPU · No network · &lt;5 min</span>
        <span class="rr-pill">Honeypot filtering</span>
        <span class="rr-pill">Dark-horse detection</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Input card ──────────────────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown('<div class="rr-section">1 · Job description</div>', unsafe_allow_html=True)
    jd_text = st.text_area(
        "Job description",
        height=180,
        placeholder="Paste the full job description here…",
        label_visibility="collapsed",
    )
    st.markdown(
        '<div class="rr-section">2 · Candidates <span class="rr-subtle">· optional</span></div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Upload ≤100 candidates (JSONL or JSON array, redrob schema), or leave empty to "
        "rank the built-in 100-candidate sample — no per-candidate embedding at runtime."
    )
    uploaded = st.file_uploader(
        "Candidate file",
        type=["jsonl", "json"],
        label_visibility="collapsed",
    )
    run = st.button(
        "🚀 Run Ranking", type="primary", disabled=not jd_text, use_container_width=True
    )

# ── Pipeline (runs on click; results persisted to session_state) ───────────────
if run:
    import sys
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

# ── Results: the evidence ledger (rendered from session_state) ─────────────────
results = st.session_state.get("results")
if not results:
    st.markdown(
        '<div class="rr-hint">Paste a job description and run the ranker to see the '
        "evidence ledger — each candidate's score, broken down by signal.</div>",
        unsafe_allow_html=True,
    )
else:
    ranked = results["ranked"]
    parsed_jd = results["parsed_jd"]
    csv_data = results["csv"]
    names = results["names"]

    honeypots = sum(1 for sc in ranked if getattr(sc, "is_honeypot", False))
    dark_horses = sum(1 for sc in ranked if getattr(sc, "hidden_gem_reasons", None))
    top = float(ranked[0].score) if ranked else 0.0

    rcol, dcol = st.columns([3, 1])
    with rcol:
        st.markdown(
            '<div class="rr-recap">Ranked <b>{n}</b> candidates · <b>{hp}</b> honeypots '
            "filtered · <b>{dh}</b> dark horses · top score <b>{top:.3f}</b></div>".format(
                n=len(ranked), hp=honeypots, dh=dark_horses, top=top
            ),
            unsafe_allow_html=True,
        )
    with dcol:
        st.download_button(
            "⬇️ Download CSV",
            data=csv_data,
            file_name="submission.csv",
            mime="text/csv",
            use_container_width=True,
        )

    honeypots = results.get("honeypots", [])
    if honeypots:
        with st.expander(
            f"🛡️ Honeypots filtered — {len(honeypots)} caught & zeroed", expanded=False
        ):
            st.caption(
                "Profiles tripping 2+ impossible-profile signals are forced to score 0 so "
                "they can't game the ranking. Over the full 100K pool they fall below the "
                "top-100 cutoff entirely; here they sit at the tail with score 0."
            )
            for hp in honeypots:
                with st.container(border=True):
                    st.markdown(_honeypot_card_html(hp), unsafe_allow_html=True)

    with st.expander("🔑 What the engine understood", expanded=False):
        _render_jd_keywords(parsed_jd)

    st.markdown(
        '<div class="rr-section">📒 Evidence ledger '
        f'<span class="rr-subtle">· top {min(_LEDGER_ROWS, len(ranked))}</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(_render_legend(), unsafe_allow_html=True)
    st.caption(
        "Each bar shows how the composite was built — segment width = signal score × its weight."
    )

    for sc in ranked[:_LEDGER_ROWS]:
        with st.container(border=True):
            st.markdown(
                _ledger_head_html(sc, names.get(sc.candidate_id, sc.candidate_id)),
                unsafe_allow_html=True,
            )
            with st.expander("Evidence & signal breakdown"):
                _render_evidence(sc)

    with st.expander(f"📄 Full ranking — all {len(ranked)} rows (table)"):
        import pandas as pd
        df = pd.DataFrame([
            {"rank": sc.rank, "candidate_id": sc.candidate_id, "score": sc.score,
             "title": sc.current_title, "yoe": sc.years_of_experience, "reasoning": sc.reasoning}
            for sc in ranked
        ])
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "rank": st.column_config.NumberColumn("Rank", format="%d", width="small"),
                "candidate_id": st.column_config.TextColumn("Candidate"),
                "score": st.column_config.ProgressColumn(
                    "Score", min_value=0.0, max_value=1.0, format="%.3f"
                ),
                "title": st.column_config.TextColumn("Title"),
                "yoe": st.column_config.NumberColumn("YoE", format="%d", width="small"),
                "reasoning": st.column_config.TextColumn("Reasoning", width="large"),
            },
        )
    st.caption(
        f"The ledger shows the top {min(_LEDGER_ROWS, len(ranked))}; the table and the "
        "downloaded CSV hold the full ranking."
    )
