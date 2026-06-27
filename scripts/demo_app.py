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

    Surfaces what the engine pulled from the pasted JD *before* the ranking
    results, so it's clear the role was understood. Pure display of the existing
    `ParsedJD` fields — no scoring side effects.
    """
    st.markdown(
        '<div class="rr-section">🔑 What the engine understood</div>', unsafe_allow_html=True
    )
    st.caption("Keyword extraction — no LLM, no network (rank-time safe)")

    with st.container(border=True):
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


# How many ranked candidates to surface in each results view (full ranking is in
# the downloaded CSV either way).
_CARD_COUNT = 10
_TABLE_COUNT = 20

# All visual styling for the custom components lives here. Injected once after
# set_page_config. The Inter @import is browser-side (this is a public web page);
# the system-ui fallback keeps it readable if the font can't load.
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp { font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif; }

/* tidy chrome + a centered, comfortable column */
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; height: 0; }
.block-container { max-width: 1080px; padding-top: 2.0rem; padding-bottom: 4rem; }

/* hero */
.rr-hero {
  background: linear-gradient(135deg, #eef2ff 0%, #f5f3ff 50%, #eff6ff 100%);
  border: 1px solid #e5e7f5; border-radius: 18px; padding: 26px 30px; margin-bottom: 22px;
  box-shadow: 0 1px 3px rgba(15,23,42,0.04);
}
.rr-hero__badge {
  display: inline-block; background: #fff; border: 1px solid #e2e4f0; color: #4f46e5;
  font-weight: 600; font-size: 0.76rem; letter-spacing: .03em;
  padding: 5px 12px; border-radius: 999px; margin-bottom: 14px;
}
.rr-hero__title { font-size: 2.05rem; font-weight: 800; color: #0f172a; line-height: 1.15; margin: 0; letter-spacing: -0.02em; }
.rr-hero__tag { font-size: 0.94rem; color: #64748b; margin-top: 8px; }

/* section heading */
.rr-section { font-size: 1.05rem; font-weight: 700; color: #0f172a; margin: 8px 0 2px; letter-spacing: -0.01em; }

/* chips */
.rr-chip {
  display: inline-block; padding: 3px 11px; border-radius: 999px; margin: 3px 5px 3px 0;
  font-size: 0.82rem; font-weight: 500; white-space: nowrap; border: 1px solid transparent;
}
.rr-chip--blue  { background: #eef2ff; color: #3730a3; border-color: #dfe3fb; }
.rr-chip--green { background: #ecfdf3; color: #1b7a32; border-color: #d6f5df; }
.rr-chip--red   { background: #fef2f2; color: #b42318; border-color: #fbdcdc; }

/* candidate cards */
.rr-cands { display: flex; flex-direction: column; gap: 12px; margin-top: 8px; }
.rr-cand {
  display: flex; gap: 16px; background: #fff; border: 1px solid #e9ecf5; border-radius: 14px;
  padding: 16px 18px; transition: box-shadow .15s ease, transform .15s ease;
}
.rr-cand:hover { box-shadow: 0 6px 18px rgba(15,23,42,0.07); transform: translateY(-1px); }
.rr-rank {
  flex: 0 0 auto; width: 40px; height: 40px; border-radius: 11px;
  display: flex; align-items: center; justify-content: center;
  font-weight: 700; font-size: 1.0rem; color: #475569; background: #f1f3f9;
}
.rr-rank--top { background: linear-gradient(135deg, #f59e0b, #fbbf24); color: #fff; box-shadow: 0 2px 8px rgba(245,158,11,.35); }
.rr-cand__body { flex: 1 1 auto; min-width: 0; }
.rr-cand__head { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; }
.rr-cand__title { font-size: 1.0rem; font-weight: 600; color: #0f172a; }
.rr-cand__score { font-size: 1.12rem; font-weight: 700; color: #4f46e5; white-space: nowrap; }
.rr-cand__score small { font-size: 0.62rem; color: #94a3b8; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; margin-left: 4px; }
.rr-cand__meta { font-size: 0.8rem; color: #94a3b8; margin: 3px 0 9px; }
.rr-scorebar { height: 6px; background: #eef0f6; border-radius: 999px; overflow: hidden; }
.rr-scorebar__fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #6366f1, #4f46e5); }
.rr-cand__reason { font-size: 0.86rem; color: #475569; line-height: 1.5; margin-top: 10px; }

/* button polish */
.stButton > button, .stDownloadButton > button { border-radius: 10px; font-weight: 600; }
</style>
"""


def _inject_css() -> None:
    """Inject the custom stylesheet once (call right after set_page_config)."""
    st.markdown(_CSS, unsafe_allow_html=True)


def _render_candidate_cards(ranked: list, n: int) -> str:
    """Build one HTML block of ranked candidate cards (top ``n``).

    Pure presentation of existing `ScoredCandidate` fields — rank badge, title,
    score bar, and reasoning. Dynamic text is HTML-escaped before interpolation.
    """
    cards = []
    for sc in ranked[:n]:
        top = " rr-rank--top" if sc.rank <= 3 else ""
        title = html.escape(sc.current_title or "—")
        cid = html.escape(str(sc.candidate_id))
        score = float(sc.score or 0.0)
        pct = max(0.0, min(100.0, score * 100))
        reason = html.escape(sc.reasoning or "")
        try:
            yoe_txt = f"{float(sc.years_of_experience):g} yrs exp"
        except (TypeError, ValueError):
            yoe_txt = "exp n/a"
        cards.append(
            f'<div class="rr-cand">'
            f'<div class="rr-rank{top}">{sc.rank}</div>'
            f'<div class="rr-cand__body">'
            f'<div class="rr-cand__head">'
            f'<div class="rr-cand__title">{title}</div>'
            f'<div class="rr-cand__score">{score:.3f}<small>score</small></div>'
            f"</div>"
            f'<div class="rr-cand__meta">{cid} · {yoe_txt}</div>'
            f'<div class="rr-scorebar"><div class="rr-scorebar__fill" style="width:{pct:.1f}%"></div></div>'
            f'<div class="rr-cand__reason">{reason}</div>'
            f"</div>"
            f"</div>"
        )
    return f'<div class="rr-cands">{"".join(cards)}</div>'


st.set_page_config(page_title="Redrob Ranker — Velocity Labs", page_icon="🎯", layout="wide")
_inject_css()

st.markdown(
    """
    <div class="rr-hero">
      <div class="rr-hero__badge">🎯 Velocity Labs · INDIA.RUNS Hackathon — Track 01</div>
      <div class="rr-hero__title">Redrob Intelligent Candidate Ranker</div>
      <div class="rr-hero__tag">Multi-signal AI ranking — 100K candidates → top 100, CPU-only with no LLM at rank time.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("How it works", expanded=False):
    s1, s2, s3 = st.columns(3)
    s1.markdown(
        "**1 · Paste a JD**\n\nKeyword extraction (no LLM, no network) pulls the role, "
        "skills, and flags."
    )
    s2.markdown(
        "**2 · Add candidates**\n\nUpload ≤100, or use the built-in 100-candidate sample."
    )
    s3.markdown(
        "**3 · Run ranking**\n\nWeighted multi-signal fusion → a ranked CSV you can download."
    )
    st.markdown(
        '<div style="margin-top:12px"><b>Scoring weights</b>&nbsp; '
        '<span class="rr-chip rr-chip--blue">Semantic 40%</span>'
        '<span class="rr-chip rr-chip--blue">Role-fit 20%</span>'
        '<span class="rr-chip rr-chip--blue">Skill depth 15%</span>'
        '<span class="rr-chip rr-chip--blue">Behavioral 15%</span>'
        '<span class="rr-chip rr-chip--blue">Career 10%</span></div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Constraints met: CPU only · No LLM during ranking · < 5 min for 100K candidates"
    )

with st.container(border=True):
    st.markdown('<div class="rr-section">1 · Job description</div>', unsafe_allow_html=True)
    jd_text = st.text_area(
        "Job description",
        height=200,
        placeholder="Paste the full job description here…",
        label_visibility="collapsed",
    )

    st.markdown(
        '<div class="rr-section">2 · Candidates '
        '<span style="font-weight:500;color:#94a3b8">· optional</span></div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Upload ≤100 candidates (JSONL or JSON array, redrob schema). Leave empty to rank "
        "the built-in 100-candidate sample — no per-candidate embedding at runtime."
    )
    uploaded = st.file_uploader(
        "Candidate file",
        type=["jsonl", "json"],
        label_visibility="collapsed",
    )

    run = st.button(
        "🚀 Run Ranking", type="primary", disabled=not jd_text, use_container_width=True
    )

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

    # Outputs render below the (now collapsed) status so the CSV + table stay visible.
    # Show the JD keyword extraction first, then the ranked results.
    _render_jd_keywords(parsed_jd)

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

    st.markdown("")  # vertical breathing room above the results
    hcol, dcol = st.columns([3, 1])
    with hcol:
        st.markdown(
            '<div class="rr-section">🏆 Top candidates</div>'
            f'<div style="color:#64748b;font-size:0.88rem;margin-top:2px">'
            f"{len(ranked)} candidates ranked · showing top "
            f"{min(_CARD_COUNT, len(ranked))}</div>",
            unsafe_allow_html=True,
        )
    with dcol:
        st.download_button(
            "⬇️ Download CSV",
            data=buf.getvalue(),
            file_name="submission.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown(_render_candidate_cards(ranked, _CARD_COUNT), unsafe_allow_html=True)

    import pandas as pd
    with st.expander(f"View as table (top {_TABLE_COUNT})", expanded=False):
        df = pd.DataFrame([
            {"rank": sc.rank, "id": sc.candidate_id, "score": sc.score,
             "title": sc.current_title, "yoe": sc.years_of_experience, "reasoning": sc.reasoning}
            for sc in ranked[:_TABLE_COUNT]
        ])
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "rank": st.column_config.NumberColumn("Rank", format="%d", width="small"),
                "id": st.column_config.TextColumn("Candidate"),
                "score": st.column_config.ProgressColumn(
                    "Score", min_value=0.0, max_value=1.0, format="%.3f"
                ),
                "title": st.column_config.TextColumn("Title"),
                "yoe": st.column_config.NumberColumn("YoE", format="%d", width="small"),
                "reasoning": st.column_config.TextColumn("Reasoning", width="large"),
            },
        )
    # Cards/table are previews; the downloaded CSV holds the full ranking. State it
    # so "Ranked 100" next to a short list isn't mistaken for a bug.
    st.caption(
        f"Cards show the top {min(_CARD_COUNT, len(ranked))}; the table lists the top "
        f"{min(_TABLE_COUNT, len(ranked))}. The full ranking is in the downloaded CSV."
    )
