"""Streamlit sandbox demo — accepts ≤100 candidates, runs full ranking pipeline."""

import json
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Redrob Ranker — Velocity Labs", page_icon="🎯", layout="wide")

st.title("🎯 Redrob Intelligent Candidate Ranker")
st.caption("Velocity Labs · INDIA.RUNS Hackathon — Track 01")

with st.expander("How it works", expanded=False):
    st.markdown("""
    **Pipeline:**
    1. Upload a JSONL file with ≤100 candidates (redrob schema)
    2. Paste the job description
    3. Click **Run Ranking** — outputs a ranked CSV

    **Scoring weights:**
    Semantic 40% · Role-fit 20% · Skill depth 15% · Behavioral 15% · Career 10%

    **Constraints met:** CPU only · No LLM during ranking · < 5 min for 100K candidates
    """)

jd_text = st.text_area(
    "Job Description",
    height=200,
    placeholder="Paste the full job description here...",
)

uploaded = st.file_uploader("Candidate JSONL (≤100 candidates)", type=["jsonl", "json"])

if st.button("🚀 Run Ranking", type="primary", disabled=not (jd_text and uploaded)):
    with st.spinner("Parsing JD and ranking candidates..."):
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))

            raw_lines = uploaded.read().decode("utf-8").splitlines()
            candidates_raw = [json.loads(line) for line in raw_lines if line.strip()]

            from src.parsers.candidate import parse_redrob_candidate
            from src.parsers.jd import _keyword_fallback
            from src.embedder import get_embedder
            from src.index import build_index, query_index
            from src.ranker import RankingEngine

            parsed_jd = _keyword_fallback(jd_text)  # fast fallback for demo
            embedder = get_embedder()

            candidates = [parse_redrob_candidate(r) for r in candidates_raw]
            texts = [c["embedding_text"] for c in candidates]
            embeddings = embedder.embed_batch(texts)

            index = build_index(embeddings, [c["candidate_id"] for c in candidates])
            jd_vec = embedder.embed_text(parsed_jd.to_embedding_text())

            from src.config import TOP_K_RETRIEVE
            k = min(len(candidates), TOP_K_RETRIEVE)
            ann_results = query_index(index, [c["candidate_id"] for c in candidates], jd_vec, k)

            cands_by_id = {c["candidate_id"]: c for c in candidates}
            engine = RankingEngine()
            ranked = engine.rank(cands_by_id, ann_results, parsed_jd)

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

            st.success(f"✅ Ranked {len(ranked)} candidates")
            st.download_button(
                "⬇️ Download submission.csv",
                data=buf.getvalue(),
                file_name="submission.csv",
                mime="text/csv",
            )

            import pandas as pd
            df = pd.DataFrame([
                {"rank": sc.rank, "id": sc.candidate_id, "score": sc.score,
                 "title": sc.current_title, "yoe": sc.years_of_experience, "reasoning": sc.reasoning}
                for sc in ranked[:20]
            ])
            st.dataframe(df, use_container_width=True)

        except Exception as exc:
            st.error(f"Error: {exc}")
            raise
