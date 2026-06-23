# Sandbox Dockerfile — for HuggingFace Spaces / Streamlit demo
# Handles ≤100 candidate input, runs ranking end-to-end, produces CSV
FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps (no GPU packages)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model into the image cache at BUILD time (network
# available here). Lets rank.py run fully offline at runtime — no live HF calls.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-base-en-v1.5')"

# Runtime is offline: HF loads the baked-in cache only (submission_spec §3).
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

# Copy source
COPY src/ ./src/
COPY rank.py precompute.py validate_submission.py ./
COPY data/sample_candidates.json ./data/

# Streamlit demo app
COPY scripts/demo_app.py ./demo_app.py

# HF Spaces routes Docker apps to port 7860 by default; listen there so the
# Space needs no app_port override. Locally: `docker run -p 7860:7860`.
EXPOSE 7860

ENV PYTHONPATH=/app
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_HEADLESS=true

CMD ["streamlit", "run", "demo_app.py"]
