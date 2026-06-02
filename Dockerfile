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

# Copy source
COPY src/ ./src/
COPY rank.py precompute.py validate_submission.py ./
COPY data/sample_candidates.json ./data/

# Streamlit demo app
COPY scripts/demo_app.py ./demo_app.py

EXPOSE 8501

ENV PYTHONPATH=/app
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true

CMD ["streamlit", "run", "demo_app.py"]
