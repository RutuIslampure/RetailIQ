# ── RetailIQ Dashboard — Docker image ──────────────────────────────
# Base: slim Python image keeps the image small but still has the
# system libs pandas/scikit-learn/psycopg2 need to build.
FROM python:3.11-slim

# Prevents .pyc files + forces stdout/stderr to be unbuffered
# (so `docker logs` shows Streamlit output in real time)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps: gcc/libpq-dev to build psycopg2, curl for the healthcheck.
# --no-install-recommends + cleaning apt lists keeps the image lean.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements file first — Docker caches this layer,
# so `docker build` won't reinstall every package unless
# requirements.txt actually changes. This is the single biggest
# speedup for repeat builds.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the app (code, data, trained models)
COPY . .

# Streamlit's default port
EXPOSE 8501

# Lets `docker ps` / orchestrators know if the app is actually alive,
# not just that the container process is running.
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0"]
