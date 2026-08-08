# Running RetailIQ in Docker

## Setup
Copy these 3 files into your project root (same folder as `app.py`):
- `Dockerfile`
- `.dockerignore`
- `docker-compose.yml`

## Run it
```bash
docker compose up --build
```
Then open http://localhost:8502

Stop with `Ctrl+C`, or `docker compose down` if run detached (`-d`).

## Without compose
```bash
docker build -t retailiq .
docker run -p 8502:8501 retailiq
```

## Why this works "the same in every environment"
- The image pins Python 3.11 and installs exact versions from `requirements.txt`,
  so you're not relying on whatever Python/pandas/scikit-learn happens to be on
  a given machine — teammates, CI, or a cloud VM all get the identical runtime.
- `.dockerignore` keeps the 44MB raw Excel/zip and the notebooks out of the
  image — the container only ships what `app.py` actually reads at runtime
  (`retail_clean.csv`, `rfm_segments.csv`, `monthly_revenue.csv`, the 3 `.pkl`
  models), so builds stay fast and the image stays small.
- `HEALTHCHECK` hits Streamlit's `/_stcore/health` endpoint, so
  `docker ps` / any orchestrator (Kubernetes, ECS, etc.) can tell if the app
  is actually serving, not just that the process hasn't crashed.

## Notes on your current setup
- `requirements.txt` lists `sqlalchemy` and `psycopg2-binary`, but `app.py`
  doesn't import or use either — it reads local CSVs/pickles only. I left
  them in requirements.txt (didn't touch it) but the image doesn't need a
  Postgres service for the app to run. If you're not using Postgres
  anywhere else in the project, you can drop those two lines to slim the
  image further.
- If you later add a real database, add a `db:` service to
  `docker-compose.yml` and point `app.py` at it via an environment variable
  (e.g. `DATABASE_URL`) rather than hardcoding credentials.
