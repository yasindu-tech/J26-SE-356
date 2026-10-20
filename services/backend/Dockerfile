# PD-XAI backend API (FastAPI). Build from the repo root:
#   docker compose build backend
# The image holds code and Python packages only. No data, scans or secrets are copied in
# (see the repo-root .dockerignore, which allows services/backend/ and nothing else).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Same layout as the repo, so app/core/config.py's REPO_ROOT resolves to /repo.
WORKDIR /repo/services/backend

COPY services/backend/requirements.txt ./
RUN pip install -r requirements.txt

COPY services/backend/ ./

# Run as a normal user, not root.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /repo/data/backend \
    && chown -R app /repo
USER app

EXPOSE 8000

# Bring the schema up to date, create missing dev accounts, then serve (docker-start.sh).
CMD ["sh", "/repo/services/backend/docker-start.sh"]
