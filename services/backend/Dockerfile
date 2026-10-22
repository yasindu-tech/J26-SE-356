# PD-XAI backend API (FastAPI). Build from the repo root:
#   docker compose build backend
# The image holds code and Python packages only. No data, scans or secrets are copied in
# (see the repo-root .dockerignore, which allows services/backend/ and the voice module's
# code and nothing else). The trained voice model is mounted at run time, not copied
# (docker-compose.yml).

# Stage 1: parselmouth (Praat, used by voice analyses) publishes ready-made packages for
# x86_64 Linux only. On ARM Linux (Docker on Apple Silicon Macs) it must be compiled, so it
# is built here, with the compilers it needs, and only the finished package is copied into
# the final image. On x86_64 this stage just downloads the ready-made package.
FROM python:3.11-slim AS voice-wheels
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/*
COPY models/voice/requirements.txt /tmp/voice-requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip wheel --no-deps --wheel-dir /wheels \
        "$(grep -o '^praat-parselmouth[^ #]*' /tmp/voice-requirements.txt)"

# Stage 2: the API image.
FROM python:3.11-slim

# Slow or unstable connections: wait longer for each download and retry instead of failing.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=10 \
    NUMBA_CACHE_DIR=/tmp/numba-cache \
    MPLCONFIGDIR=/tmp/matplotlib

# Voice analyses (models/voice/src/analyze_audio.py, run by the API as a separate process):
# ffmpeg decodes MP3/M4A/AAC/WebM uploads; libgomp1 is the OpenMP runtime LightGBM needs.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Same layout as the repo, so app/core/config.py's REPO_ROOT resolves to /repo.
WORKDIR /repo/services/backend

COPY services/backend/requirements.txt ./
COPY models/voice/requirements.txt /repo/models/voice/requirements.txt
# The cache mount keeps packages that already downloaded, so a failed or repeated build
# does not fetch them again. It lives in Docker's build cache, not in the image.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,from=voice-wheels,source=/wheels,target=/wheels \
    pip install --find-links /wheels -r requirements.txt -r /repo/models/voice/requirements.txt

COPY services/backend/ ./
COPY models/voice/src/ /repo/models/voice/src/

# Run as a normal user, not root.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /repo/data/backend \
    && chown -R app /repo
USER app

EXPOSE 8000

# Bring the schema up to date, create missing dev accounts, then serve (docker-start.sh).
CMD ["sh", "/repo/services/backend/docker-start.sh"]
