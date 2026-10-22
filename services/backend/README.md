# services/backend — FastAPI

Orchestration between the apps and the model modules: sessions, uploads,
inference, results.

## Structure
```
app/
  api/routes/   HTTP endpoints
  core/         Config, logging, security, errors
  models/       Persistence models
  schemas/      Pydantic request/response (mirror packages/shared types)
  services/     Business logic — assessment lifecycle, fusion orchestration
  workers/      Long-running inference jobs (MRI takes minutes)
  alembic/      Database migrations (env.py, versions/)
alembic.ini
tests/
```

## Rules
- **The feature-contract validator runs server-side too**, before any model is
  invoked. Client-side enforcement is a convenience; the server is the gate.
- Inference is **async** — behavioural models return in <30s, MRI takes minutes.
  Never block a request on MRI inference.
- Subject identifiers are **pseudonymised at ingest**. Raw identifiers never
  reach logs, traces or error reports.
- Every response carrying a risk score must also carry the modality breakdown
  and the availability mask that produced it.
- The API never returns the word "diagnosis" in any field, enum or message.

## Run
```bash
source .venv/bin/activate
uvicorn app.main:app --reload --port 8000 --app-dir services/backend
pytest services/backend
```

## Run with Docker

`docker-compose.yml` at the repo root starts Postgres and this API. Docker Desktop must be
running. Everything is for local development: ports are bound to `127.0.0.1`, and the default
passwords and JWT secret are dev-only.

**Accounts.** Once, copy `dev-accounts.example.json` to `dev-accounts.json` (git-ignored)
and set a real password for each person. Every time the backend container starts it
creates the accounts that are missing and leaves existing ones unchanged. Accounts live in
the database volume, so they survive `docker compose down` / `up`; only
`docker compose down -v` deletes them, and the next start creates them again.

**A. Database and API in Docker** (sign-in and the app's screens; MRI and gait analyses
do not run in the container yet):

```bash
docker compose up -d --build
npm run dev:desktop
```

**Editing the backend while it runs in Docker.** Start it with watch instead of `up`:

```bash
docker compose watch
```

It starts both services (building if needed) and keeps watching `services/backend`. When
you save a file in `app/`, it is copied into the container and the API reloads in a second
or two. A new migration in `app/alembic/` is copied and the container restarts, which
applies it. A change to `requirements.txt`, the `Dockerfile` or `docker-start.sh` rebuilds
the image. Stop with Ctrl+C (the containers keep running; `docker compose down` stops them).
This needs a recent Docker Desktop (Compose 2.22 or later; check with
`docker compose version`).

**B. Database in Docker, API on your Mac** (needed for MRI analyses, which use the FSL,
ANTs, dcm2niix and SynthStrip installed on your machine):

```bash
docker compose up -d db
export DATABASE_URL=postgresql+psycopg2://pdxai:pdxai@localhost:5433/pdxai
(cd services/backend && alembic upgrade head)
(cd services/backend && python -m app.seed --from-file ../../dev-accounts.json)
npm run dev:backend
```

For one account by hand: `python -m app.seed --username NAME --full-name "Name"` asks for the
password twice and never echoes it; add `--reset-password` to change an existing password.
The seed command creates accounts only, never patients or results, refuses the example's
placeholder passwords, and refuses to run with `ENVIRONMENT=production`.

Postgres listens on host port **5433**, so it does not clash with a Postgres already on 5432.
The data lives in the `pdxai_pgdata` volume: `docker compose down` keeps it,
`docker compose down -v` deletes it. Nothing under `data/` is copied into an image: the
repo-root `.dockerignore` allows `services/backend/` only.

## MRI analyses

`POST /api/v1/mri/analyses` takes one person's MRI as a zip of DICOM files plus age and
sex (multipart form), and answers `202` with an analysis ID. The analysis runs
`models/mri/src/predict_scan.py` in a separate process, one at a time
(`app/workers/mri_runner.py`). It takes minutes for T1 and much longer with DTI.

| Endpoint | Returns |
|---|---|
| `GET /api/v1/mri/analyses` | your analyses, newest first |
| `GET /api/v1/mri/analyses/{id}` | status and progress: steps, done, current step, note |
| `GET /api/v1/mri/analyses/{id}/result` | the `MriResult`: feature vector (raw and harmonised), raw score in log-odds with 95% interval, explanation, model card. `409` until finished |

The upload form also takes an optional `eddy_profile` field: `full` (default) or `fast`. It
sets the DTI motion-correction settings. `full` is what the model was trained on. `fast`
does fewer eddy iterations, is quicker, and is **not validated** against `full`; it is
stored on the analysis and shown on the result so a score is never read without it.

It also takes an optional `subject_label`: a pseudonymous label (letters, digits, `-` and `_`,
up to 32 characters) so the app can group one person's analyses. A name or anything with a space
is rejected. Never put a name or date of birth here.

Before the first analysis, the MRI model bundle must exist (`python models/mri/src/mri_bundle.py`)
and the backend's Python needs the MRI dependencies, FSL, ANTs, dcm2niix and the SynthStrip
Docker wrapper (see `models/mri/README.md`).

- **Where files go:** uploads and working files live under `data/backend/mri/<id>/`, which
  is git-ignored.
- **When they are deleted:** the DICOM zip and every converted image are deleted when the
  analysis ends. The status and result files are deleted once their result has been stored
  in the database.
- **Settings** (environment variables): `MRI_WORK_DIR`, `MRI_PYTHON`, `MRI_MAX_UPLOAD_MB`,
  `MRI_EDDY_THREADS`, `MRI_TIMEOUT_HOURS`.
- **Prototype limits:** this stands in for the Redis queue and ML worker in Backend-Spec,
  with the same submit-then-poll contract. An analysis that is running when the backend
  restarts is marked failed.

## Gait analyses

`POST /api/v1/gait/analyses` takes one walking video (`.mp4`, `.mov`, `.m4v`, `.avi`) and
`one_person_confirmed=true` (multipart form), plus the same optional `subject_label` as MRI, and
answers `202` with an analysis ID. The video
runs through `models/gait/src/predict_video.py` in a separate process, one at a time, in its own
worker (`app/workers/gait_runner.py`); it takes about a minute.

| Endpoint | Returns |
|---|---|
| `GET /api/v1/gait/analyses` | your analyses, newest first |
| `GET /api/v1/gait/analyses/{id}` | status (`scored`, `abstained` with the reason, `failed`) and elapsed time |
| `GET /api/v1/gait/analyses/{id}/result` | the `GaitResult` (`gait-result/1`). `409` until finished |
| `GET /api/v1/gait/analyses/{id}/overlay` | `video/mp4`: the walk with the skeleton and heel strikes drawn on it. `404` if none |

- The video is saved under its analysis ID (its own name is never kept) in `data/backend/gait/<id>/`
  and deleted when the analysis ends; the overlay video drawn from it is kept in
  `data/backend/gait_overlays/<id>.mp4`. The gait model bundle must exist (`train_xgboost.py`).
- **Settings:** `GAIT_WORK_DIR`, `GAIT_PYTHON`, `GAIT_SCRIPT`, `GAIT_BUNDLE_DIR`, `GAIT_OVERLAY_SCRIPT`, `GAIT_OVERLAY_DIR`,
  `GAIT_MAX_UPLOAD_MB` (500), `GAIT_TIMEOUT_MINUTES` (15).

## Finger-tapping analyses

`POST /api/v1/tapping/analyses` takes one finger-tapping video (`.mp4`, `.mov`, `.m4v`; multipart
field `file`), an optional `target` (`ge2`, the primary study target, or `ge1`) and the same optional
`subject_label` as MRI, and answers `202` with an analysis ID. The video runs through
`models/tapping/src/predict_clip.py` in a separate process, one at a time, in its own worker
(`app/workers/tapping_runner.py`); it takes under a minute. Same submit-then-poll contract as gait.

| Endpoint | Returns |
|---|---|
| `GET /api/v1/tapping/analyses` | your analyses, newest first |
| `GET /api/v1/tapping/analyses/{id}` | status (`scored`, `abstained` with the reason, `failed`) and elapsed time |
| `GET /api/v1/tapping/analyses/{id}/result` | the `TappingResult` (`tapping-result/1`): score, contributions, curves, one-page PDF (base64). `409` until finished |

- The video is saved under its analysis ID (its own name is never kept) in
  `data/backend/tapping/<id>/` and deleted when the analysis ends. The result is stored in the
  `tapping_analyses` table (PostgreSQL in the Docker stack; migration `d4a7c1e95b28`).
- A video the program read but could not score (no hand found, too long) is `abstained` with its
  plain reason. `failed` means the analysis itself could not run (timeout, program missing, backend restart).
- Like MRI and gait, it does not run inside the Docker `backend` container (no `lightgbm`,
  `mediapipe` or `shap` there). Run the backend natively against the Docker database (see "Run with
  Docker" above): `docker compose stop backend`, then start `uvicorn` from the repository's `.venv`.
  It needs `data/processed`, the demo models (`models/tapping/demo/train_demo_model.py`) and
  `models/tapping/artifacts/hand_landmarker.task`.
- **Settings:** `TAPPING_WORK_DIR`, `TAPPING_PYTHON`, `TAPPING_SCRIPT`, `TAPPING_MAX_UPLOAD_MB` (200),
  `TAPPING_TIMEOUT_MINUTES` (10).

## Voice analyses

`POST /api/v1/voice/analyses` takes one recording of a sustained "aaah" (`.wav`, `.flac`, `.ogg`,
`.mp3`, `.m4a`, `.aac`, `.webm`; multipart field `file`) and answers `200` with the voice
module's result: score, 95% range, triage priority and explanations, or an abstained result with
the reasons. It runs `models/voice/src/analyze_audio.py` in a separate process
(`app/workers/voice_runner.py`). A recording takes a few seconds, so the request waits for it.

- The recording is saved under a fresh ID (its own name is never kept) in
  `data/backend/voice/<id>/` and deleted before the response is sent, whatever happened. The
  result is not stored; an audit entry (`voice_analysis`) records that an analysis was made.
- `503 VOICE_ANALYSIS_FAILED` means the analysis could not run (no trained model, timeout), not
  that the voice was judged. Train the model first: `python models/voice/src/audio_model.py`.
  MP3/M4A need `ffmpeg` on the PATH.
- **Settings:** `VOICE_WORK_DIR`, `VOICE_PYTHON`, `VOICE_SCRIPT`, `VOICE_MODEL` (`audio` or
  `uci`), `VOICE_MAX_UPLOAD_MB` (50), `VOICE_TIMEOUT_SECONDS` (120).

## Database migrations

Schema changes go through Alembic, never `Base.metadata.create_all()` outside
tests (tests use `create_all` against in-memory SQLite deliberately, for
speed and isolation — see `tests/conftest.py`).

```bash
cd services/backend
alembic upgrade head                                    # apply all migrations
alembic revision --autogenerate -m "add x to clinicians" # after changing a model
alembic downgrade -1                                     # roll back one step
```

`alembic.ini` has no database URL — `app/alembic/env.py` always reads
`Settings.database_url` (env vars / `.env`), so migrations run against
whatever database the app itself is configured for.
