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

**A. Database and API in Docker** (sign-in and the app's screens; MRI and gait analyses
do not run in the container yet):

```bash
docker compose up -d --build
docker compose exec backend python -m app.seed --username yasindu --full-name "Yasindu"
npm run dev:desktop
```

**B. Database in Docker, API on your Mac** (needed for MRI analyses, which use the FSL,
ANTs, dcm2niix and SynthStrip installed on your machine):

```bash
docker compose up -d db
export DATABASE_URL=postgresql+psycopg2://pdxai:pdxai@localhost:5433/pdxai
(cd services/backend && alembic upgrade head)
(cd services/backend && python -m app.seed --username yasindu --full-name "Yasindu")
npm run dev:backend
```

The seed command asks for the password twice and never echoes it. It creates accounts only,
never patients or results. Run it again with `--reset-password` to change a password. It
refuses to run with `ENVIRONMENT=production`.

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
`one_person_confirmed=true` (multipart form), and answers `202` with an analysis ID. The video
runs through `models/gait/src/predict_video.py` in a separate process, one at a time, in its own
worker (`app/workers/gait_runner.py`); it takes about a minute.

| Endpoint | Returns |
|---|---|
| `GET /api/v1/gait/analyses` | your analyses, newest first |
| `GET /api/v1/gait/analyses/{id}` | status (`scored`, `abstained` with the reason, `failed`) and elapsed time |
| `GET /api/v1/gait/analyses/{id}/result` | the `GaitResult` (`gait-result/1`). `409` until finished |

- The video is saved under its analysis ID (its own name is never kept) in `data/backend/gait/<id>/`
  and deleted when the analysis ends. The gait model bundle must exist (`train_xgboost.py`).
- **Settings:** `GAIT_WORK_DIR`, `GAIT_PYTHON`, `GAIT_SCRIPT`, `GAIT_BUNDLE_DIR`,
  `GAIT_MAX_UPLOAD_MB` (500), `GAIT_TIMEOUT_MINUTES` (15).

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
