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
