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
