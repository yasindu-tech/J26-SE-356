#!/bin/sh
# Container start: bring the schema up to date, create any missing development accounts,
# then serve. The accounts file is mounted by docker-compose.yml as a secret; without it,
# no accounts are created and the API still starts.
set -e

alembic upgrade head

if [ -f /run/secrets/dev_accounts ]; then
  python -m app.seed --from-file /run/secrets/dev_accounts
else
  echo "No dev-accounts.json mounted: no accounts were seeded."
fi

# UVICORN_RELOAD=1 (set in docker-compose.yml for development) restarts the API whenever a
# synced source file changes; see "docker compose watch" in services/backend/README.md.
if [ "${UVICORN_RELOAD:-0}" = "1" ]; then
  exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
fi
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
