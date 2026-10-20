#!/bin/sh
# Container start: bring the schema up to date, create any missing development accounts,
# then serve. The accounts file is mounted by docker-compose.yml as a secret; without it,
# no accounts are created and the API still starts.
set -e

alembic upgrade head

if [ -r /run/secrets/dev_accounts ]; then
  # A problem with the accounts file must not stop the API; it is reported loudly instead.
  python -m app.seed --from-file /run/secrets/dev_accounts \
    || echo "WARNING: the dev accounts were NOT created. Fix dev-accounts.json and run: docker compose restart backend"
elif [ -e /run/secrets/dev_accounts ]; then
  echo "WARNING: dev-accounts.json is mounted but not readable here; no accounts were created. On the host: chmod 644 dev-accounts.json"
else
  echo "No dev-accounts.json mounted: no accounts were seeded."
fi

# UVICORN_RELOAD=1 (set in docker-compose.yml for development) restarts the API whenever a
# synced source file changes; see "docker compose watch" in services/backend/README.md.
if [ "${UVICORN_RELOAD:-0}" = "1" ]; then
  exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
fi
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
