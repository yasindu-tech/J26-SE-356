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

exec uvicorn app.main:app --host 0.0.0.0 --port 8000
