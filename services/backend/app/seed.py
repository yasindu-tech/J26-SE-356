"""Create (or reset) a clinician account for local development.

    python -m app.seed --username yasindu --full-name "Yasindu"
    docker compose exec backend python -m app.seed --username yasindu --full-name "Yasindu"

The password is asked for at the prompt and is never echoed, so it does not end up in your
shell history. For scripts, set SEED_PASSWORD instead. Accounts only: this never creates
patients, scans or results, so there is no fake data in the database.

Refuses to run when ENVIRONMENT=production. Accounts there are provisioned by an
administrator, not by a development command.
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from typing import Literal

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.security import hash_password
from app.models.clinician import Clinician

MIN_PASSWORD_LENGTH = 8

Outcome = Literal["created", "reset", "exists"]


class SeedError(Exception):
    pass


def seed_clinician(
    db: Session,
    username: str,
    password: str,
    full_name: str,
    *,
    reset_password: bool = False,
) -> Outcome:
    """Creates the account, or resets its password when asked. Never changes an existing
    account silently: without `reset_password` an existing username is left as it is."""
    username = username.strip()
    if not username or username != username.split()[0]:
        raise SeedError("The username must be one word with no spaces.")
    if len(password) < MIN_PASSWORD_LENGTH:
        raise SeedError(f"The password must be at least {MIN_PASSWORD_LENGTH} characters.")

    existing = db.query(Clinician).filter(Clinician.username == username).first()
    if existing is not None:
        if not reset_password:
            return "exists"
        existing.hashed_password = hash_password(password)
        existing.is_active = True
        db.commit()
        return "reset"

    db.add(
        Clinician(
            username=username,
            hashed_password=hash_password(password),
            full_name=full_name.strip() or username,
        )
    )
    db.commit()
    return "created"


def _password() -> str:
    from_env = os.environ.get("SEED_PASSWORD")
    if from_env:
        return from_env
    first = getpass.getpass("Password for the account: ")
    if first != getpass.getpass("Same password again: "):
        raise SeedError("The two passwords did not match.")
    return first


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--username", required=True)
    parser.add_argument("--full-name", default="")
    parser.add_argument(
        "--reset-password", action="store_true", help="set a new password if the account exists"
    )
    args = parser.parse_args(argv)

    if get_settings().environment == "production":
        print("Refusing to seed accounts with ENVIRONMENT=production.", file=sys.stderr)
        return 2

    from app.core.database import SessionLocal  # only now: importing it reads the DB settings

    try:
        password = _password()
        with SessionLocal() as db:
            outcome = seed_clinician(
                db, args.username, password, args.full_name, reset_password=args.reset_password
            )
    except SeedError as error:
        print(error, file=sys.stderr)
        return 1

    messages = {
        "created": f"Created account {args.username!r}.",
        "reset": f"Reset the password for {args.username!r}.",
        "exists": f"{args.username!r} already exists; nothing changed. Add --reset-password to set a new one.",
    }
    print(messages[outcome])
    return 0


if __name__ == "__main__":
    sys.exit(main())
