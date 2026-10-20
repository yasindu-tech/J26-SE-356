"""Create (or reset) a clinician account for local development.

    python -m app.seed --username yasindu --full-name "Yasindu"
    python -m app.seed --from-file ../../dev-accounts.json
    docker compose exec backend python -m app.seed --username yasindu --full-name "Yasindu"

With --from-file it creates every account in the file that does not exist yet and leaves
existing ones alone, so it is safe to run on every start (the Docker backend does).

The password is asked for at the prompt and is never echoed, so it does not end up in your
shell history. For scripts, set SEED_PASSWORD instead. Accounts only: this never creates
patients, scans or results, so there is no fake data in the database.

Refuses to run when ENVIRONMENT=production. Accounts there are provisioned by an
administrator, not by a development command.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from typing import Literal

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.security import hash_password
from app.models.clinician import Clinician

MIN_PASSWORD_LENGTH = 8
# The example file's placeholder. Refused, so a copied example never becomes a real password.
PLACEHOLDER_PREFIX = "CHANGE-ME"

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
    if password.upper().startswith(PLACEHOLDER_PREFIX):
        raise SeedError(
            f"Set a real password for {username!r}; the example placeholder is refused."
        )

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


def read_accounts(text: str) -> list[dict[str, str]]:
    """The accounts file: {"accounts": [{"username", "full_name", "password"}, ...]}."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise SeedError(f"The accounts file is not valid JSON: {error}") from error
    accounts = data.get("accounts") if isinstance(data, dict) else None
    if not isinstance(accounts, list) or not accounts:
        raise SeedError('The accounts file needs a non-empty "accounts" list.')
    out = []
    for i, entry in enumerate(accounts, start=1):
        if not isinstance(entry, dict) or not all(
            isinstance(entry.get(k), str) for k in ("username", "password")
        ):
            raise SeedError(f"Account {i} needs a text username and password.")
        out.append(
            {
                "username": entry["username"],
                "password": entry["password"],
                "full_name": str(entry.get("full_name", "")),
            }
        )
    return out


def seed_from_file(db: Session, text: str) -> dict[str, Outcome]:
    """Creates the accounts that are missing. Existing accounts are never changed."""
    accounts = read_accounts(text)
    # Check every entry before writing any, so a bad file changes nothing.
    for entry in accounts:
        if len(entry["password"]) < MIN_PASSWORD_LENGTH or entry["password"].upper().startswith(
            PLACEHOLDER_PREFIX
        ):
            raise SeedError(
                f"Set a real password of at least {MIN_PASSWORD_LENGTH} characters for "
                f"{entry['username']!r}."
            )
    return {
        entry["username"].strip(): seed_clinician(
            db, entry["username"], entry["password"], entry["full_name"]
        )
        for entry in accounts
    }


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
    who = parser.add_mutually_exclusive_group(required=True)
    who.add_argument("--username")
    who.add_argument("--from-file", help="JSON file of accounts (see dev-accounts.example.json)")
    parser.add_argument("--full-name", default="")
    parser.add_argument(
        "--reset-password", action="store_true", help="set a new password if the account exists"
    )
    args = parser.parse_args(argv)

    if get_settings().environment == "production":
        print("Refusing to seed accounts with ENVIRONMENT=production.", file=sys.stderr)
        return 2

    from app.core.database import SessionLocal  # only now: importing it reads the DB settings

    if args.from_file:
        try:
            with open(args.from_file, encoding="utf-8") as handle, SessionLocal() as db:
                outcomes = seed_from_file(db, handle.read())
        except (OSError, SeedError) as error:
            print(error, file=sys.stderr)
            return 1
        for username, outcome in outcomes.items():
            print(
                f"{username}: {'created' if outcome == 'created' else 'already exists, unchanged'}"
            )
        return 0

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
