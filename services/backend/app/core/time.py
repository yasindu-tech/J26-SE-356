from datetime import UTC, datetime


def utc_now() -> datetime:
    """Naive UTC now — deliberately drops tzinfo.

    SQLite (used in tests) doesn't preserve timezone-aware datetimes the way
    Postgres does, so comparing a tz-aware "now" against a value read back
    from the DB raises TypeError there. Storing and comparing naive-but-
    always-UTC datetimes everywhere sidesteps that instead of special-casing
    SQLite. PyJWT accepts naive datetimes for `exp` the same way.
    """
    return datetime.now(UTC).replace(tzinfo=None)
