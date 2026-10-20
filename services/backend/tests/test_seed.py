import pytest
from sqlalchemy.orm import Session

from app.core.security import verify_password
from app.models.clinician import Clinician
from app.seed import SeedError, main, seed_clinician


def test_creates_an_account_that_can_sign_in(db_session: Session) -> None:
    assert seed_clinician(db_session, "yasindu", "long-enough-pw", "Yasindu") == "created"
    row = db_session.query(Clinician).filter_by(username="yasindu").one()
    assert verify_password("long-enough-pw", row.hashed_password)
    assert row.is_active and row.full_name == "Yasindu"


def test_leaves_an_existing_account_alone_unless_asked(db_session: Session) -> None:
    seed_clinician(db_session, "yasindu", "first-password", "Yasindu")
    assert seed_clinician(db_session, "yasindu", "second-password", "Yasindu") == "exists"
    row = db_session.query(Clinician).filter_by(username="yasindu").one()
    assert verify_password("first-password", row.hashed_password)

    assert (
        seed_clinician(db_session, "yasindu", "second-password", "Yasindu", reset_password=True)
        == "reset"
    )
    db_session.refresh(row)
    assert verify_password("second-password", row.hashed_password)


def test_trims_the_username_and_refuses_spaces_or_short_passwords(db_session: Session) -> None:
    assert seed_clinician(db_session, "  j.fernando ", "long-enough-pw", "") == "created"
    assert (
        db_session.query(Clinician).filter_by(username="j.fernando").one().full_name == "j.fernando"
    )
    with pytest.raises(SeedError, match="no spaces"):
        seed_clinician(db_session, "jane doe", "long-enough-pw", "")
    with pytest.raises(SeedError, match="at least"):
        seed_clinician(db_session, "short", "1234567", "")


def test_refuses_to_run_in_production(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.core.config as config_module

    monkeypatch.setenv("ENVIRONMENT", "production")
    config_module.get_settings.cache_clear()
    try:
        assert main(["--username", "anyone"]) == 2
    finally:
        config_module.get_settings.cache_clear()
