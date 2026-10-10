from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.database import Base, get_db
from app.core.security import hash_password
from app.main import app
from app.models.clinician import Clinician


@pytest.fixture
def db_session() -> Generator[Session, None, None]:
    # In-memory SQLite, one shared connection (StaticPool) so every Session
    # in a test sees the same database — never touches the real Postgres
    # config, so tests don't need a running database.
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)

    session = session_factory()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def client(db_session: Session) -> Generator[TestClient, None, None]:
    def override_get_db() -> Generator[Session, None, None]:
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


@pytest.fixture
def seeded_clinician(db_session: Session) -> Clinician:
    clinician = Clinician(
        username="j.fernando",
        hashed_password=hash_password("correct-horse-battery-staple"),
        full_name="J. Fernando",
    )
    db_session.add(clinician)
    db_session.commit()
    db_session.refresh(clinician)
    return clinician
