from fastapi.testclient import TestClient

from app.models.clinician import Clinician
from app.models.refresh_token import RefreshToken


def test_login_succeeds_with_correct_credentials(
    client: TestClient, seeded_clinician: Clinician
) -> None:
    response = client.post(
        "/api/v1/auth/login",
        json={"username": "j.fernando", "password": "correct-horse-battery-staple"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["access_token"]
    assert body["refresh_token"]
    assert body["expires_in"] > 0


def test_login_fails_with_wrong_password(client: TestClient, seeded_clinician: Clinician) -> None:
    response = client.post(
        "/api/v1/auth/login",
        json={"username": "j.fernando", "password": "wrong-password"},
    )

    assert response.status_code == 401


def test_login_fails_with_unknown_username(client: TestClient) -> None:
    response = client.post(
        "/api/v1/auth/login",
        json={"username": "nobody", "password": "whatever"},
    )

    assert response.status_code == 401


def test_login_error_message_does_not_reveal_which_field_was_wrong(
    client: TestClient, seeded_clinician: Clinician
) -> None:
    wrong_password = client.post(
        "/api/v1/auth/login", json={"username": "j.fernando", "password": "wrong"}
    ).json()
    unknown_user = client.post(
        "/api/v1/auth/login", json={"username": "nobody", "password": "wrong"}
    ).json()

    assert wrong_password["error"] == unknown_user["error"]


def test_refresh_issues_a_new_working_token_pair(
    client: TestClient, seeded_clinician: Clinician
) -> None:
    login = client.post(
        "/api/v1/auth/login",
        json={"username": "j.fernando", "password": "correct-horse-battery-staple"},
    ).json()

    response = client.post("/api/v1/auth/refresh", json={"refresh_token": login["refresh_token"]})

    assert response.status_code == 200
    assert response.json()["access_token"] != login["access_token"]


def test_refresh_rejects_an_already_used_token(
    client: TestClient, seeded_clinician: Clinician
) -> None:
    """Guards the rotation logic: reusing a refresh token must fail, since
    that's what makes a stolen-and-replayed token detectable."""
    login = client.post(
        "/api/v1/auth/login",
        json={"username": "j.fernando", "password": "correct-horse-battery-staple"},
    ).json()

    first_use = client.post("/api/v1/auth/refresh", json={"refresh_token": login["refresh_token"]})
    second_use = client.post("/api/v1/auth/refresh", json={"refresh_token": login["refresh_token"]})

    assert first_use.status_code == 200
    assert second_use.status_code == 401


def test_refresh_rejects_an_unknown_token(client: TestClient) -> None:
    response = client.post("/api/v1/auth/refresh", json={"refresh_token": "not-a-real-token"})

    assert response.status_code == 401


def test_logout_revokes_the_refresh_token(client: TestClient, seeded_clinician: Clinician) -> None:
    """Guards that /logout actually revokes server-side, not just a client
    forgetting its token — the whole reason RefreshToken is a DB table."""
    login = client.post(
        "/api/v1/auth/login",
        json={"username": "j.fernando", "password": "correct-horse-battery-staple"},
    ).json()

    logout_response = client.post(
        "/api/v1/auth/logout", json={"refresh_token": login["refresh_token"]}
    )
    reuse_attempt = client.post(
        "/api/v1/auth/refresh", json={"refresh_token": login["refresh_token"]}
    )

    assert logout_response.status_code == 204
    assert reuse_attempt.status_code == 401


def test_logout_with_an_already_invalid_token_is_a_no_op(client: TestClient) -> None:
    response = client.post("/api/v1/auth/logout", json={"refresh_token": "garbage"})

    assert response.status_code == 204


def test_refresh_tokens_are_never_stored_in_plaintext(
    client: TestClient, seeded_clinician: Clinician, db_session
) -> None:
    login = client.post(
        "/api/v1/auth/login",
        json={"username": "j.fernando", "password": "correct-horse-battery-staple"},
    ).json()

    stored = (
        db_session.query(RefreshToken)
        .filter(RefreshToken.clinician_id == seeded_clinician.id)
        .one()
    )

    assert stored.token_hash != login["refresh_token"]


def test_deactivated_clinician_cannot_refresh(
    client: TestClient, seeded_clinician: Clinician, db_session
) -> None:
    login = client.post(
        "/api/v1/auth/login",
        json={"username": "j.fernando", "password": "correct-horse-battery-staple"},
    ).json()

    seeded_clinician.is_active = False
    db_session.commit()

    response = client.post("/api/v1/auth/refresh", json={"refresh_token": login["refresh_token"]})

    assert response.status_code == 401


def test_replaying_a_rotated_refresh_token_revokes_the_whole_session(
    client: TestClient, seeded_clinician: Clinician
) -> None:
    """A rotated-away token being replayed looks like theft, so the next
    legitimate refresh (using the token issued by the first, legitimate use)
    must also be rejected — not just the replay itself."""
    login = client.post(
        "/api/v1/auth/login",
        json={"username": "j.fernando", "password": "correct-horse-battery-staple"},
    ).json()

    first_use = client.post(
        "/api/v1/auth/refresh", json={"refresh_token": login["refresh_token"]}
    ).json()
    replay = client.post("/api/v1/auth/refresh", json={"refresh_token": login["refresh_token"]})
    legitimate_next_use = client.post(
        "/api/v1/auth/refresh", json={"refresh_token": first_use["refresh_token"]}
    )

    assert replay.status_code == 401
    assert legitimate_next_use.status_code == 401


def test_me_returns_the_signed_in_clinician(
    client: TestClient, seeded_clinician: Clinician
) -> None:
    login = client.post(
        "/api/v1/auth/login",
        json={"username": "j.fernando", "password": "correct-horse-battery-staple"},
    ).json()

    response = client.get(
        "/api/v1/auth/me", headers={"Authorization": f"Bearer {login['access_token']}"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["username"] == "j.fernando"
    assert body["full_name"] == "J. Fernando"


def test_me_without_a_token_is_not_authenticated(client: TestClient) -> None:
    response = client.get("/api/v1/auth/me")

    assert response.status_code in (401, 403)
    body = response.json()
    assert "error" in body
    assert "code" in body["error"]


def test_login_error_envelope_has_the_shared_shape(
    client: TestClient, seeded_clinician: Clinician
) -> None:
    response = client.post("/api/v1/auth/login", json={"username": "nobody", "password": "wrong"})

    body = response.json()
    assert body == {
        "error": {
            "code": "INVALID_CREDENTIALS",
            "message": "Incorrect username or password",
            "details": None,
        }
    }


def test_login_with_a_too_long_password_is_a_validation_error(client: TestClient) -> None:
    response = client.post(
        "/api/v1/auth/login", json={"username": "j.fernando", "password": "x" * 1000}
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"
