import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


@pytest.fixture
def client(monkeypatch):
    """TestClient backed by an isolated in-memory DB, with OTP delivery captured."""
    import main
    from app.db import Base, get_db
    from app.api.routes import auth as auth_route

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSession = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(bind=engine)

    def override_get_db():
        db = TestingSession()
        try:
            yield db
        finally:
            db.close()

    main.app.dependency_overrides[get_db] = override_get_db

    box = {}
    # OTP delivery was refactored into a background task; auth.py imports
    # send_otp_email by name, so patch it in the route module's namespace.
    monkeypatch.setattr(auth_route, "send_otp_email", lambda email, code: box.update(code=code))

    c = TestClient(main.app)
    c.otp_box = box  # type: ignore[attr-defined]
    yield c
    main.app.dependency_overrides.clear()


SIGNUP = {
    "name": "Kumar",
    "email": "kumar@example.com",
    "password": "secret123",
    "phone": "9999999999",
}


def test_signup_sends_otp(client):
    r = client.post("/auth/signup", json=SIGNUP)
    assert r.status_code == 200
    assert "code" in client.otp_box  # OTP was issued
    assert len(client.otp_box["code"]) == 6


def test_signup_rejects_short_password(client):
    r = client.post("/auth/signup", json={**SIGNUP, "password": "123"})
    assert r.status_code == 400


def test_signup_rejects_bad_email(client):
    r = client.post("/auth/signup", json={**SIGNUP, "email": "not-an-email"})
    assert r.status_code == 422  # pydantic EmailStr validation


def test_signin_blocked_until_verified(client):
    client.post("/auth/signup", json=SIGNUP)
    r = client.post("/auth/signin", json={"email": SIGNUP["email"], "password": SIGNUP["password"]})
    assert r.status_code == 401


def test_verify_wrong_code_fails(client):
    client.post("/auth/signup", json=SIGNUP)
    r = client.post("/auth/verify-otp", json={"email": SIGNUP["email"], "code": "000000"})
    assert r.status_code == 400


def test_full_flow_signup_verify_signin_me(client):
    # 1. sign up
    client.post("/auth/signup", json=SIGNUP)
    code = client.otp_box["code"]

    # 2. verify with the real code → get a token
    r = client.post("/auth/verify-otp", json={"email": SIGNUP["email"], "code": code})
    assert r.status_code == 200
    token = r.json()["access_token"]
    assert r.json()["user"]["email"] == SIGNUP["email"]

    # 3. sign in
    r = client.post("/auth/signin", json={"email": SIGNUP["email"], "password": SIGNUP["password"]})
    assert r.status_code == 200
    assert r.json()["user"]["name"] == "Kumar"

    # 4. /me with token
    r = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    assert r.json()["email"] == SIGNUP["email"]

    # 5. /me without token → 401
    assert client.get("/auth/me").status_code == 401


def test_signin_wrong_password(client):
    client.post("/auth/signup", json=SIGNUP)
    client.post("/auth/verify-otp", json={"email": SIGNUP["email"], "code": client.otp_box["code"]})
    r = client.post("/auth/signin", json={"email": SIGNUP["email"], "password": "wrongpass"})
    assert r.status_code == 401


def test_duplicate_verified_signup_rejected(client):
    client.post("/auth/signup", json=SIGNUP)
    client.post("/auth/verify-otp", json={"email": SIGNUP["email"], "code": client.otp_box["code"]})
    r = client.post("/auth/signup", json=SIGNUP)
    assert r.status_code == 400


# ── Forgot / reset password ──
def _signup_and_verify(client):
    client.post("/auth/signup", json=SIGNUP)
    client.post("/auth/verify-otp", json={"email": SIGNUP["email"], "code": client.otp_box["code"]})


def test_forgot_password_sends_code(client):
    _signup_and_verify(client)
    client.otp_box.clear()
    r = client.post("/auth/forgot-password", json={"email": SIGNUP["email"]})
    assert r.status_code == 200
    assert "code" in client.otp_box  # a reset code was issued


def test_forgot_password_unknown_email_is_silent(client):
    client.otp_box.clear()
    r = client.post("/auth/forgot-password", json={"email": "nobody@example.com"})
    assert r.status_code == 200  # generic response
    assert "code" not in client.otp_box  # no code issued for a non-existent account


def test_reset_password_full_flow(client):
    _signup_and_verify(client)
    client.otp_box.clear()
    client.post("/auth/forgot-password", json={"email": SIGNUP["email"]})
    reset_code = client.otp_box["code"]

    r = client.post(
        "/auth/reset-password",
        json={"email": SIGNUP["email"], "code": reset_code, "new_password": "newsecret456"},
    )
    assert r.status_code == 200
    assert "access_token" in r.json()

    # old password rejected, new password works
    assert client.post("/auth/signin", json={"email": SIGNUP["email"], "password": SIGNUP["password"]}).status_code == 401
    assert client.post("/auth/signin", json={"email": SIGNUP["email"], "password": "newsecret456"}).status_code == 200


def test_reset_password_wrong_code(client):
    _signup_and_verify(client)
    client.post("/auth/forgot-password", json={"email": SIGNUP["email"]})
    r = client.post(
        "/auth/reset-password",
        json={"email": SIGNUP["email"], "code": "000000", "new_password": "newsecret456"},
    )
    assert r.status_code == 400
