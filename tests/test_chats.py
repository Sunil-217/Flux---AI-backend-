import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


@pytest.fixture
def auth_client(monkeypatch):
    """TestClient on an isolated in-memory DB, returning (client, auth_headers)."""
    import main
    from app.db import Base, get_db
    from app.services import auth_service

    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
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
    monkeypatch.setattr(auth_service, "send_otp_email", lambda email, code: box.update(code=code))

    c = TestClient(main.app)
    c.post("/auth/signup", json={
        "name": "Kumar", "email": "kumar@example.com",
        "password": "secret123", "phone": "9999999999",
    })
    r = c.post("/auth/verify-otp", json={"email": "kumar@example.com", "code": box["code"]})
    token = r.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    yield c, headers
    main.app.dependency_overrides.clear()


def test_chats_empty_by_default(auth_client):
    c, headers = auth_client
    r = c.get("/chats", headers=headers)
    assert r.status_code == 200
    assert r.json() == {"data": []}


def test_chats_save_and_load_roundtrip(auth_client):
    c, headers = auth_client
    sessions = [
        {"id": "s1", "title": "First chat", "messages": [{"id": "m1", "role": "user", "content": "hi"}]},
        {"id": "s2", "title": "Second", "messages": []},
    ]
    r = c.put("/chats", json={"data": sessions}, headers=headers)
    assert r.status_code == 200
    assert r.json()["ok"] is True

    r = c.get("/chats", headers=headers)
    assert r.status_code == 200
    assert r.json()["data"] == sessions


def test_chats_require_auth(auth_client):
    c, _ = auth_client
    assert c.get("/chats").status_code == 401
    assert c.put("/chats", json={"data": []}).status_code == 401


def test_chats_are_per_user(auth_client, monkeypatch):
    c, headers = auth_client
    # Save data for user 1
    c.put("/chats", json={"data": [{"id": "x"}]}, headers=headers)

    # Create a second user and confirm they start empty (isolation)
    from app.services import auth_service
    box2 = {}
    monkeypatch.setattr(auth_service, "send_otp_email", lambda email, code: box2.update(code=code))
    c.post("/auth/signup", json={
        "name": "Asha", "email": "asha@example.com",
        "password": "secret123", "phone": "8888888888",
    })
    r = c.post("/auth/verify-otp", json={"email": "asha@example.com", "code": box2["code"]})
    headers2 = {"Authorization": f"Bearer {r.json()['access_token']}"}

    assert c.get("/chats", headers=headers2).json() == {"data": []}
    # User 1's data is untouched
    assert c.get("/chats", headers=headers).json()["data"] == [{"id": "x"}]
