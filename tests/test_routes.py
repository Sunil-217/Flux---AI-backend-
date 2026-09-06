from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


@pytest.fixture
def client():
    """Authenticated TestClient — bypasses get_current_user with a dummy user,
    on an isolated in-memory DB shared across threads (StaticPool)."""
    import main
    from app.core.security import get_current_user
    from app.db import Base, get_db

    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    TestingSession = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(bind=engine)

    # The chat/upload/delete routes enforce ownership (user_owns_chat); seed a
    # row for the dummy user (id=1) so the chat ids these tests use are "owned".
    import json as _json
    from app.models import UserChats

    _seed = TestingSession()
    _seed.add(UserChats(user_id=1, data=_json.dumps([{"id": "c1"}, {"id": "session-xyz"}])))
    _seed.commit()
    _seed.close()

    def override_get_db():
        db = TestingSession()
        try:
            yield db
        finally:
            db.close()

    main.app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(
        id=1, name="Test", email="test@example.com"
    )
    main.app.dependency_overrides[get_db] = override_get_db
    c = TestClient(main.app)
    yield c
    main.app.dependency_overrides.clear()


# ── /chat (streaming) ──
def test_chat_streams_events(client, monkeypatch):
    import app.api.routes.chat as chat_route
    from app.services import rag_service

    def fake_stream(chat_id, question, history, image=None, *args):
        yield rag_service._sse({"type": "token", "content": "Hi"})
        yield rag_service._sse({"type": "token", "content": " there"})
        yield rag_service._sse({"type": "done"})

    monkeypatch.setattr(chat_route, "stream_question", fake_stream)

    resp = client.post("/chat", json={"chat_id": "c1", "question": "hi", "history": []})
    assert resp.status_code == 200
    assert "Hi" in resp.text and "there" in resp.text and "done" in resp.text


def test_chat_forwards_history(client, monkeypatch):
    import app.api.routes.chat as chat_route
    from app.services import rag_service

    captured = {}

    def fake_stream(chat_id, question, history, image=None, *args):
        captured["history"] = history
        captured["question"] = question
        yield rag_service._sse({"type": "done"})

    monkeypatch.setattr(chat_route, "stream_question", fake_stream)

    client.post(
        "/chat",
        json={"chat_id": "c1", "question": "next?", "history": [{"role": "user", "content": "earlier"}]},
    )
    assert captured["question"] == "next?"
    assert captured["history"] == [{"role": "user", "content": "earlier"}]


def test_chat_requires_fields(client):
    resp = client.post("/chat", json={"question": "hi"})  # missing chat_id
    assert resp.status_code == 422


# ── /delete ──
def test_delete_success(client, monkeypatch):
    import app.api.routes.delete as delete_route

    called = {}
    monkeypatch.setattr(delete_route, "delete_collection", lambda cid: called.setdefault("id", cid))

    resp = client.delete("/delete/session-xyz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["message"] == "Chat deleted successfully"
    assert called["id"] == "session-xyz"


# ── /upload ──
def test_upload_rejects_unsupported_type(client):
    # The route now accepts many document/text types; only genuinely
    # unsupported extensions are rejected — with a 400 + detail.
    resp = client.post(
        "/upload",
        files={"file": ("malware.exe", b"MZ\x90\x00", "application/octet-stream")},
        data={"chat_id": "c1"},
    )
    assert resp.status_code == 400
    assert "Unsupported file type" in resp.json()["detail"]


def test_upload_accepts_pdf(client, monkeypatch, tmp_path):
    import app.api.routes.upload as upload_route

    monkeypatch.setattr(upload_route, "UPLOAD_DIR", str(tmp_path))

    resp = client.post(
        "/upload",
        files={"file": ("doc.pdf", b"%PDF-1.4 fake content", "application/pdf")},
        data={"chat_id": "c1"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("message") == "File uploaded successfully"
    assert body.get("filename") == "doc.pdf"


# ── root (public) ──
def test_home_route(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "message" in resp.json()


# ── auth enforcement (no override → must be rejected) ──
def test_protected_endpoints_require_auth():
    import main

    c = TestClient(main.app)  # no dependency override
    assert c.post("/chat", json={"chat_id": "x", "question": "hi", "history": []}).status_code == 401
    assert c.delete("/delete/x").status_code == 401
    assert (
        c.post("/upload", files={"file": ("a.pdf", b"x", "application/pdf")}, data={"chat_id": "x"}).status_code
        == 401
    )


# ── Health / keep-alive ──
def test_health_returns_ok_without_authentication(client):
    """Hit on a schedule by an external pinger, so it must not need a token."""
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_health_works_with_no_auth_header_at_all():
    """The `client` fixture overrides get_current_user. This one does not —
    a scheduler sends no Authorization header."""
    from fastapi.testclient import TestClient

    import main

    assert TestClient(main.app).get("/health").json() == {"status": "ok"}


def test_health_touches_no_database_llm_or_network(monkeypatch):
    """The whole point of a keep-alive is that it is nearly free. If it grew a
    dependency it would start costing quota — or start failing for reasons that
    have nothing to do with whether the process is up."""
    from fastapi.testclient import TestClient

    import main
    from app.db import SessionLocal  # noqa: F401

    def boom(*a, **k):
        raise AssertionError("the health endpoint must not do this")

    monkeypatch.setattr("app.db.SessionLocal", boom)
    monkeypatch.setattr("app.services.llm_provider._call", boom)
    monkeypatch.setattr("app.services.web_search_service.web_search", boom)

    assert TestClient(main.app).get("/health").status_code == 200


def test_health_is_a_get_only_endpoint():
    from fastapi.testclient import TestClient

    import main

    c = TestClient(main.app)
    assert c.get("/health").status_code == 200
    assert c.post("/health").status_code == 405


def test_root_still_answers_and_is_unchanged(client):
    """`/` was already reachable and is a fine secondary target; /health is the
    stable contract, so the root stays free to change."""
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"message": "Close AI Backend Running"}


def test_a_cross_origin_get_to_health_is_not_rejected_by_the_server():
    """CORS is enforced by the BROWSER, not by us — a scheduler sends no Origin
    at all. This pins that an unknown Origin still gets a 200 body rather than
    the server refusing the request outright."""
    from fastapi.testclient import TestClient

    import main

    r = TestClient(main.app).get("/health", headers={"Origin": "https://cron.example.com"})
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
