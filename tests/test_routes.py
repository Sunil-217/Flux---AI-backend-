from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Authenticated TestClient — bypasses get_current_user with a dummy user."""
    import main
    from app.core.security import get_current_user

    main.app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(
        id=1, name="Test", email="test@example.com"
    )
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
def test_upload_rejects_non_pdf(client):
    resp = client.post(
        "/upload",
        files={"file": ("notes.txt", b"hello", "text/plain")},
        data={"chat_id": "c1"},
    )
    assert resp.status_code == 200
    assert resp.json().get("error") == "Only PDF files are allowed"


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
