from fastapi.testclient import TestClient


def _client():
    import main
    return TestClient(main.app)


# ── /chat (streaming) ──
def test_chat_streams_events(monkeypatch):
    import app.api.routes.chat as chat_route
    from app.services import rag_service

    def fake_stream(chat_id, question, history):
        yield rag_service._sse({"type": "token", "content": "Hi"})
        yield rag_service._sse({"type": "token", "content": " there"})
        yield rag_service._sse({"type": "done"})

    monkeypatch.setattr(chat_route, "stream_question", fake_stream)

    resp = _client().post("/chat", json={"chat_id": "c1", "question": "hi", "history": []})
    assert resp.status_code == 200
    assert "Hi" in resp.text
    assert "there" in resp.text
    assert "done" in resp.text


def test_chat_forwards_history(monkeypatch):
    import app.api.routes.chat as chat_route
    from app.services import rag_service

    captured = {}

    def fake_stream(chat_id, question, history):
        captured["history"] = history
        captured["question"] = question
        yield rag_service._sse({"type": "done"})

    monkeypatch.setattr(chat_route, "stream_question", fake_stream)

    _client().post(
        "/chat",
        json={
            "chat_id": "c1",
            "question": "next?",
            "history": [{"role": "user", "content": "earlier"}],
        },
    )
    assert captured["question"] == "next?"
    assert captured["history"] == [{"role": "user", "content": "earlier"}]


def test_chat_requires_fields():
    resp = _client().post("/chat", json={"question": "hi"})  # missing chat_id
    assert resp.status_code == 422


# ── /delete ──
def test_delete_success(monkeypatch):
    import app.api.routes.delete as delete_route

    called = {}
    monkeypatch.setattr(delete_route, "delete_collection", lambda cid: called.setdefault("id", cid))

    resp = _client().delete("/delete/session-xyz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["message"] == "Chat deleted successfully"
    assert body["chat_id"] == "session-xyz"
    assert called["id"] == "session-xyz"


# ── /upload ──
def test_upload_rejects_non_pdf():
    resp = _client().post(
        "/upload",
        files={"file": ("notes.txt", b"hello", "text/plain")},
        data={"chat_id": "c1"},
    )
    assert resp.status_code == 200
    assert resp.json().get("error") == "Only PDF files are allowed"


def test_upload_accepts_pdf(monkeypatch, tmp_path):
    import app.api.routes.upload as upload_route

    monkeypatch.setattr(upload_route, "UPLOAD_DIR", str(tmp_path))

    resp = _client().post(
        "/upload",
        files={"file": ("doc.pdf", b"%PDF-1.4 fake content", "application/pdf")},
        data={"chat_id": "c1"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("message") == "File uploaded successfully"
    assert body.get("filename") == "doc.pdf"
    assert body.get("total_chunks", 0) >= 1


# ── root ──
def test_home_route():
    resp = _client().get("/")
    assert resp.status_code == 200
    assert "message" in resp.json()
