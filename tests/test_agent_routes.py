"""POST /agent/task and GET /agent/providers — auth, delegation, secrecy.

Also pins the promise that adding this endpoint changed nothing about /chat.
"""

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


@pytest.fixture
def app_and_db():
    import main
    from app.db import Base, get_db
    from app.models import UserChats

    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    TestingSession = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(bind=engine)

    seed = TestingSession()
    seed.add(UserChats(user_id=1, data=json.dumps([{"id": "c1"}])))
    seed.commit()
    seed.close()

    def override_get_db():
        db = TestingSession()
        try:
            yield db
        finally:
            db.close()

    main.app.dependency_overrides[get_db] = override_get_db
    yield main
    main.app.dependency_overrides.clear()


@pytest.fixture
def client(app_and_db):
    from app.core.security import get_current_user

    app_and_db.app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(
        id=1, name="Test", email="test@example.com"
    )
    return TestClient(app_and_db.app)


def _events(body: str):
    return [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ")]


def _post(client, **kw):
    payload = {"chat_id": "c1", "question": "hi", **kw}
    return client.post("/agent/task", json=payload)


# ── Authentication and ownership ─────────────────────────────────────────────

def test_unauthenticated_requests_are_rejected(app_and_db):
    """The agent path reads the user's documents and memory. It must not be
    reachable without the same authentication chat requires."""
    r = TestClient(app_and_db.app).post("/agent/task", json={"chat_id": "c1", "question": "hi"})
    assert r.status_code in (401, 403)


def test_a_chat_the_user_does_not_own_is_404(client):
    """Chat ids are generated client-side, so ownership is checked server-side —
    otherwise guessing an id would read someone else's documents."""
    assert _post(client, chat_id="someone-elses-chat").status_code == 404


def test_an_empty_question_is_rejected_by_validation(client):
    assert _post(client, question="").status_code == 422


# ── Delegation to ordinary chat ──────────────────────────────────────────────

def test_simple_messages_are_served_by_the_normal_chat_stream(client, monkeypatch):
    """Sending every message here must be safe: a greeting gets the ordinary
    chat answer, not a planning round-trip."""
    import app.api.routes.agent as route
    from app.services import rag_service

    called = []

    def fake_stream(chat_id, question, history, image=None, *a):
        called.append(question)
        yield rag_service._sse({"type": "token", "content": "hello"})

    monkeypatch.setattr("app.services.rag_service.stream_question", fake_stream)
    monkeypatch.setattr(route, "AGENT_ORCHESTRATION_ENABLED", True)

    events = _events(_post(client, question="hi da").text)
    assert called == ["hi da"]
    assert [e["type"] for e in events] == ["token", "done"]


def test_orchestration_can_be_switched_off_without_a_redeploy(client, monkeypatch):
    import app.api.routes.agent as route
    from app.services import rag_service

    monkeypatch.setattr(route, "AGENT_ORCHESTRATION_ENABLED", False)
    monkeypatch.setattr(
        "app.services.rag_service.stream_question",
        lambda *a, **k: iter([rag_service._sse({"type": "token", "content": "plain"})]),
    )
    monkeypatch.setattr(
        "app.agents.orchestrator.run_task",
        lambda *a, **k: pytest.fail("orchestration must not run when disabled"),
    )

    events = _events(_post(client, question="Research the market, compare vendors, write a report").text)
    assert [e["type"] for e in events] == ["token", "done"]


def test_a_complex_goal_reaches_the_orchestrator_and_streams_status(client, monkeypatch):
    import app.api.routes.agent as route

    monkeypatch.setattr(route, "AGENT_ORCHESTRATION_ENABLED", True)
    monkeypatch.setattr("app.agents.orchestrator.run_task", lambda *a, **k: iter([
        {"type": "status", "stage": "planning", "label": "Planning"},
        {"type": "token", "content": "the report"},
        {"type": "done"},
    ]))

    events = _events(_post(client, question="Research 5 competitors of Notion, compare their pricing and write a report").text)
    assert [e["type"] for e in events] == ["status", "token", "done"]
    assert events[0]["label"] == "Planning"


def test_a_delegate_event_falls_through_to_chat(client, monkeypatch):
    """`delegate` is an instruction to the route, never something the client
    should have to interpret."""
    import app.api.routes.agent as route
    from app.services import rag_service

    monkeypatch.setattr(route, "AGENT_ORCHESTRATION_ENABLED", True)
    monkeypatch.setattr("app.agents.orchestrator.run_task",
                        lambda *a, **k: iter([{"type": "delegate", "reason": "simple"}]))
    monkeypatch.setattr(
        "app.services.rag_service.stream_question",
        lambda *a, **k: iter([rag_service._sse({"type": "token", "content": "normal answer"})]),
    )

    events = _events(_post(client, question="Research the market, compare vendors, write a report").text)
    assert [e["type"] for e in events] == ["token", "done"]
    assert "delegate" not in [e["type"] for e in events]


# ── Diagnostics never expose secrets ─────────────────────────────────────────

def test_provider_diagnostics_require_auth(app_and_db):
    assert TestClient(app_and_db.app).get("/agent/providers").status_code in (401, 403)


def test_provider_diagnostics_report_health_without_credentials(client):
    r = client.get("/agent/providers")
    assert r.status_code == 200
    body = r.text
    assert "groq" in r.json()["providers"] and "nvidia" in r.json()["providers"]
    assert "test-groq-key" not in body and "test-nvidia-key" not in body
    assert "api_key" not in body.lower()


# ── Backward compatibility ───────────────────────────────────────────────────

def test_chat_endpoint_is_unchanged(client, monkeypatch):
    """/chat keeps its exact contract — this feature is additive."""
    from app.services import rag_service

    monkeypatch.setattr(
        "app.api.routes.chat.stream_question",
        lambda *a, **k: iter([
            rag_service._sse({"type": "token", "content": "Hi"}),
            rag_service._sse({"type": "done"}),
        ]),
    )
    r = client.post("/chat", json={"chat_id": "c1", "question": "hello"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert [e["type"] for e in _events(r.text)] == ["token", "done"]


def test_agent_task_accepts_the_same_body_as_chat(client, monkeypatch):
    """A client can post an identical payload to either endpoint."""
    import app.api.routes.agent as route
    from app.services import rag_service

    monkeypatch.setattr(route, "AGENT_ORCHESTRATION_ENABLED", True)
    monkeypatch.setattr(
        "app.services.rag_service.stream_question",
        lambda *a, **k: iter([rag_service._sse({"type": "done"})]),
    )
    body = {
        "chat_id": "c1", "question": "hi", "history": [{"role": "user", "content": "earlier"}],
        "style": "concise", "custom_instructions": "be brief",
        "web_search": False, "active_docs": ["a.pdf"],
    }
    assert client.post("/agent/task", json=body).status_code == 200
    assert client.post("/chat", json=body).status_code == 200


# ── Task ownership (GET /agent/tasks) ────────────────────────────────────────

@pytest.fixture
def seeded_tasks(app_and_db, monkeypatch):
    """Two tasks: one owned by user 1 (the authenticated caller), one by user 2."""
    from app.agents import store
    from app.agents.state import SubTask, TaskState, TaskStatus
    from app.db import get_db

    session_factory = app_and_db.app.dependency_overrides[get_db]

    def _factory():
        gen = session_factory()
        return next(gen)

    monkeypatch.setattr(store, "SessionLocal", _factory)

    made = {}
    for uid, key in ((1, "mine"), (2, "theirs")):
        st = TaskState(user_goal=f"goal for {uid}", chat_id="c1", user_id=uid)
        st.plan = [SubTask(id=1, agent="research", task="t")]
        st.agent_outputs = {1: "output"}
        st.status = TaskStatus.COMPLETED
        store.save(st, uid)
        made[key] = st.task_id
    return made


def test_a_user_sees_only_their_own_tasks(client, seeded_tasks):
    body = client.get("/agent/tasks").json()
    assert [t["task_id"] for t in body["tasks"]] == [seeded_tasks["mine"]]


def test_another_users_task_is_404_not_403(client, seeded_tasks):
    """403 would confirm the id exists — information the caller should not get
    from an object they do not own."""
    assert client.get(f"/agent/tasks/{seeded_tasks['mine']}").status_code == 200
    assert client.get(f"/agent/tasks/{seeded_tasks['theirs']}").status_code == 404
    assert client.get("/agent/tasks/does-not-exist").status_code == 404


def test_a_task_read_returns_the_full_record(client, seeded_tasks):
    body = client.get(f"/agent/tasks/{seeded_tasks['mine']}").json()
    assert body["status"] == "COMPLETED"
    assert body["plan"][0]["agent"] == "research"
    assert body["agent_outputs"]["1"] == "output"


def test_task_endpoints_require_authentication(app_and_db):
    c = TestClient(app_and_db.app)
    assert c.get("/agent/tasks").status_code in (401, 403)
    assert c.get("/agent/tasks/anything").status_code in (401, 403)
