"""
Smoke test for the webhook event-filter feature:
  create → list → PATCH events → verify persisted → reject empty events.
"""
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


@pytest.fixture
def admin_client():
    """TestClient on an isolated in-memory DB, with require_admin overridden."""
    import main
    from app.db import Base, get_db
    from app.core.security import require_admin

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
    main.app.dependency_overrides[require_admin] = lambda: SimpleNamespace(
        id=1, name="Admin", email="admin@example.com", is_admin=True
    )
    c = TestClient(main.app)
    yield c
    main.app.dependency_overrides.clear()


def test_webhook_update_events_flow(admin_client):
    from app.services.webhook_service import WEBHOOK_EVENTS

    assert len(WEBHOOK_EVENTS) >= 2  # need at least two to flip the subscription

    # ── Create with a single event ──
    create = admin_client.post(
        "/admin/webhooks",
        json={"url": "https://example.com/hook", "events": [WEBHOOK_EVENTS[0]]},
    )
    assert create.status_code == 200, create.text
    body = create.json()
    hook_id = body["id"]
    assert body["events"] == [WEBHOOK_EVENTS[0]]
    assert body["secret"].startswith("whsec_")  # secret shown once

    # ── Listing returns it + the full event catalogue ──
    listing = admin_client.get("/admin/webhooks")
    assert listing.status_code == 200
    lbody = listing.json()
    assert any(h["id"] == hook_id for h in lbody["webhooks"])
    assert lbody["events"] == WEBHOOK_EVENTS

    # ── PATCH to a different event set (the new Edit-events UI path) ──
    new_events = [WEBHOOK_EVENTS[0], WEBHOOK_EVENTS[1]]
    patch = admin_client.patch(f"/admin/webhooks/{hook_id}", json={"events": new_events})
    assert patch.status_code == 200, patch.text
    # _clean_events sorts by catalogue order, so compare against that ordering.
    assert patch.json()["events"] == [e for e in WEBHOOK_EVENTS if e in set(new_events)]

    # ── Verify persistence via a fresh list ──
    relist = admin_client.get("/admin/webhooks").json()["webhooks"]
    persisted = next(h for h in relist if h["id"] == hook_id)
    assert set(persisted["events"]) == set(new_events)

    # ── Empty event list must be rejected (a hook needs ≥1 event) ──
    bad = admin_client.patch(f"/admin/webhooks/{hook_id}", json={"events": []})
    assert bad.status_code == 400

    # ── Unknown events are silently dropped; if all unknown → 400 ──
    junk = admin_client.patch(f"/admin/webhooks/{hook_id}", json={"events": ["nope.bogus"]})
    assert junk.status_code == 400

    # ── Cleanup ──
    assert admin_client.delete(f"/admin/webhooks/{hook_id}").status_code == 200


def test_webhook_update_requires_admin():
    """Without the admin override, the route must reject (401/403)."""
    import main

    c = TestClient(main.app)
    resp = c.patch("/admin/webhooks/1", json={"events": []})
    assert resp.status_code in (401, 403)
