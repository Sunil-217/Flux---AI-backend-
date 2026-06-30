"""CRUD coverage for the admin Payment Gateways routes (/admin/payment-gateways).

Auth is bypassed by overriding `require_admin` with a dummy admin, on an
isolated in-memory DB — same approach as test_routes.py."""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


@pytest.fixture
def client():
    import main
    from app.core.security import require_admin
    from app.db import Base, get_db

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

    main.app.dependency_overrides[require_admin] = lambda: SimpleNamespace(
        id=1, name="Admin", email="admin@example.com"
    )
    main.app.dependency_overrides[get_db] = override_get_db
    c = TestClient(main.app)
    yield c
    main.app.dependency_overrides.clear()


def test_list_empty(client):
    resp = client.get("/admin/payment-gateways")
    assert resp.status_code == 200
    assert resp.json() == {"gateways": []}


def test_create_then_list(client):
    resp = client.post("/admin/payment-gateways", json={"name": "Wire Transfer"})
    assert resp.status_code == 200
    g = resp.json()
    assert g["id"] > 0
    assert g["name"] == "Wire Transfer"
    # Defaults applied.
    assert g["deposit_enabled"] is True
    assert g["withdrawal_enabled"] is True
    assert g["active"] is True
    assert g["linked"] is False  # no remote PSP bound yet

    listed = client.get("/admin/payment-gateways").json()["gateways"]
    assert len(listed) == 1
    assert listed[0]["name"] == "Wire Transfer"


def test_create_requires_name(client):
    resp = client.post("/admin/payment-gateways", json={"name": ""})
    assert resp.status_code == 422  # pydantic min_length


def test_patch_toggles_fields(client):
    gid = client.post("/admin/payment-gateways", json={"name": "VirtualPay"}).json()["id"]
    resp = client.patch(
        f"/admin/payment-gateways/{gid}",
        json={"withdrawal_enabled": False, "active": False},
    )
    assert resp.status_code == 200
    g = resp.json()
    assert g["withdrawal_enabled"] is False
    assert g["active"] is False
    assert g["deposit_enabled"] is True  # untouched


def test_patch_missing_404(client):
    resp = client.patch("/admin/payment-gateways/9999", json={"active": False})
    assert resp.status_code == 404


def test_delete(client):
    gid = client.post("/admin/payment-gateways", json={"name": "Temp"}).json()["id"]
    assert client.delete(f"/admin/payment-gateways/{gid}").status_code == 200
    assert client.get("/admin/payment-gateways").json()["gateways"] == []
    # Deleting again is a 404.
    assert client.delete(f"/admin/payment-gateways/{gid}").status_code == 404


def test_list_sorted_by_sort_order(client):
    # Created in non-alphabetical order; sort_order is assigned incrementally,
    # so the list comes back in creation order regardless of name.
    for name in ("Zeta", "Alpha", "Mango"):
        client.post("/admin/payment-gateways", json={"name": name})
    names = [g["name"] for g in client.get("/admin/payment-gateways").json()["gateways"]]
    assert names == ["Zeta", "Alpha", "Mango"]


# ── Fluxway catalog + onboarding (Fluxway client mocked) ──
def test_catalog_unconfigured(client, monkeypatch):
    from app.services import psp_service

    monkeypatch.setattr(psp_service, "is_configured", lambda: False)
    resp = client.get("/admin/payment-gateways/catalog")
    assert resp.status_code == 200
    assert resp.json() == {"configured": False, "flow_type": None, "targets": []}


def test_catalog_configured(client, monkeypatch):
    from app.services import psp_service

    monkeypatch.setattr(psp_service, "is_configured", lambda: True)
    monkeypatch.setattr(
        psp_service,
        "list_catalog",
        lambda: {
            "flow_type": "PSP",
            "targets": [
                {
                    "id": "ft_1",
                    "name": "StonePay",
                    "logo": None,
                    "credential_schema": {"type": "object", "properties": {"apiKey": {"type": "string"}}, "required": ["apiKey"]},
                    "input_schema": None,
                    "operations": [{"flow_action_id": "fa_1", "flow_definition_id": "fd_1"}],
                }
            ],
        },
    )
    body = client.get("/admin/payment-gateways/catalog").json()
    assert body["configured"] is True
    assert body["flow_type"] == "PSP"
    assert body["targets"][0]["name"] == "StonePay"
    assert body["targets"][0]["credential_schema"]["required"] == ["apiKey"]


def test_onboard_unconfigured_400(client, monkeypatch):
    from app.services import psp_service

    monkeypatch.setattr(psp_service, "is_configured", lambda: False)
    resp = client.post(
        "/admin/payment-gateways/onboard",
        json={"flow_target_id": "ft_1", "name": "StonePay", "credential": {"apiKey": "x"}},
    )
    assert resp.status_code == 400


def test_onboard_success_persists_local_row(client, monkeypatch):
    from app.services import psp_service

    monkeypatch.setattr(psp_service, "is_configured", lambda: True)
    monkeypatch.setattr(
        psp_service,
        "get_target",
        lambda fid: {"id": fid, "name": "StonePay", "logo": None,
                     "operations": [{"flow_action_id": "fa_1", "flow_definition_id": "fd_1"}]},
    )
    captured = {}

    def fake_onboard(**kwargs):
        captured.update(kwargs)
        return {"id": "psp_123", "brandId": "br_1", "environmentId": "env_1"}

    monkeypatch.setattr(psp_service, "onboard_psp", fake_onboard)

    resp = client.post(
        "/admin/payment-gateways/onboard",
        json={"flow_target_id": "ft_9", "name": "StonePay", "credential": {"apiKey": "secret"}},
    )
    assert resp.status_code == 200
    g = resp.json()
    assert g["name"] == "StonePay"
    assert g["linked"] is True  # psp_id was set from the remote response
    # Credentials were forwarded to Fluxway, and operations came from the target.
    assert captured["credential"] == {"apiKey": "secret"}
    assert captured["operations"] == [{"flow_action_id": "fa_1", "flow_definition_id": "fd_1"}]
    # The local row is now listed.
    assert [x["name"] for x in client.get("/admin/payment-gateways").json()["gateways"]] == ["StonePay"]


def test_onboard_unknown_target_404(client, monkeypatch):
    from app.services import psp_service

    monkeypatch.setattr(psp_service, "is_configured", lambda: True)
    monkeypatch.setattr(psp_service, "get_target", lambda fid: None)
    resp = client.post(
        "/admin/payment-gateways/onboard",
        json={"flow_target_id": "ghost", "name": "X", "credential": {}},
    )
    assert resp.status_code == 404


def test_onboard_fluxway_error_502(client, monkeypatch):
    from app.services import psp_service

    monkeypatch.setattr(psp_service, "is_configured", lambda: True)

    def boom(fid):
        raise psp_service.FluxwayError("Fluxway down")

    monkeypatch.setattr(psp_service, "get_target", boom)
    resp = client.post(
        "/admin/payment-gateways/onboard",
        json={"flow_target_id": "ft_1", "name": "X", "credential": {}},
    )
    assert resp.status_code == 502
    assert "Fluxway down" in resp.json()["detail"]
