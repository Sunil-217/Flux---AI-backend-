"""Payment Gateways admin — /admin/payment-gateways/*.

Backs the "Payment Gateways" menu in the admin control panel. Phase 1 persists
the gateway config locally (CRUD). Linking each gateway to a remote orchestrated
PSP on the Fluxway backend is Phase 2 — the Fluxway-binding columns
(`psp_id`, `brand_id`, …) already exist on the model and stay null until then.

Every endpoint requires a platform-admin token; mutating calls write an
`AuditLog` row via the shared `_record` helper, exactly like /admin/plans."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.api.routes.admin import _record
from app.core.security import require_admin
from app.db import get_db
from app.models import PaymentGateway, User
from app.services import psp_service

router = APIRouter(prefix="/admin/payment-gateways", tags=["payment-gateways"])


def _out(g: PaymentGateway) -> dict:
    return {
        "id": g.id,
        "name": g.name,
        "description": g.description,
        "logo": g.logo,
        "deposit_enabled": g.deposit_enabled,
        "withdrawal_enabled": g.withdrawal_enabled,
        "active": g.active,
        "sort_order": g.sort_order,
        "provider_code": g.provider_code,
        # Whether this gateway is linked to a remote orchestrated PSP yet.
        "linked": bool(g.psp_id),
        "created_at": g.created_at.isoformat() if g.created_at else None,
        "updated_at": g.updated_at.isoformat() if g.updated_at else None,
    }


class GatewayIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    description: Optional[str] = Field(default=None, max_length=200)
    logo: Optional[str] = None
    deposit_enabled: bool = True
    withdrawal_enabled: bool = True
    active: bool = True
    provider_code: Optional[str] = Field(default=None, max_length=40)


class GatewayPatch(BaseModel):
    name: Optional[str] = Field(default=None, max_length=80)
    description: Optional[str] = Field(default=None, max_length=200)
    logo: Optional[str] = None
    deposit_enabled: Optional[bool] = None
    withdrawal_enabled: Optional[bool] = None
    active: Optional[bool] = None
    provider_code: Optional[str] = Field(default=None, max_length=40)
    sort_order: Optional[int] = None


@router.get("")
def list_gateways(admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    rows = (
        db.query(PaymentGateway)
        .order_by(PaymentGateway.sort_order.asc(), PaymentGateway.name.asc())
        .all()
    )
    return {"gateways": [_out(g) for g in rows]}


@router.post("")
def create_gateway(req: GatewayIn, admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    name = req.name.strip()
    if not name:
        raise HTTPException(400, "Payment method name is required.")
    max_order = db.query(func.coalesce(func.max(PaymentGateway.sort_order), -1)).scalar()
    g = PaymentGateway(
        name=name,
        description=(req.description or "").strip() or None,
        logo=(req.logo or "").strip() or None,
        deposit_enabled=req.deposit_enabled,
        withdrawal_enabled=req.withdrawal_enabled,
        active=req.active,
        provider_code=(req.provider_code or "").strip() or None,
        sort_order=int(max_order) + 1,
        created_by=admin.email,
    )
    db.add(g)
    _record(db, admin, "gateway.create", detail=f"gateway {name}")
    db.commit()
    db.refresh(g)
    return _out(g)


@router.patch("/{gateway_id}")
def update_gateway(
    gateway_id: int,
    req: GatewayPatch,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    g = db.query(PaymentGateway).filter(PaymentGateway.id == gateway_id).first()
    if g is None:
        raise HTTPException(404, "Payment gateway not found.")

    changes: List[str] = []
    if req.name is not None and req.name.strip():
        g.name = req.name.strip(); changes.append("name")
    if req.description is not None:
        g.description = req.description.strip() or None; changes.append("description")
    if req.logo is not None:
        g.logo = req.logo.strip() or None; changes.append("logo")
    if req.deposit_enabled is not None:
        g.deposit_enabled = req.deposit_enabled; changes.append("deposit")
    if req.withdrawal_enabled is not None:
        g.withdrawal_enabled = req.withdrawal_enabled; changes.append("withdrawal")
    if req.active is not None:
        g.active = req.active; changes.append("active")
    if req.provider_code is not None:
        g.provider_code = req.provider_code.strip() or None; changes.append("provider_code")
    if req.sort_order is not None:
        g.sort_order = req.sort_order; changes.append("sort_order")

    if changes:
        g.updated_at = datetime.utcnow()
        _record(db, admin, "gateway.update", detail=f"gateway {g.name}: {', '.join(changes)}")
        db.commit()
        db.refresh(g)
    return _out(g)


@router.delete("/{gateway_id}")
def delete_gateway(gateway_id: int, admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    g = db.query(PaymentGateway).filter(PaymentGateway.id == gateway_id).first()
    if g is None:
        raise HTTPException(404, "Payment gateway not found.")
    name = g.name
    db.delete(g)
    _record(db, admin, "gateway.delete", detail=f"gateway {name}")
    db.commit()
    return {"ok": True}


# ── Fluxway PSP orchestration ─────────────────────────────────────────────────
# The catalog drives the "Add" dropdown; onboarding enables a PSP at brand level
# on BOTH Fluxway (encrypted credentials) and Close AI (local row).
@router.get("/catalog")
def gateway_catalog(admin: User = Depends(require_admin)):
    """Live PSP catalog from Fluxway (targets + credential JSON Schema). When
    Fluxway isn't configured, `configured=false` so the UI hides the picker and
    offers manual add instead."""
    if not psp_service.is_configured():
        return {"configured": False, "flow_type": None, "targets": []}
    try:
        catalog = psp_service.list_catalog()
    except psp_service.FluxwayError as exc:
        raise HTTPException(502, str(exc))
    return {"configured": True, **catalog}


class OnboardIn(BaseModel):
    flow_target_id: str = Field(min_length=1)
    name: str = Field(min_length=1, max_length=80)
    credential: Dict[str, Any] = Field(default_factory=dict)  # schema-driven fields
    description: Optional[str] = Field(default=None, max_length=200)
    logo: Optional[str] = None
    provider_code: Optional[str] = Field(default=None, max_length=40)


@router.post("/onboard")
def onboard_gateway(req: OnboardIn, admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    """Onboard + enable a PSP on Fluxway, then persist the local gateway row.
    Credentials are forwarded to Fluxway (encrypted there) and never stored here."""
    if not psp_service.is_configured():
        raise HTTPException(400, "Fluxway is not configured. Set FLUXWAY_BASE_URL and FLUXWAY_SECRET_TOKEN.")

    target = None
    try:
        target = psp_service.get_target(req.flow_target_id)
        if target is None:
            raise HTTPException(404, "That PSP is not in the Fluxway catalog.")
        created = psp_service.onboard_psp(
            name=req.name.strip(),
            flow_target_id=req.flow_target_id,
            credential=req.credential,
            operations=target["operations"],
            description=(req.description or "").strip() or None,
            logo=(req.logo or "").strip() or None,
        )
    except psp_service.FluxwayError as exc:
        raise HTTPException(502, str(exc))

    # Persist the local mirror. `created` is Fluxway's PSP details envelope-data.
    max_order = db.query(func.coalesce(func.max(PaymentGateway.sort_order), -1)).scalar()
    g = PaymentGateway(
        name=req.name.strip(),
        description=(req.description or "").strip() or None,
        logo=(req.logo or "").strip() or (target.get("logo") if target else None) or None,
        deposit_enabled=True,
        withdrawal_enabled=True,
        active=True,
        sort_order=int(max_order) + 1,
        provider_code=(req.provider_code or "").strip() or None,
        psp_id=(created or {}).get("id"),
        brand_id=(created or {}).get("brandId"),
        environment_id=(created or {}).get("environmentId"),
        flow_target_id=req.flow_target_id,
        last_synced_at=datetime.utcnow(),
        created_by=admin.email,
    )
    db.add(g)
    _record(db, admin, "gateway.onboard", detail=f"gateway {g.name} (psp {g.psp_id})")
    db.commit()
    db.refresh(g)
    return _out(g)
