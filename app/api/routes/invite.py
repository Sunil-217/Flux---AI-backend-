"""Invite-based onboarding (public).

An admin issues an invite for a specific email (see /admin/invites); the
recipient opens the link, sets a name + password, and is signed in immediately —
no OTP, because the admin already vouched for them.
"""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.rate_limit import limiter
from app.core.security import create_access_token
from app.db import get_db
from app.models import Invite
from app.services import auth_service
from app.services.webhook_service import dispatch_event

router = APIRouter(tags=["invite"])


@router.get("/invite/{token}")
def check_invite(token: str, db: Session = Depends(get_db)):
    """Validate an invite link without consuming it (drives the accept screen)."""
    inv = db.query(Invite).filter(Invite.token == token).first()
    if inv is None:
        raise HTTPException(404, "This invite link is invalid.")
    if inv.accepted:
        raise HTTPException(400, "This invite has already been used. Please sign in.")
    if inv.expires_at < datetime.utcnow():
        raise HTTPException(400, "This invite has expired. Ask your admin for a new one.")
    return {"email": inv.email, "valid": True}


class AcceptInviteRequest(BaseModel):
    token: str
    name: str
    password: str
    phone: str | None = None


@router.post("/invite/accept")
@limiter.limit("5/minute")
def accept_invite(request: Request, req: AcceptInviteRequest, db: Session = Depends(get_db)):
    if len(req.password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters.")
    if not req.name.strip():
        raise HTTPException(400, "Name is required.")
    try:
        user = auth_service.accept_invite(
            db, req.token.strip(), req.name.strip(), req.password, (req.phone or "").strip()
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    dispatch_event(
        "user.signup",
        {
            "id": user.id,
            "email": user.email,
            "name": user.name,
            "is_admin": bool(getattr(user, "is_admin", False)),
            "via": "invite",
        },
    )
    return {
        "access_token": create_access_token(user.id),
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "name": user.name,
            "email": user.email,
            "phone": user.phone,
            "is_admin": bool(getattr(user, "is_admin", False)),
        },
    }
