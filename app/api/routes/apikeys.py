"""Developer API keys — create / list / revoke.

The raw key is returned exactly once (at creation). Only its SHA-256 hash is
stored, so a DB leak never exposes usable keys.
"""

import hashlib
import secrets
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.security import get_current_user
from app.db import get_db
from app.models import ApiKey, User
from app.services.webhook_service import dispatch_event

router = APIRouter()

MAX_KEYS_PER_USER = 10


def hash_key(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _display_prefix(raw: str) -> str:
    return f"{raw[:7]}…{raw[-4:]}"


class CreateKeyRequest(BaseModel):
    name: str = Field(min_length=1, max_length=60)


class KeyInfo(BaseModel):
    id: int
    name: str
    prefix: str
    revoked: bool
    usage_count: int
    total_tokens: int
    created_at: Optional[str] = None
    last_used_at: Optional[str] = None


def _info(k: ApiKey) -> KeyInfo:
    return KeyInfo(
        id=k.id,
        name=k.name,
        prefix=k.prefix,
        revoked=k.revoked,
        usage_count=k.usage_count or 0,
        total_tokens=k.total_tokens or 0,
        created_at=k.created_at.isoformat() if k.created_at else None,
        last_used_at=k.last_used_at.isoformat() if k.last_used_at else None,
    )


@router.post("/api-keys")
def create_key(
    req: CreateKeyRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if getattr(user, "api_blocked", False):
        raise HTTPException(403, "API access for your account has been disabled by an administrator.")
    active = (
        db.query(ApiKey)
        .filter(ApiKey.user_id == user.id, ApiKey.revoked == False)  # noqa: E712
        .count()
    )
    if active >= MAX_KEYS_PER_USER:
        raise HTTPException(400, f"Limit reached ({MAX_KEYS_PER_USER} active keys). Revoke one first.")

    raw = "ck_" + secrets.token_urlsafe(32)
    rec = ApiKey(
        user_id=user.id,
        name=req.name.strip(),
        key_hash=hash_key(raw),
        prefix=_display_prefix(raw),
        created_at=datetime.utcnow(),
    )
    db.add(rec)
    db.commit()
    db.refresh(rec)
    # Notify any subscribed platform webhooks (fire-and-forget, never blocks).
    dispatch_event(
        "apikey.created",
        {"user_id": user.id, "email": user.email, "key_name": rec.name, "prefix": rec.prefix},
    )
    # `key` is shown ONCE — the client must copy it now.
    return {"key": raw, "info": _info(rec).model_dump()}


@router.get("/api-keys")
def list_keys(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    rows = (
        db.query(ApiKey)
        .filter(ApiKey.user_id == user.id)
        .order_by(ApiKey.created_at.desc())
        .all()
    )
    return {"keys": [_info(k).model_dump() for k in rows]}


@router.delete("/api-keys/{key_id}")
def revoke_key(
    key_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    rec = db.get(ApiKey, key_id)
    if rec is None or rec.user_id != user.id:
        raise HTTPException(404, "Key not found")
    rec.revoked = True
    db.commit()
    return {"ok": True}
