"""Platform admin control panel — /admin/*.

Every endpoint requires a platform-admin token (see `require_admin`). Read
endpoints power the dashboard; the mutating ones (verify / promote / ban /
delete) each write an `AuditLog` row so privileged actions stay traceable.

Designated admins are bootstrapped from ADMIN_EMAILS at startup (main.py); those
accounts are also *protected* here — they can't be demoted, banned, or deleted,
so an admin can never accidentally lock the whole platform out of its own panel.
"""

import json
import os
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.config import ADMIN_EMAILS
from app.core.security import require_admin
from app.db import get_db
from app.models import (
    ApiKey,
    AuditLog,
    OtpCode,
    SharedChat,
    User,
    UserChats,
    UserMemory,
)
from app.services.feature_service import get_effective_features, set_features

router = APIRouter(prefix="/admin", tags=["admin"])

UPLOAD_DIR = "uploads"


# ── Helpers ──────────────────────────────────────────────────────────────────
def _is_protected(email: Optional[str]) -> bool:
    """A bootstrapped superadmin (in ADMIN_EMAILS) is shielded from
    demote/ban/delete so the platform can't be locked out of its own panel."""
    return bool(email) and email.lower() in ADMIN_EMAILS


def _count_chats(blob: Optional[str]) -> int:
    if not blob:
        return 0
    try:
        data = json.loads(blob)
        return len(data) if isinstance(data, list) else 0
    except Exception:
        return 0


def _chat_ids(blob: Optional[str]) -> List[str]:
    if not blob:
        return []
    try:
        data = json.loads(blob)
        return [s["id"] for s in data if isinstance(s, dict) and s.get("id")]
    except Exception:
        return []


def _record(
    db: Session,
    actor: User,
    action: str,
    target: Optional[User] = None,
    detail: Optional[str] = None,
) -> None:
    """Append an audit row. Best-effort — never block the action on log failure."""
    try:
        db.add(
            AuditLog(
                actor_id=actor.id,
                actor_email=actor.email,
                action=action,
                target_id=target.id if target else None,
                target_email=target.email if target else None,
                detail=detail,
            )
        )
    except Exception:
        pass


def _user_row(u: User, chat_count: int = 0, key_count: int = 0) -> dict:
    return {
        "id": u.id,
        "name": u.name,
        "email": u.email,
        "phone": u.phone,
        "is_verified": bool(u.is_verified),
        "is_admin": bool(getattr(u, "is_admin", False)),
        "is_banned": bool(getattr(u, "is_banned", False)),
        "is_protected": _is_protected(u.email),
        "created_at": u.created_at.isoformat() if u.created_at else None,
        "chat_count": chat_count,
        "api_key_count": key_count,
    }


# ── Stats / dashboard ────────────────────────────────────────────────────────
@router.get("/stats")
def stats(admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    now = datetime.utcnow()
    d7 = now - timedelta(days=7)
    d30 = now - timedelta(days=30)

    total = db.query(func.count(User.id)).scalar() or 0
    verified = db.query(func.count(User.id)).filter(User.is_verified == True).scalar() or 0  # noqa: E712
    admins = db.query(func.count(User.id)).filter(User.is_admin == True).scalar() or 0  # noqa: E712
    banned = db.query(func.count(User.id)).filter(User.is_banned == True).scalar() or 0  # noqa: E712
    new_7d = db.query(func.count(User.id)).filter(User.created_at >= d7).scalar() or 0
    new_30d = db.query(func.count(User.id)).filter(User.created_at >= d30).scalar() or 0

    api_keys = db.query(func.count(ApiKey.id)).scalar() or 0
    active_keys = db.query(func.count(ApiKey.id)).filter(ApiKey.revoked == False).scalar() or 0  # noqa: E712
    api_calls = db.query(func.coalesce(func.sum(ApiKey.usage_count), 0)).scalar() or 0
    shared = db.query(func.count(SharedChat.id)).scalar() or 0
    memory_users = db.query(func.count(UserMemory.user_id)).scalar() or 0

    # Total chats across the platform = sum of each user's sessions blob length.
    total_chats = 0
    for (blob,) in db.query(UserChats.data).all():
        total_chats += _count_chats(blob)

    recent = (
        db.query(User).order_by(User.created_at.desc()).limit(6).all()
    )

    return {
        "users": {
            "total": total,
            "verified": verified,
            "unverified": total - verified,
            "admins": admins,
            "banned": banned,
            "new_7d": new_7d,
            "new_30d": new_30d,
        },
        "content": {
            "chats": total_chats,
            "api_keys": api_keys,
            "active_api_keys": active_keys,
            "api_calls": int(api_calls),
            "shared_chats": shared,
            "memory_users": memory_users,
        },
        "recent_signups": [
            {
                "id": u.id,
                "name": u.name,
                "email": u.email,
                "is_verified": bool(u.is_verified),
                "created_at": u.created_at.isoformat() if u.created_at else None,
            }
            for u in recent
        ],
    }


# ── User listing ─────────────────────────────────────────────────────────────
@router.get("/users")
def list_users(
    q: str = Query("", description="search by name or email"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    query = db.query(User)
    term = q.strip().lower()
    if term:
        like = f"%{term}%"
        query = query.filter(
            func.lower(User.name).like(like) | func.lower(User.email).like(like)
        )

    total = query.count()
    rows = (
        query.order_by(User.created_at.desc()).offset(offset).limit(limit).all()
    )
    ids = [u.id for u in rows]

    # Batch the per-user counts so the table doesn't N+1.
    chat_counts = {
        uid: _count_chats(blob)
        for uid, blob in db.query(UserChats.user_id, UserChats.data)
        .filter(UserChats.user_id.in_(ids))
        .all()
    } if ids else {}

    key_counts = dict(
        db.query(ApiKey.user_id, func.count(ApiKey.id))
        .filter(ApiKey.user_id.in_(ids), ApiKey.revoked == False)  # noqa: E712
        .group_by(ApiKey.user_id)
        .all()
    ) if ids else {}

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "users": [
            _user_row(u, chat_counts.get(u.id, 0), key_counts.get(u.id, 0))
            for u in rows
        ],
    }


@router.get("/users/{user_id}")
def get_user(
    user_id: int,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    u = db.get(User, user_id)
    if u is None:
        raise HTTPException(404, "User not found.")
    rec = db.get(UserChats, u.id)
    key_count = (
        db.query(func.count(ApiKey.id))
        .filter(ApiKey.user_id == u.id, ApiKey.revoked == False)  # noqa: E712
        .scalar()
        or 0
    )
    return _user_row(u, _count_chats(rec.data if rec else None), key_count)


# ── User mutations ───────────────────────────────────────────────────────────
class UpdateUserRequest(BaseModel):
    is_verified: Optional[bool] = None
    is_admin: Optional[bool] = None
    is_banned: Optional[bool] = None


@router.patch("/users/{user_id}")
def update_user(
    user_id: int,
    req: UpdateUserRequest,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    u = db.get(User, user_id)
    if u is None:
        raise HTTPException(404, "User not found.")

    changes: List[str] = []

    # ── Verify / unverify ──
    if req.is_verified is not None and bool(u.is_verified) != req.is_verified:
        u.is_verified = req.is_verified
        changes.append("verify" if req.is_verified else "unverify")
        _record(db, admin, "user.verify" if req.is_verified else "user.unverify", u)

    # ── Promote / demote (admin) ──
    if req.is_admin is not None and bool(getattr(u, "is_admin", False)) != req.is_admin:
        if not req.is_admin and (u.id == admin.id or _is_protected(u.email)):
            raise HTTPException(400, "This admin account is protected and can't be demoted.")
        u.is_admin = req.is_admin
        changes.append("promote" if req.is_admin else "demote")
        _record(db, admin, "user.promote" if req.is_admin else "user.demote", u)

    # ── Ban / unban ──
    if req.is_banned is not None and bool(getattr(u, "is_banned", False)) != req.is_banned:
        if req.is_banned and (u.id == admin.id or _is_protected(u.email)):
            raise HTTPException(400, "You can't ban yourself or a protected admin.")
        u.is_banned = req.is_banned
        changes.append("ban" if req.is_banned else "unban")
        _record(db, admin, "user.ban" if req.is_banned else "user.unban", u)

    if changes:
        db.commit()
        db.refresh(u)

    rec = db.get(UserChats, u.id)
    key_count = (
        db.query(func.count(ApiKey.id))
        .filter(ApiKey.user_id == u.id, ApiKey.revoked == False)  # noqa: E712
        .scalar()
        or 0
    )
    return _user_row(u, _count_chats(rec.data if rec else None), key_count)


@router.delete("/users/{user_id}")
def delete_user(
    user_id: int,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    u = db.get(User, user_id)
    if u is None:
        raise HTTPException(404, "User not found.")
    if u.id == admin.id:
        raise HTTPException(400, "You can't delete your own account here.")
    if _is_protected(u.email):
        raise HTTPException(400, "This admin account is protected and can't be deleted.")

    target_email = u.email

    # 1. Best-effort cleanup of each chat's vectors / uploads / cached web context.
    rec = db.get(UserChats, u.id)
    for cid in _chat_ids(rec.data if rec else None):
        try:
            from app.services.chroma_service import delete_collection, sanitize_chat_id

            delete_collection(cid)
            folder = os.path.join(UPLOAD_DIR, sanitize_chat_id(cid))
            if os.path.exists(folder):
                shutil.rmtree(folder, ignore_errors=True)
        except Exception:
            pass
        try:
            from app.services.rag_service import delete_web_context

            delete_web_context(cid)
        except Exception:
            pass

    # 2. Remove the user's rows across all tables, then the user itself.
    db.query(UserChats).filter(UserChats.user_id == u.id).delete()
    db.query(UserMemory).filter(UserMemory.user_id == u.id).delete()
    db.query(ApiKey).filter(ApiKey.user_id == u.id).delete()
    db.query(SharedChat).filter(SharedChat.owner_id == u.id).delete()
    db.query(OtpCode).filter(OtpCode.email == u.email).delete()
    _record(db, admin, "user.delete", u, detail=f"deleted account {target_email}")
    db.delete(u)
    db.commit()

    return {"ok": True, "deleted": target_email}


# ── API key management (admin can audit / revoke / delete any user's keys) ─────
def _key_row(k: ApiKey) -> dict:
    return {
        "id": k.id,
        "name": k.name,
        "prefix": k.prefix,
        "revoked": bool(k.revoked),
        "usage_count": k.usage_count or 0,
        "total_tokens": k.total_tokens or 0,
        "created_at": k.created_at.isoformat() if k.created_at else None,
        "last_used_at": k.last_used_at.isoformat() if k.last_used_at else None,
    }


@router.get("/users/{user_id}/api-keys")
def user_api_keys(
    user_id: int,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    if db.get(User, user_id) is None:
        raise HTTPException(404, "User not found.")
    rows = (
        db.query(ApiKey)
        .filter(ApiKey.user_id == user_id)
        .order_by(ApiKey.created_at.desc())
        .all()
    )
    return {"keys": [_key_row(k) for k in rows]}


@router.post("/api-keys/{key_id}/revoke")
def admin_revoke_key(
    key_id: int,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Disable a key (apps using it stop working) but keep the row for audit."""
    rec = db.get(ApiKey, key_id)
    if rec is None:
        raise HTTPException(404, "Key not found.")
    rec.revoked = True
    owner = db.get(User, rec.user_id)
    _record(db, admin, "apikey.revoke", owner, detail=f"key {rec.prefix}")
    db.commit()
    return {"ok": True}


@router.delete("/api-keys/{key_id}")
def admin_delete_key(
    key_id: int,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Permanently remove a key."""
    rec = db.get(ApiKey, key_id)
    if rec is None:
        raise HTTPException(404, "Key not found.")
    owner = db.get(User, rec.user_id)
    prefix = rec.prefix
    db.delete(rec)
    _record(db, admin, "apikey.delete", owner, detail=f"key {prefix}")
    db.commit()
    return {"ok": True}


# ── Feature flags ────────────────────────────────────────────────────────────
class FeaturesPatch(BaseModel):
    features: Dict[str, bool]


@router.get("/features")
def get_features(admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    return {"features": get_effective_features(db)}


@router.patch("/features")
def patch_features(
    req: FeaturesPatch,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    effective = set_features(db, req.features)
    # Record which flags were touched (and their new values).
    _record(db, admin, "features.update", detail=json.dumps(req.features))
    db.commit()
    return {"features": effective}


# ── Audit log ────────────────────────────────────────────────────────────────
@router.get("/audit")
def audit_log(
    limit: int = Query(50, ge=1, le=200),
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    rows = (
        db.query(AuditLog).order_by(AuditLog.created_at.desc()).limit(limit).all()
    )
    return {
        "entries": [
            {
                "id": r.id,
                "actor_email": r.actor_email,
                "action": r.action,
                "target_email": r.target_email,
                "detail": r.detail,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ]
    }
