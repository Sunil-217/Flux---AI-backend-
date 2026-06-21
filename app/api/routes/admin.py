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
import re
import secrets
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.config import (
    ADMIN_EMAILS,
    FRONTEND_URL,
    IS_PRODUCTION,
    NVIDIA_API_KEY,
    GROQ_API_KEY,
    TAVILY_API_KEY,
    SMTP_HOST,
    SMTP_USER,
    SMTP_PASS,
)
from app.core.security import require_admin
from app.db import engine, get_db
from app.models import (
    ApiKey,
    AuditLog,
    Broadcast,
    Invite,
    KnowledgeDocument,
    OtpCode,
    SharedChat,
    User,
    UserChats,
    UserMemory,
    Webhook,
)
from app.api.routes.kb import _collection_name as _kb_collection, _read_widget_config
from app.models import Plan
from app.services import plan_service
from app.services.chroma_service import delete_kb_collection, delete_kb_document_chunks
from app.services.email_service import send_announcement_bulk, send_announcement_email, send_invite_email
from app.services.feature_service import get_effective_features, set_features
from app.services.webhook_service import WEBHOOK_EVENTS, deliver_test, dispatch_event

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
        "api_blocked": bool(getattr(u, "api_blocked", False)),
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

    # Total chats + per-user counts (one pass over the sessions blobs).
    total_chats = 0
    chat_by_user: dict = {}
    for uid, blob in db.query(UserChats.user_id, UserChats.data).all():
        cnt = _count_chats(blob)
        total_chats += cnt
        if cnt:
            chat_by_user[uid] = cnt

    # Top 5 users by chat volume (most active).
    top_ids = sorted(chat_by_user, key=lambda k: chat_by_user[k], reverse=True)[:5]
    top_rows = (
        {u.id: u for u in db.query(User).filter(User.id.in_(top_ids)).all()}
        if top_ids
        else {}
    )
    top_users = [
        {
            "id": uid,
            "name": top_rows[uid].name,
            "email": top_rows[uid].email,
            "chat_count": chat_by_user[uid],
        }
        for uid in top_ids
        if uid in top_rows
    ]

    # Signups per day for the last 14 days (drives the dashboard chart).
    start14 = datetime(now.year, now.month, now.day) - timedelta(days=13)
    by_day: dict = {}
    for (ca,) in db.query(User.created_at).filter(User.created_at >= start14).all():
        if ca:
            key = ca.date().isoformat()
            by_day[key] = by_day.get(key, 0) + 1
    signups_by_day = [
        {"date": (start14 + timedelta(days=i)).date().isoformat(), "count": by_day.get((start14 + timedelta(days=i)).date().isoformat(), 0)}
        for i in range(14)
    ]

    # System health — DB engine, environment, and which AI providers are wired up
    # (booleans only — never expose key values).
    drivername = engine.url.drivername
    database = (
        "PostgreSQL" if drivername.startswith("postgres")
        else "SQLite" if drivername.startswith("sqlite")
        else drivername
    )
    system = {
        "database": database,
        "environment": "production" if IS_PRODUCTION else "development",
        "providers": {
            "nvidia": bool(NVIDIA_API_KEY),
            "groq": bool(GROQ_API_KEY),
            "tavily": bool(TAVILY_API_KEY),
            "email": bool(SMTP_HOST and SMTP_USER and SMTP_PASS),
        },
    }

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
        "signups_by_day": signups_by_day,
        "top_users": top_users,
        "system": system,
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


# ── Per-user activity log (timeline derived from existing data) ───────────────
@router.get("/users/{user_id}/activity")
def user_activity(
    user_id: int,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """A per-user timeline built from data we already keep: account creation,
    every admin action taken ON this user (audit log), and API-key create/use.
    Plus a footprint summary (chats, keys, memory, shared chats). No new
    tracking — works for every existing user immediately."""
    u = db.get(User, user_id)
    if u is None:
        raise HTTPException(404, "User not found.")

    events: List[dict] = []

    if u.created_at:
        events.append({
            "type": "account.created",
            "label": "Account created",
            "detail": None,
            "actor": None,
            "at": u.created_at.isoformat(),
        })

    # Admin actions taken on this user (this is the user's half of the audit log).
    for a in (
        db.query(AuditLog)
        .filter(AuditLog.target_id == u.id)
        .order_by(AuditLog.created_at.desc())
        .limit(100)
        .all()
    ):
        events.append({
            "type": a.action,
            "label": None,  # the frontend maps the action → a friendly label
            "detail": a.detail,
            "actor": a.actor_email,
            "at": a.created_at.isoformat() if a.created_at else None,
        })

    # Developer-API-key lifecycle.
    for k in db.query(ApiKey).filter(ApiKey.user_id == u.id).all():
        if k.created_at:
            events.append({
                "type": "apikey.created",
                "label": "Created an API key",
                "detail": f"{k.name} · {k.prefix}",
                "actor": None,
                "at": k.created_at.isoformat(),
            })
        if k.last_used_at:
            events.append({
                "type": "apikey.used",
                "label": "Last used an API key",
                "detail": f"{k.name} · {k.usage_count} reqs",
                "actor": None,
                "at": k.last_used_at.isoformat(),
            })

    events.sort(key=lambda e: e["at"] or "", reverse=True)

    rec = db.get(UserChats, u.id)
    mem = db.get(UserMemory, u.id)
    mem_count = 0
    if mem:
        try:
            mem_count = len(json.loads(mem.facts) or [])
        except Exception:
            mem_count = 0
    key_total = db.query(func.count(ApiKey.id)).filter(ApiKey.user_id == u.id).scalar() or 0
    shared = db.query(func.count(SharedChat.id)).filter(SharedChat.owner_id == u.id).scalar() or 0

    return {
        "footprint": {
            "chats": _count_chats(rec.data if rec else None),
            "api_keys": int(key_total),
            "memory_facts": mem_count,
            "shared_chats": int(shared),
        },
        "events": events,
    }


# ── User mutations ───────────────────────────────────────────────────────────
class UpdateUserRequest(BaseModel):
    is_verified: Optional[bool] = None
    is_admin: Optional[bool] = None
    is_banned: Optional[bool] = None
    api_blocked: Optional[bool] = None


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

    # ── Block / unblock API access ──
    if req.api_blocked is not None and bool(getattr(u, "api_blocked", False)) != req.api_blocked:
        u.api_blocked = req.api_blocked
        if req.api_blocked:
            # Blocking also revokes every existing key so it stops working now.
            db.query(ApiKey).filter(ApiKey.user_id == u.id, ApiKey.revoked == False).update(  # noqa: E712
                {ApiKey.revoked: True}
            )
        changes.append("api_block" if req.api_blocked else "api_unblock")
        _record(db, admin, "apikey.block_user" if req.api_blocked else "apikey.unblock_user", u)

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

    dispatch_event("user.deleted", {"id": user_id, "email": target_email})
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
    """Permanently remove a key and its entire knowledge base (docs + vectors)."""
    rec = db.get(ApiKey, key_id)
    if rec is None:
        raise HTTPException(404, "Key not found.")
    owner = db.get(User, rec.user_id)
    prefix = rec.prefix
    # Cascade: drop the app's documents + ChromaDB collection so nothing is orphaned.
    db.query(KnowledgeDocument).filter(KnowledgeDocument.api_key_id == key_id).delete()
    try:
        delete_kb_collection(_kb_collection(key_id))
    except Exception:
        pass
    db.delete(rec)
    _record(db, admin, "apikey.delete", owner, detail=f"key {prefix}")
    db.commit()
    return {"ok": True}


# ── Developer apps (super-admin: every ck_ key across all users + its KB) ──────
# Each developer API key (ck_) is one "app" with an isolated knowledge base.
# These endpoints give a platform admin a cross-tenant view: who owns which app,
# what plan they're on, how many documents they uploaded, and the docs/activity.
ENTERPRISE_DOC_LIMIT = plan_service.UNLIMITED_DOC_LIMIT  # don't flag uncapped plans as full


def _plan_meta(plan_map: dict, plan: Optional[str]) -> tuple:
    p = plan_map.get(plan or "free") or plan_map.get("free") or {}
    return p.get("label", "Free"), int(p.get("doc_limit", 1))


def _app_row(k: ApiKey, owner: Optional[User], doc_count: int, total_size: int, plan_map: dict) -> dict:
    label, limit = _plan_meta(plan_map, k.plan)
    docs = int(doc_count or 0)
    return {
        "id": k.id,
        "name": k.name,
        "prefix": k.prefix,
        "plan": k.plan or "free",
        "plan_label": label,
        "doc_count": docs,
        "doc_limit": limit,
        "near_limit": limit < ENTERPRISE_DOC_LIMIT and docs >= limit,
        "total_size": int(total_size or 0),
        "usage_count": k.usage_count or 0,
        "total_tokens": k.total_tokens or 0,
        "revoked": bool(k.revoked),
        "widget_token": k.widget_token,
        "created_at": k.created_at.isoformat() if k.created_at else None,
        "last_used_at": k.last_used_at.isoformat() if k.last_used_at else None,
        "owner_id": owner.id if owner else k.user_id,
        "owner_email": owner.email if owner else None,
        "owner_name": owner.name if owner else None,
        "owner_api_blocked": bool(getattr(owner, "api_blocked", False)) if owner else False,
        "owner_banned": bool(getattr(owner, "is_banned", False)) if owner else False,
    }


def _doc_stats(db: Session, key_ids: List[int]) -> tuple:
    """Batch (count, total_size) of documents per api_key_id to avoid N+1."""
    counts: Dict[int, int] = {}
    sizes: Dict[int, int] = {}
    if key_ids:
        for kid, cnt, sz in (
            db.query(
                KnowledgeDocument.api_key_id,
                func.count(KnowledgeDocument.id),
                func.coalesce(func.sum(KnowledgeDocument.file_size), 0),
            )
            .filter(KnowledgeDocument.api_key_id.in_(key_ids))
            .group_by(KnowledgeDocument.api_key_id)
            .all()
        ):
            counts[kid] = cnt
            sizes[kid] = sz
    return counts, sizes


@router.get("/apps")
def list_apps(
    q: str = Query("", description="search owner email/name or app name/prefix"),
    plan: str = Query("", description="filter by plan key"),
    status: str = Query("all", description="all | active | revoked"),
    sort: str = Query("recent", description="recent | created | usage | docs"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    query = db.query(ApiKey, User).join(User, User.id == ApiKey.user_id)
    term = q.strip().lower()
    if term:
        like = f"%{term}%"
        query = query.filter(
            func.lower(User.email).like(like)
            | func.lower(User.name).like(like)
            | func.lower(ApiKey.name).like(like)
            | func.lower(ApiKey.prefix).like(like)
        )
    if plan.strip():
        query = query.filter(ApiKey.plan == plan.strip())
    if status == "active":
        query = query.filter(ApiKey.revoked == False)  # noqa: E712
    elif status == "revoked":
        query = query.filter(ApiKey.revoked == True)  # noqa: E712

    total = query.count()

    if sort == "usage":
        query = query.order_by(ApiKey.usage_count.desc())
    elif sort == "created":
        query = query.order_by(ApiKey.created_at.asc())
    else:  # "recent" (default) and "docs" (re-sorted in Python below)
        query = query.order_by(ApiKey.created_at.desc())

    rows = query.offset(offset).limit(limit).all()
    counts, sizes = _doc_stats(db, [k.id for k, _ in rows])
    pmap = plan_service.get_plan_map(db)
    apps = [_app_row(k, u, counts.get(k.id, 0), sizes.get(k.id, 0), pmap) for k, u in rows]
    if sort == "docs":  # doc_count is aggregated, not a column — sort the page here
        apps.sort(key=lambda a: a["doc_count"], reverse=True)

    return {"total": total, "limit": limit, "offset": offset, "apps": apps}


@router.get("/apps/summary")
def apps_summary(admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    total = db.query(func.count(ApiKey.id)).scalar() or 0
    active = db.query(func.count(ApiKey.id)).filter(ApiKey.revoked == False).scalar() or 0  # noqa: E712
    developers = db.query(func.count(func.distinct(ApiKey.user_id))).scalar() or 0
    total_docs = db.query(func.count(KnowledgeDocument.id)).scalar() or 0
    total_size = db.query(func.coalesce(func.sum(KnowledgeDocument.file_size), 0)).scalar() or 0
    api_calls = db.query(func.coalesce(func.sum(ApiKey.usage_count), 0)).scalar() or 0

    all_plans = plan_service.get_plans(db, include_inactive=True)
    plan_counts: Dict[str, int] = {p["key"]: 0 for p in all_plans}
    for plan_key, cnt in db.query(ApiKey.plan, func.count(ApiKey.id)).group_by(ApiKey.plan).all():
        key = plan_key or "free"
        plan_counts[key] = plan_counts.get(key, 0) + cnt

    return {
        "total_apps": total,
        "active_apps": active,
        "revoked_apps": total - active,
        "developers": developers,
        "total_docs": total_docs,
        "total_size": int(total_size or 0),
        "api_calls": api_calls,
        "plans": [
            {"key": p["key"], "label": p["label"], "price": p["price"], "count": plan_counts.get(p["key"], 0)}
            for p in all_plans
        ],
    }


@router.get("/api-keys/{key_id}/documents")
def app_documents(
    key_id: int,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    rec = db.get(ApiKey, key_id)
    if rec is None:
        raise HTTPException(404, "Key not found.")
    owner = db.get(User, rec.user_id)
    docs = (
        db.query(KnowledgeDocument)
        .filter(KnowledgeDocument.api_key_id == key_id)
        .order_by(KnowledgeDocument.uploaded_at.desc())
        .all()
    )
    label, limit = _plan_meta(plan_service.get_plan_map(db), rec.plan)
    return {
        "key_id": key_id,
        "name": rec.name,
        "plan": rec.plan or "free",
        "plan_label": label,
        "doc_limit": limit,
        "owner_email": owner.email if owner else None,
        "documents": [
            {
                "id": d.id,
                "filename": d.filename,
                "file_size": d.file_size or 0,
                "chunk_count": d.chunk_count or 0,
                "uploaded_at": d.uploaded_at.isoformat() if d.uploaded_at else None,
            }
            for d in docs
        ],
    }


@router.get("/api-keys/{key_id}/activity")
def app_activity(
    key_id: int,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """A per-app timeline built from data we already keep: key creation, every
    document upload, and the last API call. No new tracking table."""
    rec = db.get(ApiKey, key_id)
    if rec is None:
        raise HTTPException(404, "Key not found.")

    events: List[dict] = []
    if rec.created_at:
        events.append({"type": "app.created", "label": "App / key created",
                       "at": rec.created_at.isoformat(), "detail": None})
    docs = (
        db.query(KnowledgeDocument)
        .filter(KnowledgeDocument.api_key_id == key_id)
        .order_by(KnowledgeDocument.uploaded_at.desc())
        .all()
    )
    for d in docs:
        events.append({
            "type": "document.uploaded",
            "label": f"Uploaded “{d.filename}”",
            "at": d.uploaded_at.isoformat() if d.uploaded_at else None,
            "detail": f"{d.chunk_count or 0} chunks",
        })
    if rec.last_used_at:
        events.append({"type": "api.used", "label": "Most recent API call",
                       "at": rec.last_used_at.isoformat(), "detail": f"{rec.usage_count or 0} calls total"})
    events.sort(key=lambda e: e.get("at") or "", reverse=True)

    label, limit = _plan_meta(plan_service.get_plan_map(db), rec.plan)
    total_size = (
        db.query(func.coalesce(func.sum(KnowledgeDocument.file_size), 0))
        .filter(KnowledgeDocument.api_key_id == key_id)
        .scalar()
        or 0
    )
    return {
        "key_id": key_id,
        "events": events,
        "footprint": {
            "plan": rec.plan or "free",
            "plan_label": label,
            "doc_count": len(docs),
            "doc_limit": limit,
            "total_size": int(total_size),
            "usage_count": rec.usage_count or 0,
            "total_tokens": rec.total_tokens or 0,
        },
    }


class AppPatch(BaseModel):
    plan: Optional[str] = None
    name: Optional[str] = None


@router.patch("/api-keys/{key_id}")
def admin_update_key(
    key_id: int,
    req: AppPatch,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Admin override of a developer's app: change its plan (no payment system
    yet, so the admin grants plans manually) and/or rename it."""
    rec = db.get(ApiKey, key_id)
    if rec is None:
        raise HTTPException(404, "Key not found.")
    owner = db.get(User, rec.user_id)

    changes: List[str] = []
    if req.plan is not None:
        if req.plan not in plan_service.get_plan_map(db):
            raise HTTPException(400, f"Unknown plan '{req.plan}'.")
        if (rec.plan or "free") != req.plan:
            changes.append(f"plan {rec.plan or 'free'}→{req.plan}")
            rec.plan = req.plan
    if req.name is not None and req.name.strip() and req.name.strip() != rec.name:
        rec.name = req.name.strip()[:60]
        changes.append("rename")

    if changes:
        _record(db, admin, "apikey.update", owner, detail=f"key {rec.prefix}: {', '.join(changes)}")
        db.commit()
        db.refresh(rec)

    counts, sizes = _doc_stats(db, [rec.id])
    return _app_row(rec, owner, counts.get(rec.id, 0), sizes.get(rec.id, 0), plan_service.get_plan_map(db))


@router.delete("/api-keys/{key_id}/documents/{doc_id}")
def admin_delete_document(
    key_id: int,
    doc_id: int,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Remove one document from an app's knowledge base — DB row, ChromaDB
    vector chunks, and the stored file."""
    rec = db.get(ApiKey, key_id)
    if rec is None:
        raise HTTPException(404, "Key not found.")
    doc = db.get(KnowledgeDocument, doc_id)
    if doc is None or doc.api_key_id != key_id:
        raise HTTPException(404, "Document not found.")
    owner = db.get(User, rec.user_id)
    fname = doc.filename

    try:
        delete_kb_document_chunks(_kb_collection(key_id), doc.upload_uid)
    except Exception:
        pass  # vectors may already be gone — still remove the row + file
    file_path = os.path.join("uploads", _kb_collection(key_id), f"{doc.upload_uid}_{doc.filename}")
    try:
        os.remove(file_path)
    except FileNotFoundError:
        pass

    db.delete(doc)
    _record(db, admin, "apikey.doc_delete", owner, detail=f"key {rec.prefix}: {fname}")
    db.commit()
    return {"ok": True}


# ── Plan management (super-admin: edit pricing + services for app tiers) ───────
# Plans live in the `plans` table (plan_service). doc_limit is enforced on upload
# and rate_limit on the public API, so editing here changes real behaviour — not
# just the pricing page.
_PLAN_KEY_RE = re.compile(r"^[a-z0-9_]+$")


def _plan_app_counts(db: Session) -> Dict[str, int]:
    return {
        (k or "free"): c
        for k, c in db.query(ApiKey.plan, func.count(ApiKey.id)).group_by(ApiKey.plan).all()
    }


def _plan_out(p: dict, app_count: int) -> dict:
    return {**p, "app_count": app_count}


def _clean_features(feats: Optional[List[str]]) -> str:
    cleaned = [str(f).strip() for f in (feats or []) if str(f).strip()][:20]
    return json.dumps(cleaned)


class PlanIn(BaseModel):
    key: str = Field(min_length=1, max_length=30)
    label: str = Field(min_length=1, max_length=40)
    price: str = Field(default="₹0", max_length=40)
    doc_limit: int = Field(default=1, ge=0, le=1_000_000)
    rate_limit: int = Field(default=20, ge=1, le=100_000)
    blurb: str = Field(default="", max_length=200)
    features: List[str] = Field(default_factory=list)
    highlighted: bool = False
    active: bool = True


class PlanPatch(BaseModel):
    label: Optional[str] = Field(default=None, max_length=40)
    price: Optional[str] = Field(default=None, max_length=40)
    doc_limit: Optional[int] = Field(default=None, ge=0, le=1_000_000)
    rate_limit: Optional[int] = Field(default=None, ge=1, le=100_000)
    blurb: Optional[str] = Field(default=None, max_length=200)
    features: Optional[List[str]] = None
    highlighted: Optional[bool] = None
    active: Optional[bool] = None
    sort_order: Optional[int] = None


@router.get("/plans")
def admin_list_plans(admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    counts = _plan_app_counts(db)
    plans = plan_service.get_plans(db, include_inactive=True)
    return {"plans": [_plan_out(p, counts.get(p["key"], 0)) for p in plans]}


@router.post("/plans")
def admin_create_plan(req: PlanIn, admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    key = req.key.strip().lower()
    if not _PLAN_KEY_RE.match(key):
        raise HTTPException(400, "Plan key must be lowercase letters, numbers, or underscores.")
    if db.query(Plan).filter(Plan.key == key).first() is not None:
        raise HTTPException(400, f"A plan with key '{key}' already exists.")
    max_order = db.query(func.coalesce(func.max(Plan.sort_order), -1)).scalar()
    p = Plan(
        key=key, label=req.label.strip(), price=(req.price.strip() or "₹0"),
        doc_limit=req.doc_limit, rate_limit=req.rate_limit, blurb=req.blurb.strip(),
        features=_clean_features(req.features), sort_order=int(max_order) + 1,
        active=req.active, highlighted=req.highlighted,
    )
    db.add(p)
    _record(db, admin, "plan.create", detail=f"plan {key} ({req.label.strip()})")
    db.commit()
    db.refresh(p)
    counts = _plan_app_counts(db)
    return _plan_out(plan_service._to_dict(p), counts.get(key, 0))


@router.patch("/plans/{key}")
def admin_update_plan(key: str, req: PlanPatch, admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    p = db.query(Plan).filter(Plan.key == key).first()
    if p is None:
        raise HTTPException(404, "Plan not found.")
    # The 'free' tier is the default fallback for every new key — keep it usable.
    if key == "free" and req.active is False:
        raise HTTPException(400, "The Free plan is the default tier and can't be deactivated.")

    changes: List[str] = []
    if req.label is not None and req.label.strip():
        p.label = req.label.strip(); changes.append("label")
    if req.price is not None:
        p.price = req.price.strip() or "₹0"; changes.append("price")
    if req.doc_limit is not None:
        p.doc_limit = req.doc_limit; changes.append("doc_limit")
    if req.rate_limit is not None:
        p.rate_limit = req.rate_limit; changes.append("rate_limit")
    if req.blurb is not None:
        p.blurb = req.blurb.strip(); changes.append("blurb")
    if req.features is not None:
        p.features = _clean_features(req.features); changes.append("features")
    if req.highlighted is not None:
        p.highlighted = req.highlighted; changes.append("highlighted")
    if req.active is not None:
        p.active = req.active; changes.append("active")
    if req.sort_order is not None:
        p.sort_order = req.sort_order; changes.append("sort_order")

    if changes:
        p.updated_at = datetime.utcnow()
        _record(db, admin, "plan.update", detail=f"plan {key}: {', '.join(changes)}")
        db.commit()
        db.refresh(p)
    counts = _plan_app_counts(db)
    return _plan_out(plan_service._to_dict(p), counts.get(key, 0))


@router.delete("/plans/{key}")
def admin_delete_plan(key: str, admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    if key == "free":
        raise HTTPException(400, "The Free plan is the default tier and can't be deleted.")
    p = db.query(Plan).filter(Plan.key == key).first()
    if p is None:
        raise HTTPException(404, "Plan not found.")
    in_use = db.query(func.count(ApiKey.id)).filter(ApiKey.plan == key).scalar() or 0
    if in_use:
        raise HTTPException(400, f"{in_use} app(s) are on this plan — move them to another plan first.")
    db.delete(p)
    _record(db, admin, "plan.delete", detail=f"plan {key}")
    db.commit()
    return {"ok": True}


# ── Widget custom-CSS reviews (super-admin moderates dev-submitted CSS) ────────
class CssRejectPayload(BaseModel):
    note: str = ""


def _css_review_item(k: ApiKey, owner: Optional[User], cfg: dict) -> dict:
    return {
        "key_id": k.id,
        "app_name": k.name,
        "prefix": k.prefix,
        "owner_email": owner.email if owner else None,
        "owner_name": owner.name if owner else None,
        "pending_css": cfg.get("customCssPending", ""),
        "current_css": cfg.get("customCss", ""),
        "status": cfg.get("cssStatus", "none"),
        "submitted_at": cfg.get("cssSubmittedAt"),
    }


@router.get("/css-reviews")
def admin_css_reviews(admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    """Every app with custom CSS awaiting review — the super-admin's moderation queue."""
    rows = db.query(ApiKey).filter(ApiKey.widget_config.isnot(None)).all()
    items = []
    for k in rows:
        cfg = _read_widget_config(k)
        if cfg.get("cssStatus") == "pending":
            items.append(_css_review_item(k, db.get(User, k.user_id), cfg))
    items.sort(key=lambda x: x.get("submitted_at") or "", reverse=True)
    return {"reviews": items, "pending": len(items)}


@router.post("/css-reviews/{key_id}/approve")
def admin_approve_css(
    key_id: int,
    background: BackgroundTasks,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    rec = db.get(ApiKey, key_id)
    if rec is None:
        raise HTTPException(404, "App not found.")
    cfg = _read_widget_config(rec)
    if cfg.get("cssStatus") != "pending":
        raise HTTPException(400, "No pending CSS to approve.")
    cfg["customCss"] = cfg.get("customCssPending", "")
    cfg["customCssPending"] = ""
    cfg["cssStatus"] = "approved"
    cfg["cssNote"] = ""
    rec.widget_config = json.dumps(cfg)
    owner = db.get(User, rec.user_id)
    _record(db, admin, "widget.css_approve", owner, detail=f"key {rec.prefix}")
    db.commit()
    if owner and owner.email:
        background.add_task(
            send_announcement_email, owner.email, "Your widget CSS was approved",
            f"Your custom CSS for the app “{rec.name}” has been approved and is now live on your widget. 🎉",
        )
    return {"ok": True}


@router.post("/css-reviews/{key_id}/reject")
def admin_reject_css(
    key_id: int,
    req: CssRejectPayload,
    background: BackgroundTasks,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    rec = db.get(ApiKey, key_id)
    if rec is None:
        raise HTTPException(404, "App not found.")
    cfg = _read_widget_config(rec)
    if cfg.get("cssStatus") != "pending":
        raise HTTPException(400, "No pending CSS to reject.")
    note = (req.note or "").strip()[:300]
    cfg["cssStatus"] = "rejected"
    cfg["cssNote"] = note
    # Keep the pending CSS so the dev sees what was rejected; live CSS is untouched.
    rec.widget_config = json.dumps(cfg)
    owner = db.get(User, rec.user_id)
    _record(db, admin, "widget.css_reject", owner, detail=f"key {rec.prefix}: {note[:60]}")
    db.commit()
    if owner and owner.email:
        background.add_task(
            send_announcement_email, owner.email, "Your widget CSS needs changes",
            f"Your custom CSS for the app “{rec.name}” was not approved.\n\n"
            f"Reason: {note or 'not specified'}\n\nEdit it and resubmit in the Developer Console → Appearance.",
        )
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


# ── Broadcasts (platform announcement banner) ─────────────────────────────────
_BROADCAST_LEVELS = {"info", "warning", "success"}


def _broadcast_row(b: Broadcast) -> dict:
    return {
        "id": b.id,
        "message": b.message,
        "level": b.level,
        "active": bool(b.active),
        "created_by": b.created_by,
        "created_at": b.created_at.isoformat() if b.created_at else None,
    }


class BroadcastCreate(BaseModel):
    message: str
    level: str = "info"
    # When email_users is true, the announcement is ALSO emailed to every
    # verified, non-banned user (in the background). `subject` is the email
    # subject line (falls back to a default).
    subject: Optional[str] = None
    email_users: bool = False


class BroadcastPatch(BaseModel):
    active: bool


def _blast_announcement(subject: str, message: str, recipients: List[str]) -> None:
    """Background task: email the whole list over ONE reused SMTP connection so a
    blast goes out in ~1-2s instead of re-connecting per recipient. Best-effort."""
    try:
        send_announcement_bulk(recipients, subject, message)
    except Exception:
        pass


@router.get("/broadcasts")
def list_broadcasts(admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    rows = db.query(Broadcast).order_by(Broadcast.created_at.desc()).limit(50).all()
    return {"broadcasts": [_broadcast_row(b) for b in rows]}


@router.get("/announcement-audience")
def announcement_audience(admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    """How many users an email announcement would reach (verified, not banned).
    Drives the recipient-count shown in the composer before a mass send."""
    count = (
        db.query(func.count(User.id))
        .filter(User.is_verified == True, User.is_banned == False)  # noqa: E712
        .scalar()
        or 0
    )
    return {"recipients": int(count)}


@router.post("/broadcasts")
def create_broadcast(
    req: BroadcastCreate,
    background: BackgroundTasks,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    message = (req.message or "").strip()
    if not message:
        raise HTTPException(400, "Message can't be empty.")
    if len(message) > 500:
        raise HTTPException(400, "Message is too long (max 500 characters).")
    level = req.level if req.level in _BROADCAST_LEVELS else "info"
    # One active banner at a time — deactivate any currently-active ones first.
    db.query(Broadcast).filter(Broadcast.active == True).update({Broadcast.active: False})  # noqa: E712
    b = Broadcast(message=message, level=level, active=True, created_by=admin.email)
    db.add(b)
    _record(db, admin, "broadcast.create", detail=f"[{level}] {message[:120]}")

    # ── Optional email blast to every verified, non-banned user ──
    emailed = 0
    if req.email_users:
        subject = (req.subject or "").strip() or "Announcement from Close AI"
        recipients = [
            e
            for (e,) in db.query(User.email)
            .filter(User.is_verified == True, User.is_banned == False)  # noqa: E712
            .all()
            if e
        ]
        emailed = len(recipients)
        if recipients:
            # Background so SMTP latency for N users never blocks the request.
            background.add_task(_blast_announcement, subject, message, recipients)
            _record(db, admin, "broadcast.email_blast", detail=f"{emailed} recipients · {subject[:80]}")

    db.commit()
    db.refresh(b)
    dispatch_event("broadcast.published", {"id": b.id, "message": message, "level": level})
    row = _broadcast_row(b)
    row["emailed"] = emailed
    return row


@router.patch("/broadcasts/{broadcast_id}")
def update_broadcast(
    broadcast_id: int,
    req: BroadcastPatch,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    b = db.get(Broadcast, broadcast_id)
    if b is None:
        raise HTTPException(404, "Broadcast not found.")
    if req.active and not b.active:
        # Activating this one — deactivate the rest so only one shows.
        db.query(Broadcast).filter(Broadcast.active == True, Broadcast.id != b.id).update(  # noqa: E712
            {Broadcast.active: False}
        )
    b.active = req.active
    _record(
        db, admin,
        "broadcast.activate" if req.active else "broadcast.deactivate",
        detail=b.message[:120],
    )
    db.commit()
    db.refresh(b)
    return _broadcast_row(b)


@router.delete("/broadcasts/{broadcast_id}")
def delete_broadcast(
    broadcast_id: int,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    b = db.get(Broadcast, broadcast_id)
    if b is None:
        raise HTTPException(404, "Broadcast not found.")
    detail = b.message[:120]
    db.delete(b)
    _record(db, admin, "broadcast.delete", detail=detail)
    db.commit()
    return {"ok": True}


# ── Invites (admin-issued onboarding links) ───────────────────────────────────
INVITE_TTL_DAYS = 7


def _invite_row(inv: Invite) -> dict:
    return {
        "id": inv.id,
        "email": inv.email,
        "invited_by": inv.invited_by,
        "accepted": bool(inv.accepted),
        "expired": bool(inv.expires_at and inv.expires_at < datetime.utcnow()),
        "expires_at": inv.expires_at.isoformat() if inv.expires_at else None,
        "created_at": inv.created_at.isoformat() if inv.created_at else None,
        "link": f"{FRONTEND_URL}/?invite={inv.token}",
    }


class InviteCreate(BaseModel):
    email: EmailStr


@router.get("/invites")
def list_invites(admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    rows = db.query(Invite).order_by(Invite.created_at.desc()).limit(100).all()
    return {"invites": [_invite_row(i) for i in rows]}


@router.post("/invites")
def create_invite(
    req: InviteCreate,
    background: BackgroundTasks,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    email = req.email.lower().strip()
    existing_user = db.query(User).filter(User.email == email).first()
    if existing_user and existing_user.is_verified:
        raise HTTPException(400, "That email already has a registered account.")
    # Replace any prior un-accepted invite for this email (one live link per email).
    db.query(Invite).filter(Invite.email == email, Invite.accepted == False).delete()  # noqa: E712
    token = secrets.token_urlsafe(32)
    inv = Invite(
        email=email,
        token=token,
        invited_by=admin.email,
        expires_at=datetime.utcnow() + timedelta(days=INVITE_TTL_DAYS),
    )
    db.add(inv)
    _record(db, admin, "invite.create", detail=email)
    db.commit()
    db.refresh(inv)
    # Email the link in the background (best-effort); the admin also gets it in
    # the response to copy directly, so a blocked SMTP never fails the request.
    background.add_task(send_invite_email, email, f"{FRONTEND_URL}/?invite={token}", admin.email)
    return _invite_row(inv)


@router.delete("/invites/{invite_id}")
def delete_invite(
    invite_id: int,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    inv = db.get(Invite, invite_id)
    if inv is None:
        raise HTTPException(404, "Invite not found.")
    email = inv.email
    db.delete(inv)
    _record(db, admin, "invite.revoke", detail=email)
    db.commit()
    return {"ok": True}


# ── Webhooks (outbound platform event notifications) ──────────────────────────
def _webhook_events(h: Webhook) -> List[str]:
    try:
        evs = json.loads(h.events or "[]")
        return evs if isinstance(evs, list) else []
    except Exception:
        return []


def _webhook_row(h: Webhook) -> dict:
    return {
        "id": h.id,
        "url": h.url,
        "events": _webhook_events(h),
        "enabled": bool(h.enabled),
        "created_by": h.created_by,
        "last_status": h.last_status,
        "last_triggered_at": h.last_triggered_at.isoformat() if h.last_triggered_at else None,
        "created_at": h.created_at.isoformat() if h.created_at else None,
    }


def _clean_events(events: List[str]) -> List[str]:
    # Keep only known events, de-duplicated, preserving the catalogue order.
    chosen = set(events)
    return [e for e in WEBHOOK_EVENTS if e in chosen]


class WebhookCreate(BaseModel):
    url: str
    events: List[str]


class WebhookPatch(BaseModel):
    enabled: Optional[bool] = None
    events: Optional[List[str]] = None


@router.get("/webhooks")
def list_webhooks(admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    rows = db.query(Webhook).order_by(Webhook.created_at.desc()).all()
    return {"webhooks": [_webhook_row(h) for h in rows], "events": WEBHOOK_EVENTS}


@router.post("/webhooks")
def create_webhook(
    req: WebhookCreate,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    url = (req.url or "").strip()
    if not (url.startswith("http://") or url.startswith("https://")):
        raise HTTPException(400, "Webhook URL must start with http:// or https://")
    events = _clean_events(req.events)
    if not events:
        raise HTTPException(400, "Select at least one event to subscribe to.")
    secret = "whsec_" + secrets.token_urlsafe(24)
    h = Webhook(url=url, secret=secret, events=json.dumps(events), enabled=True, created_by=admin.email)
    db.add(h)
    _record(db, admin, "webhook.create", detail=url)
    db.commit()
    db.refresh(h)
    row = _webhook_row(h)
    # The signing secret is shown ONCE on creation (like an API key).
    row["secret"] = secret
    return row


@router.patch("/webhooks/{webhook_id}")
def update_webhook(
    webhook_id: int,
    req: WebhookPatch,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    h = db.get(Webhook, webhook_id)
    if h is None:
        raise HTTPException(404, "Webhook not found.")
    if req.enabled is not None and bool(h.enabled) != req.enabled:
        h.enabled = req.enabled
        _record(db, admin, "webhook.enable" if req.enabled else "webhook.disable", detail=h.url)
    if req.events is not None:
        events = _clean_events(req.events)
        if not events:
            raise HTTPException(400, "A webhook must subscribe to at least one event.")
        h.events = json.dumps(events)
        _record(db, admin, "webhook.update", detail=h.url)
    db.commit()
    db.refresh(h)
    return _webhook_row(h)


@router.delete("/webhooks/{webhook_id}")
def delete_webhook(
    webhook_id: int,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    h = db.get(Webhook, webhook_id)
    if h is None:
        raise HTTPException(404, "Webhook not found.")
    url = h.url
    db.delete(h)
    _record(db, admin, "webhook.delete", detail=url)
    db.commit()
    return {"ok": True}


@router.post("/webhooks/{webhook_id}/test")
def test_webhook(
    webhook_id: int,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Send a sample event to this webhook now and report the delivery status."""
    h = db.get(Webhook, webhook_id)
    if h is None:
        raise HTTPException(404, "Webhook not found.")
    deliver_test(h.id)        # synchronous single delivery (commits in its own session)
    db.refresh(h)
    return {"ok": True, "last_status": h.last_status}


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
