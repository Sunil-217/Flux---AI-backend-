"""Read-only public chat sharing: create a snapshot (auth) and view it (public)."""

import json
import uuid
from typing import Any, List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db import get_db
from app.core.security import get_current_user
from app.models import User, SharedChat

router = APIRouter(tags=["share"])


class ShareRequest(BaseModel):
    title: str = "Shared chat"
    messages: List[Any]


@router.post("/share")
def create_share(
    req: ShareRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # Full UUID4 hex (128 bits) — short slugs would be brute-forceable at scale.
    sid = uuid.uuid4().hex
    clean = []
    for m in (req.messages or [])[:200]:
        if not isinstance(m, dict):
            continue
        item = {"role": m.get("role"), "content": (m.get("content") or "")[:20000]}
        if m.get("image"):
            item["image"] = m.get("image")
        clean.append(item)
    rec = SharedChat(
        id=sid,
        owner_id=user.id,
        title=(req.title or "Shared chat")[:200],
        data=json.dumps(clean),
    )
    db.add(rec)
    db.commit()
    return {"id": sid}


@router.get("/share/{share_id}")
def get_share(share_id: str, db: Session = Depends(get_db)):
    rec = db.get(SharedChat, share_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="Shared chat not found")
    try:
        messages = json.loads(rec.data)
    except Exception:
        messages = []
    return {
        "title": rec.title,
        "messages": messages,
        "created_at": rec.created_at.isoformat(),
    }
