"""Server-side per-user chat storage (load / save the user's sessions)."""

import json
from datetime import datetime
from typing import Any, List

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db import get_db
from app.core.security import get_current_user
from app.models import User, UserChats

router = APIRouter(tags=["chats"])


class ChatsPayload(BaseModel):
    data: List[Any]  # the sessions array (opaque to the backend)


@router.get("/chats")
def get_chats(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    rec = db.get(UserChats, user.id)
    if rec is None:
        return {"data": []}
    try:
        return {"data": json.loads(rec.data)}
    except Exception:
        return {"data": []}


@router.put("/chats")
def save_chats(
    payload: ChatsPayload,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    serialized = json.dumps(payload.data)
    rec = db.get(UserChats, user.id)
    if rec is None:
        db.add(UserChats(user_id=user.id, data=serialized))
    else:
        rec.data = serialized
        rec.updated_at = datetime.utcnow()
    db.commit()
    return {"ok": True}
