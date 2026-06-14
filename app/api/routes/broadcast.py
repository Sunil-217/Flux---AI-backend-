"""Public read of the active platform announcement (banner).

Any client (even unauthenticated) can ask for the current announcement so the
app can show a banner. Read-only — posting / editing announcements is admin-only
(see /admin/broadcasts)."""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Broadcast

router = APIRouter(tags=["broadcast"])


@router.get("/broadcast")
def active_broadcast(db: Session = Depends(get_db)):
    rec = (
        db.query(Broadcast)
        .filter(Broadcast.active == True)  # noqa: E712
        .order_by(Broadcast.created_at.desc())
        .first()
    )
    if rec is None:
        return {"broadcast": None}
    return {
        "broadcast": {
            "id": rec.id,
            "message": rec.message,
            "level": rec.level,
            "created_at": rec.created_at.isoformat() if rec.created_at else None,
        }
    }
