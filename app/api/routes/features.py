"""Public read of the platform feature flags.

Any client (even unauthenticated) can ask which features are enabled so the UI
can hide disabled capabilities. Read-only — changing flags is admin-only
(see /admin/features).
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db import get_db
from app.services.feature_service import get_effective_features

router = APIRouter(tags=["features"])


@router.get("/features")
def public_features(db: Session = Depends(get_db)):
    return {"features": get_effective_features(db)}
