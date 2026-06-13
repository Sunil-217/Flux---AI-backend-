"""Read / write the platform feature flags.

Effective state = DEFAULT_FEATURES overlaid with any saved overrides. Unknown
keys are ignored so a stale frontend can never write a bogus flag.
"""

from datetime import datetime

from sqlalchemy.orm import Session

from app.core.features import DEFAULT_FEATURES
from app.models import FeatureFlag


def get_effective_features(db: Session) -> dict[str, bool]:
    flags = dict(DEFAULT_FEATURES)
    for row in db.query(FeatureFlag).all():
        if row.key in flags:
            flags[row.key] = bool(row.enabled)
    return flags


def set_features(db: Session, updates: dict) -> dict[str, bool]:
    """Upsert the given {key: bool} overrides (only valid keys), then return the
    new effective map. Commits once."""
    changed = False
    for key, enabled in (updates or {}).items():
        if key not in DEFAULT_FEATURES:
            continue
        enabled = bool(enabled)
        row = db.get(FeatureFlag, key)
        if row is None:
            db.add(FeatureFlag(key=key, enabled=enabled))
        else:
            row.enabled = enabled
            row.updated_at = datetime.utcnow()
        changed = True
    if changed:
        db.commit()
    return get_effective_features(db)
