"""Plan tiers for developer apps — DB-backed, admin-editable.

Plans used to be a hardcoded list in `kb.py`; they now live in the `plans`
table so admins can edit price, document limits, API rate limits and the list of
services from the Plans tab (no redeploy needed). This module is the single
source of truth: every caller reads plans through here.

`DEFAULT_PLANS` is the initial seed AND a safety fallback if the table is ever
empty, so doc-limit / rate-limit enforcement never breaks.
"""

import json
from typing import List, Optional

from sqlalchemy.orm import Session

from app.models import Plan

# "Unlimited" sentinel — a plan with this doc_limit is treated as uncapped and
# never flagged as "near limit".
UNLIMITED_DOC_LIMIT = 100000

DEFAULT_PLANS = [
    {
        "key": "free", "label": "Free", "price": "₹0", "doc_limit": 1, "rate_limit": 20,
        "blurb": "1 document, embeddable chat widget", "highlighted": False,
        "features": ["1 knowledge-base document", "Embeddable chat widget", "20 API requests / min", "Community support"],
    },
    {
        "key": "go", "label": "Go", "price": "₹299/mo", "doc_limit": 3, "rate_limit": 40,
        "blurb": "3 documents, higher rate limits", "highlighted": False,
        "features": ["3 knowledge-base documents", "Embeddable chat widget", "40 API requests / min", "Email support"],
    },
    {
        "key": "pro", "label": "Pro", "price": "₹799/mo", "doc_limit": 7, "rate_limit": 80,
        "blurb": "7 documents, priority answers", "highlighted": True,
        "features": ["7 knowledge-base documents", "Priority answers", "80 API requests / min", "Email support"],
    },
    {
        "key": "max", "label": "Max", "price": "₹1,499/mo", "doc_limit": 10, "rate_limit": 150,
        "blurb": "10 documents, analytics", "highlighted": False,
        "features": ["10 knowledge-base documents", "Usage analytics", "150 API requests / min", "Priority support"],
    },
    {
        "key": "enterprise", "label": "Enterprise", "price": "Custom", "doc_limit": UNLIMITED_DOC_LIMIT, "rate_limit": 1000,
        "blurb": "Unlimited documents, SSO, SLA", "highlighted": False,
        "features": ["Unlimited documents", "SSO & SLA", "1000 API requests / min", "Dedicated support"],
    },
]


def _parse_features(raw: Optional[str]) -> List[str]:
    try:
        feats = json.loads(raw) if raw else []
        return [str(f) for f in feats] if isinstance(feats, list) else []
    except Exception:
        return []


def _to_dict(p: Plan) -> dict:
    return {
        "key": p.key,
        "label": p.label,
        "price": p.price,
        "doc_limit": p.doc_limit,
        "rate_limit": p.rate_limit,
        "blurb": p.blurb or "",
        "features": _parse_features(p.features),
        "sort_order": p.sort_order,
        "active": bool(p.active),
        "highlighted": bool(p.highlighted),
    }


def _default_dicts() -> List[dict]:
    return [{**p, "sort_order": i, "active": True} for i, p in enumerate(DEFAULT_PLANS)]


def seed_default_plans(db: Session) -> None:
    """Populate the plans table on first boot. Idempotent — no-op if any row
    already exists, so admin edits are never overwritten on restart."""
    if db.query(Plan).count() > 0:
        return
    for i, p in enumerate(DEFAULT_PLANS):
        db.add(
            Plan(
                key=p["key"], label=p["label"], price=p["price"], doc_limit=p["doc_limit"],
                rate_limit=p["rate_limit"], blurb=p["blurb"], features=json.dumps(p["features"]),
                sort_order=i, active=True, highlighted=p["highlighted"],
            )
        )
    db.commit()


def get_plans(db: Session, include_inactive: bool = False) -> List[dict]:
    q = db.query(Plan)
    if not include_inactive:
        q = q.filter(Plan.active == True)  # noqa: E712
    rows = q.order_by(Plan.sort_order.asc(), Plan.id.asc()).all()
    if not rows:
        # Fallback before seeding (or if the table was emptied) so callers never
        # see zero plans and enforcement keeps working.
        plans = _default_dicts()
        return plans if include_inactive else [p for p in plans if p["active"]]
    return [_to_dict(r) for r in rows]


def get_plan_map(db: Session) -> dict:
    """All plans (incl. inactive) keyed by plan key."""
    return {p["key"]: p for p in get_plans(db, include_inactive=True)}


def get_plan(db: Session, key: Optional[str]) -> Optional[dict]:
    m = get_plan_map(db)
    return m.get(key or "free") or m.get("free") or (next(iter(m.values()), None))


def doc_limit_for(db: Session, key: Optional[str]) -> int:
    p = get_plan(db, key)
    return int(p["doc_limit"]) if p else 1


def rate_limit_for(db: Session, key: Optional[str]) -> int:
    p = get_plan(db, key)
    return int(p.get("rate_limit", 20)) if p else 20


def plan_label(db: Session, key: Optional[str]) -> str:
    p = get_plan(db, key)
    return p["label"] if p else "Free"
