"""Outbound platform webhooks.

When a subscribed platform event fires, POST a signed JSON payload to every
admin-registered webhook subscribed to it. Delivery runs in a daemon thread
(fire-and-forget) so it never blocks — or fails — the request that triggered it.
Each request is signed with the webhook's secret (HMAC-SHA256) in the
`X-CloseAI-Signature` header so receivers can verify authenticity.
"""

import hashlib
import hmac
import json
import logging
import threading
from datetime import datetime

import requests

from app.db import SessionLocal
from app.models import Webhook

log = logging.getLogger("close_ai.webhooks")

# The platform events an admin can subscribe a webhook to. The frontend reads
# this catalogue from GET /admin/webhooks, so adding one here surfaces it in the UI.
WEBHOOK_EVENTS = [
    "user.signup",
    "user.deleted",
    "apikey.created",
    "broadcast.published",
]


def _sign(secret: str, body: bytes) -> str:
    return "sha256=" + hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()


def _events_of(hook: Webhook) -> list:
    try:
        evs = json.loads(hook.events or "[]")
        return evs if isinstance(evs, list) else []
    except Exception:
        return []


def _deliver_one(hook_id: int, event: str, payload: dict, force: bool = False) -> None:
    """Open a fresh session (we may be in a worker thread), POST the signed
    payload, and record the result. `force` delivers even to a disabled hook
    (used by the explicit 'Test' button)."""
    db = SessionLocal()
    try:
        hook = db.get(Webhook, hook_id)
        if hook is None:
            return
        if not hook.enabled and not force:
            return
        body = json.dumps(
            {"event": event, "data": payload, "sent_at": datetime.utcnow().isoformat() + "Z"},
            default=str,
        ).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "CloseAI-Webhook/1.0",
            "X-CloseAI-Event": event,
            "X-CloseAI-Signature": _sign(hook.secret or "", body),
        }
        status = "error"
        try:
            resp = requests.post(hook.url, data=body, headers=headers, timeout=8)
            status = str(resp.status_code)
        except Exception as exc:  # noqa: BLE001 — record the failure, never raise
            status = f"error: {type(exc).__name__}"
            log.warning("Webhook %s delivery failed: %s", hook_id, exc)
        hook.last_status = status[:120]
        hook.last_triggered_at = datetime.utcnow()
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()


def deliver_test(hook_id: int) -> None:
    """Synchronously deliver a sample event (so the route can return the status).
    Works even on a disabled hook."""
    _deliver_one(
        hook_id,
        "webhook.test",
        {"message": "Test event from Close AI — your webhook endpoint is reachable."},
        force=True,
    )


def dispatch_event(event: str, payload: dict) -> None:
    """Find every enabled webhook subscribed to `event` and deliver to each in a
    background thread. Best-effort: never raises into the caller."""
    try:
        db = SessionLocal()
        try:
            hooks = db.query(Webhook).filter(Webhook.enabled == True).all()  # noqa: E712
            targets = [h.id for h in hooks if event in _events_of(h)]
        finally:
            db.close()
        for hook_id in targets:
            threading.Thread(
                target=_deliver_one, args=(hook_id, event, payload), daemon=True
            ).start()
    except Exception:
        log.exception("dispatch_event failed for %s", event)
