"""Public developer API — OpenAI-compatible, authenticated by Close AI keys.

Developers point any OpenAI SDK at this server:

    from openai import OpenAI
    client = OpenAI(base_url="http://<host>:8000/v1", api_key="ck_...")
    client.chat.completions.create(model="close-chat", messages=[...])

Endpoints:
  GET  /v1/models            — list available model aliases
  POST /v1/chat/completions  — chat (stream + non-stream), OpenAI response shape
"""

import json
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from app.api.routes.apikeys import hash_key
from app.db import get_db
from app.models import ApiKey, User
from app.services.rag_service import (
    MODEL,
    CODE_MODEL,
    client as nvidia_client,
    _groq_or_nvidia,
)
from app.services import plan_service

router = APIRouter()

# Model aliases exposed to developers → (upstream model id, which client).
MODEL_ALIASES = {
    "close-chat": (MODEL, "groq"),        # fast general chat (llama-3.3-70b on Groq)
    "close-code": (CODE_MODEL, "nvidia"),  # strongest free coder (qwen3-coder-480b)
}
DEFAULT_ALIAS = "close-chat"

# Per-key sliding-window rate limit. In-memory: fine for a single uvicorn
# worker; swap for Redis if this ever runs multi-worker. The per-minute ceiling
# comes from the key's plan (plan_service), so upgrading a plan lifts the limit.
RATE_LIMIT_PER_MIN = 20  # fallback when a plan can't be resolved
_windows: dict[int, deque] = defaultdict(deque)


def _check_rate(key_id: int, limit: int = RATE_LIMIT_PER_MIN) -> None:
    now = time.time()
    win = _windows[key_id]
    while win and now - win[0] > 60:
        win.popleft()
    if len(win) >= limit:
        raise HTTPException(429, f"Rate limit exceeded ({limit} requests/min). Slow down or upgrade your plan.")
    win.append(now)


def require_api_key(
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
) -> ApiKey:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Missing API key. Send: Authorization: Bearer ck_...")
    raw = authorization.split(" ", 1)[1].strip()
    if not raw.startswith("ck_"):
        raise HTTPException(401, "Invalid API key format (expected ck_...).")
    rec = db.query(ApiKey).filter(ApiKey.key_hash == hash_key(raw)).first()
    if rec is None or rec.revoked:
        raise HTTPException(401, "Invalid or revoked API key.")
    # The key owner can be blocked from the API (or banned entirely) by an admin.
    owner = db.get(User, rec.user_id)
    if owner is None or getattr(owner, "api_blocked", False) or getattr(owner, "is_banned", False):
        raise HTTPException(403, "API access for this account has been blocked.")
    _check_rate(rec.id, plan_service.rate_limit_for(db, rec.plan))
    return rec


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = DEFAULT_ALIAS
    messages: List[ChatMessage] = Field(min_length=1)
    temperature: float = Field(default=0.4, ge=0, le=2)
    max_tokens: int = Field(default=1024, ge=1, le=4096)
    stream: bool = False


def _resolve(alias: str):
    if alias in MODEL_ALIASES:
        upstream, provider = MODEL_ALIASES[alias]
    else:
        # Unknown name → default chat model (lenient, like many gateways).
        upstream, provider = MODEL_ALIASES[DEFAULT_ALIAS]
    return upstream, (nvidia_client if provider == "nvidia" else _groq_or_nvidia())


def _bump_usage(db: Session, key_id: int, tokens: int) -> None:
    rec = db.get(ApiKey, key_id)
    if rec is not None:
        rec.usage_count = (rec.usage_count or 0) + 1
        rec.total_tokens = (rec.total_tokens or 0) + max(0, tokens)
        rec.last_used_at = datetime.utcnow()
        db.commit()


@router.get("/v1/models")
def list_models(key: ApiKey = Depends(require_api_key)):
    return {
        "object": "list",
        "data": [
            {"id": alias, "object": "model", "owned_by": "close-ai"}
            for alias in MODEL_ALIASES
        ],
    }


@router.post("/v1/chat/completions")
async def chat_completions(
    req: ChatCompletionRequest,
    key: ApiKey = Depends(require_api_key),
    db: Session = Depends(get_db),
):
    upstream_model, upstream = _resolve(req.model)
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    if not req.stream:
        try:
            resp = await run_in_threadpool(
                lambda: upstream.chat.completions.create(
                    model=upstream_model,
                    messages=messages,
                    temperature=req.temperature,
                    max_tokens=req.max_tokens,
                )
            )
        except Exception:
            raise HTTPException(502, "Upstream model call failed. Try again.")
        content = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        completion_tokens = getattr(usage, "completion_tokens", None) or max(1, len(content) // 4)
        _bump_usage(db, key.id, completion_tokens)
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "completion_tokens": completion_tokens,
                "total_tokens": getattr(usage, "total_tokens", 0) or completion_tokens,
            },
        }

    # ── Streaming (SSE, OpenAI chunk format, ends with data: [DONE]) ──
    def event_stream():
        sent_chars = 0
        try:
            stream = upstream.chat.completions.create(
                model=upstream_model,
                messages=messages,
                temperature=req.temperature,
                max_tokens=req.max_tokens,
                stream=True,
            )
            for chunk in stream:
                try:
                    delta = chunk.choices[0].delta.content or ""
                except Exception:
                    delta = ""
                if not delta:
                    continue
                sent_chars += len(delta)
                payload = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": req.model,
                    "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(payload)}\n\n"
            done = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": req.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(done)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception:
            yield f'data: {json.dumps({"error": "Upstream model call failed."})}\n\n'
            yield "data: [DONE]\n\n"
        finally:
            # Rough token estimate for streams (~4 chars/token).
            from app.db import SessionLocal

            s = SessionLocal()
            try:
                _bump_usage(s, key.id, sent_chars // 4)
            finally:
                s.close()

    return StreamingResponse(event_stream(), media_type="text/event-stream")
