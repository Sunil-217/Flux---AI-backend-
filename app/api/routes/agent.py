"""POST /agent/task — the autonomous multi-agent path.

Added as a SEPARATE endpoint. /chat is untouched: same request model, same
events, same behaviour, so every existing client keeps working exactly as
before and this can be adopted per-request.

The endpoint is safe to send *every* message to. When the orchestrator decides
a goal does not warrant a plan — small talk, a one-shot question, planning
unavailable — it emits `delegate` and this route transparently runs the ordinary
`stream_question` instead. The caller gets the normal chat stream; the only
difference is that a genuinely multi-step goal also gets `status` events.

Authentication and ownership are the existing ones. An agent run reads the same
user's documents and memory through the same checks as chat, so there is no path
here to data a chat request could not already reach.
"""

import json
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.config import AGENT_ORCHESTRATION_ENABLED
from app.core.security import get_current_user, user_owns_chat
from app.db import get_db
from app.models import User

router = APIRouter(tags=["agent"])


class HistoryMessage(BaseModel):
    role: str
    content: str


class AgentTaskRequest(BaseModel):
    chat_id: str
    # Same field name as ChatRequest so a client can post the same body here.
    question: str = Field(..., min_length=1, max_length=8000)
    history: Optional[List[HistoryMessage]] = []
    style: Optional[str] = None
    custom_instructions: Optional[str] = None
    web_search: Optional[bool] = True
    active_docs: Optional[List[str]] = None


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@router.post("/agent/task")
async def agent_task(
    request: AgentTaskRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not user_owns_chat(db, user, request.chat_id):
        raise HTTPException(status_code=404, detail="Chat not found")

    history = [{"role": m.role, "content": m.content} for m in (request.history or [])]
    web_enabled = request.web_search if request.web_search is not None else True
    active_docs = request.active_docs or []

    # Long-term memory, read through the same helper the chat route uses.
    from app.agents.memory import long_term_block
    custom_instructions = (request.custom_instructions or "") + long_term_block(db, user.id)

    def _fallback_chat():
        """The ordinary chat stream, byte for byte — this is what /chat does."""
        from app.services.rag_service import stream_question

        yield from stream_question(
            request.chat_id, request.question, history, None,
            request.style, custom_instructions or None, web_enabled, active_docs,
        )

    def _events():
        if not AGENT_ORCHESTRATION_ENABLED:
            yield from _fallback_chat()
            yield _sse({"type": "done"})
            return

        from app.agents.orchestrator import run_task, should_orchestrate
        from app.services.chroma_service import get_or_create_collection

        try:
            has_documents = get_or_create_collection(request.chat_id).count() > 0
        except Exception:
            has_documents = False

        # Cheap triage first: a simple message never pays for the agent stack.
        if not should_orchestrate(request.question):
            yield from _fallback_chat()
            yield _sse({"type": "done"})
            return

        for event in run_task(
            request.question,
            chat_id=request.chat_id,
            history=history,
            web_enabled=web_enabled,
            active_docs=active_docs,
            has_documents=has_documents,
        ):
            # The orchestrator asking to delegate is not an answer — run the
            # ordinary path and stream that instead.
            if event.get("type") == "delegate":
                yield from _fallback_chat()
                yield _sse({"type": "done"})
                return
            yield _sse(event)

    return StreamingResponse(
        _events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/agent/providers")
async def providers(user: User = Depends(get_current_user)):
    """Non-secret provider health, for diagnosing a misconfigured deployment.

    Reports whether each provider is configured and reachable and which models
    have been marked dead — never a key, a base URL credential, or a raw
    provider message. Authenticated so it is not a public fingerprint of the
    deployment's LLM setup.
    """
    from app.core.config import (
        CHAT_PROVIDER, CODE_PROVIDER, ORCHESTRATOR_PROVIDER,
        PLANNER_PROVIDER, ROUTER_PROVIDER, VISION_PROVIDER,
    )
    from app.services.llm_provider import provider_status

    return {
        "providers": provider_status(),
        "roles": {
            "chat": CHAT_PROVIDER, "router": ROUTER_PROVIDER, "code": CODE_PROVIDER,
            "vision": VISION_PROVIDER, "plan": PLANNER_PROVIDER,
            "orchestrate": ORCHESTRATOR_PROVIDER,
        },
        "orchestration_enabled": AGENT_ORCHESTRATION_ENABLED,
    }
