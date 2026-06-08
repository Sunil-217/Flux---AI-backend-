from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.security import get_current_user, user_owns_chat
from app.db import get_db
from app.models import User
from app.services.rag_service import (
    stream_question
)

router = APIRouter()


class HistoryMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):

    chat_id: str

    question: str

    history: Optional[List[HistoryMessage]] = []

    # Optional base64 data-URI image / screenshot to ask about (vision).
    image: Optional[str] = None

    # Optional response-style preset + custom instructions from the user's Settings.
    style: Optional[str] = None
    custom_instructions: Optional[str] = None

    # Whether to allow live web grounding for this turn (user toggle).
    web_search: Optional[bool] = True

    # Restrict document Q&A to these uploaded filenames (multi-doc picker).
    active_docs: Optional[List[str]] = None


@router.post("/chat")
async def chat(
    request: ChatRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not user_owns_chat(db, user, request.chat_id):
        raise HTTPException(status_code=404, detail="Chat not found")

    history = [
        {"role": m.role, "content": m.content}
        for m in (request.history or [])
    ]

    return StreamingResponse(
        stream_question(
            request.chat_id,
            request.question,
            history,
            request.image,
            request.style,
            request.custom_instructions,
            request.web_search if request.web_search is not None else True,
            request.active_docs,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
