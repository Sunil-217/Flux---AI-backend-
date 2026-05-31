from typing import List, Optional

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

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


@router.post("/chat")
async def chat(
    request: ChatRequest
):

    history = [
        {"role": m.role, "content": m.content}
        for m in (request.history or [])
    ]

    return StreamingResponse(
        stream_question(
            request.chat_id,
            request.question,
            history
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
