from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from app.services.rag_service import (
    ask_question
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

    response = ask_question(
        request.chat_id,
        request.question,
        history
    )

    return response

