from fastapi import APIRouter
from pydantic import BaseModel

from app.services.rag_service import (
    ask_question
)

router = APIRouter()


class ChatRequest(BaseModel):

    chat_id: str

    question: str


@router.post("/chat")
async def chat(
    request: ChatRequest
):

    response = ask_question(
        request.chat_id,
        request.question
    )

    return response

