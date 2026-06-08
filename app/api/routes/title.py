from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.core.security import get_current_user
from app.models import User
from app.services.rag_service import generate_title

router = APIRouter()


class TitleRequest(BaseModel):
    question: str


@router.post("/title")
def title(req: TitleRequest, user: User = Depends(get_current_user)):
    """Return a short, smart title for a chat based on its first message."""
    return {"title": generate_title(req.question)}
