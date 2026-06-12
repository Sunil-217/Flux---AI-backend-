from fastapi import APIRouter, Depends
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from app.core.security import get_current_user
from app.models import User
from app.services.research_service import deep_research

router = APIRouter()


class ResearchRequest(BaseModel):
    question: str


@router.post("/research")
async def research(req: ResearchRequest, user: User = Depends(get_current_user)):
    """Deep research: plan queries → search the web → synthesize a cited report.

    Blocking LLM + search calls, so off-load to a threadpool (same pattern as
    assist.py) to avoid pinning a FastAPI worker for the whole pipeline.
    """
    return await run_in_threadpool(deep_research, req.question)
