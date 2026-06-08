from typing import List

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from app.core.security import get_current_user
from app.models import User
from app.services.rag_service import (
    generate_followups,
    translate_text,
    summarize_conversation,
    edit_code_file,
    plan_code_changes,
    answer_code_question,
)

router = APIRouter()


class FollowupsRequest(BaseModel):
    question: str
    answer: str


class TranslateRequest(BaseModel):
    text: str
    language: str


class SummaryMessage(BaseModel):
    role: str
    content: str


class SummaryRequest(BaseModel):
    history: List[SummaryMessage]


class CodeTurn(BaseModel):
    role: str
    content: str


class EditFileRequest(BaseModel):
    filename: str
    content: str
    instruction: str
    history: List[CodeTurn] = []


class AgentPlanRequest(BaseModel):
    tree: List[str]
    instruction: str
    history: List[CodeTurn] = []


class CodeContextFile(BaseModel):
    path: str
    content: str


class CodeAnswerRequest(BaseModel):
    question: str
    files: List[CodeContextFile]
    history: List[CodeTurn] = []


# Each of these handlers calls a blocking NVIDIA SDK call. We MUST off-load to a
# threadpool so a slow upstream doesn't pin a FastAPI worker and freeze every
# concurrent request.


@router.post("/followups")
async def followups(req: FollowupsRequest, user: User = Depends(get_current_user)):
    qs = await run_in_threadpool(generate_followups, req.question, req.answer)
    return {"questions": qs}


@router.post("/translate")
async def translate(req: TranslateRequest, user: User = Depends(get_current_user)):
    text = await run_in_threadpool(translate_text, req.text, req.language)
    return {"text": text}


@router.post("/summary")
async def summary(req: SummaryRequest, user: User = Depends(get_current_user)):
    history = [{"role": m.role, "content": m.content} for m in req.history]
    s = await run_in_threadpool(summarize_conversation, history)
    return {"summary": s}


@router.post("/edit-file")
async def edit_file(req: EditFileRequest, user: User = Depends(get_current_user)):
    history = [{"role": h.role, "content": h.content} for h in req.history]
    content = await run_in_threadpool(
        edit_code_file, req.filename, req.content, req.instruction, history
    )
    return {"content": content}


@router.post("/agent-plan")
async def agent_plan(req: AgentPlanRequest, user: User = Depends(get_current_user)):
    history = [{"role": h.role, "content": h.content} for h in req.history]
    return await run_in_threadpool(plan_code_changes, req.tree, req.instruction, history)


@router.post("/code-answer")
async def code_answer(req: CodeAnswerRequest, user: User = Depends(get_current_user)):
    files = [{"path": f.path, "content": f.content} for f in req.files]
    history = [{"role": h.role, "content": h.content} for h in req.history]
    answer = await run_in_threadpool(answer_code_question, req.question, files, history)
    return {"answer": answer}
