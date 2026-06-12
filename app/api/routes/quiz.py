"""Quiz generator — multiple-choice questions from a chat's documents or raw text."""

import json
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from app.core.security import get_current_user
from app.models import User
from app.services.chroma_service import get_or_create_collection
from app.services.rag_service import _groq_or_nvidia, MODEL

router = APIRouter()

_MAX_CONTENT_CHARS = 6000

_QUIZ_SYSTEM = (
    "You are a quiz generator. Given study material and a question count N, output "
    "ONLY a JSON array of exactly N multiple-choice questions testing understanding "
    "of the material.\n"
    "Each item is an object with exactly these keys:\n"
    "  \"q\": the question (string),\n"
    "  \"options\": an array of exactly 4 plausible answer strings,\n"
    "  \"answer\": the 0-based index (0-3) of the correct option,\n"
    "  \"explanation\": one short sentence explaining why the answer is correct.\n"
    "Rules: questions must be answerable from the material alone, options must be "
    "distinct, exactly one correct. Vary which index is correct. "
    "Output ONLY the JSON array — no prose, no markdown fences."
)


class QuizRequest(BaseModel):
    chat_id: Optional[str] = None
    content: Optional[str] = None
    count: int = 5


def _sample_collection_text(chat_id: str) -> str:
    """Pull up to ~6000 chars of document text from the chat's Chroma collection."""
    try:
        collection = get_or_create_collection(chat_id)
        if (collection.count() or 0) == 0:
            return ""
        results = collection.get(limit=20, include=["documents"])
        documents = results.get("documents") or []
        text = ""
        for doc in documents:
            if not doc:
                continue
            text += doc + "\n\n"
            if len(text) >= _MAX_CONTENT_CHARS:
                break
        return text[:_MAX_CONTENT_CHARS]
    except Exception:
        return ""


def _valid_question(item) -> Optional[dict]:
    """Validate one quiz item; return a normalized dict or None."""
    if not isinstance(item, dict):
        return None
    q = str(item.get("q") or item.get("question") or "").strip()
    options = item.get("options")
    if not q or not isinstance(options, list) or len(options) != 4:
        return None
    options = [str(o).strip() for o in options]
    if any(not o for o in options):
        return None
    try:
        answer = int(item.get("answer"))
    except (TypeError, ValueError):
        return None
    if not 0 <= answer <= 3:
        return None
    explanation = str(item.get("explanation") or "").strip()
    return {"q": q, "options": options, "answer": answer, "explanation": explanation}


def generate_quiz(content: str, count: int) -> list:
    """Generate `count` validated MCQs from `content`. [] on hard failure."""
    count = max(3, min(int(count or 5), 10))
    content = (content or "").strip()[:_MAX_CONTENT_CHARS]
    if not content:
        return []
    try:
        resp = _groq_or_nvidia().chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": _QUIZ_SYSTEM},
                {
                    "role": "user",
                    "content": f"Generate exactly {count} questions.\n\nMATERIAL:\n{content}",
                },
            ],
            temperature=0.4,
            max_tokens=2500,
        )
        raw = (resp.choices[0].message.content or "").strip()
        start, end = raw.find("["), raw.rfind("]")
        if start == -1 or end == -1:
            return []
        data = json.loads(raw[start : end + 1])
        if not isinstance(data, list):
            return []
        questions = []
        for item in data:
            valid = _valid_question(item)
            if valid:
                questions.append(valid)
        return questions[:count]
    except Exception:
        return []


@router.post("/quiz")
async def quiz(req: QuizRequest, user: User = Depends(get_current_user)):
    content = ""
    if req.chat_id:
        content = await run_in_threadpool(_sample_collection_text, req.chat_id)
    if not content:
        content = (req.content or "").strip()[:_MAX_CONTENT_CHARS]
    if not content:
        raise HTTPException(
            status_code=400,
            detail="Nothing to quiz on — upload a document to this chat or provide content.",
        )

    questions = await run_in_threadpool(generate_quiz, content, req.count)
    if not questions:
        raise HTTPException(status_code=502, detail="Couldn't generate a quiz. Please try again.")
    return {"questions": questions}
