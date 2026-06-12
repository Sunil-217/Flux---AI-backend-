"""Memory across chats — durable user facts extracted from conversations.

POST /memory/extract  → pull new facts from a Q/A exchange and merge them
GET  /memory          → list remembered facts
DELETE /memory        → forget everything
DELETE /memory/{i}    → forget one fact by index
"""

import json
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from app.core.security import get_current_user
from app.db import get_db
from app.models import User, UserMemory
from app.services.rag_service import _groq_or_nvidia, MODEL

router = APIRouter()

_MAX_FACTS = 40

_EXTRACT_SYSTEM = (
    "You extract durable facts about the USER from a chat exchange. Output ONLY a "
    "JSON array of short strings — facts about the user THEMSELVES that are worth "
    "remembering across future conversations: their preferences, role/occupation, "
    "studies, projects they're building, tools they use, and the language/style "
    "they like to talk in.\n"
    "Rules:\n"
    "- Each fact is one short third-person sentence, e.g. \"Is a final year student\", "
    "\"Prefers replies in Tamil/Tanglish\", \"Building a RAG chatbot app\".\n"
    "- ONLY include things stated or clearly implied by the USER about themselves. "
    "Never include facts about the world, the assistant, or one-off task details.\n"
    "- If there is nothing durable worth remembering, output exactly: []\n"
    "- Output the JSON array only — no prose, no markdown fences."
)


class ExtractRequest(BaseModel):
    question: str
    answer: str


def _load_facts(db: Session, user_id: int) -> list:
    rec = db.get(UserMemory, user_id)
    if rec is None:
        return []
    try:
        facts = json.loads(rec.facts or "[]")
    except Exception:
        return []
    return [str(f).strip() for f in facts if str(f).strip()] if isinstance(facts, list) else []


def _save_facts(db: Session, user_id: int, facts: list) -> None:
    rec = db.get(UserMemory, user_id)
    if rec is None:
        rec = UserMemory(user_id=user_id, facts=json.dumps(facts), updated_at=datetime.utcnow())
        db.add(rec)
    else:
        rec.facts = json.dumps(facts)
        rec.updated_at = datetime.utcnow()
    db.commit()


def _extract_facts(question: str, answer: str) -> list:
    """LLM call: extract durable user facts from one exchange. [] on failure."""
    try:
        resp = _groq_or_nvidia().chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": _EXTRACT_SYSTEM},
                {
                    "role": "user",
                    "content": f"User: {(question or '')[:1500]}\n\nAssistant: {(answer or '')[:1500]}",
                },
            ],
            temperature=0.1,
            max_tokens=300,
        )
        raw = (resp.choices[0].message.content or "").strip()
        start, end = raw.find("["), raw.rfind("]")
        if start == -1 or end == -1:
            return []
        data = json.loads(raw[start : end + 1])
        if not isinstance(data, list):
            return []
        return [str(f).strip()[:200] for f in data if str(f).strip()][:10]
    except Exception:
        return []


@router.post("/memory/extract")
async def extract_memory(
    req: ExtractRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    new_facts = await run_in_threadpool(_extract_facts, req.question, req.answer)

    facts = _load_facts(db, user.id)
    if new_facts:
        # Merge: dedupe case-insensitively, keep existing order, cap at _MAX_FACTS.
        seen = {f.lower() for f in facts}
        for f in new_facts:
            if f.lower() not in seen:
                seen.add(f.lower())
                facts.append(f)
        facts = facts[-_MAX_FACTS:]
        _save_facts(db, user.id, facts)

    return {"facts": facts}


@router.get("/memory")
async def get_memory(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return {"facts": _load_facts(db, user.id)}


@router.delete("/memory")
async def clear_memory(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _save_facts(db, user.id, [])
    return {"ok": True}


@router.delete("/memory/{index}")
async def delete_memory_fact(
    index: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    facts = _load_facts(db, user.id)
    if index < 0 or index >= len(facts):
        raise HTTPException(status_code=404, detail="Fact not found")
    facts.pop(index)
    _save_facts(db, user.id, facts)
    return {"facts": facts}
