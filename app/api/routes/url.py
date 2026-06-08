import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from app.core.security import get_current_user, user_owns_chat
from app.db import get_db
from app.models import User
from app.services.url_service import fetch_url_text
from app.services.embedding_service import chunk_text, create_embeddings
from app.services.chroma_service import get_or_create_collection, sanitize_chat_id

router = APIRouter()


class UrlRequest(BaseModel):
    chat_id: str
    url: str


def _ingest(chat_id: str, safe_id: str, source: str, chunks: list):
    embeddings = create_embeddings(chunks)
    collection = get_or_create_collection(chat_id)
    uid = uuid.uuid4().hex[:8]  # unique per ingest so IDs never collide
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"{safe_id}_{uid}_{i}" for i in range(len(chunks))],
        metadatas=[{"filename": source, "chat_id": chat_id} for _ in range(len(chunks))],
    )


@router.post("/upload-url")
async def upload_url(
    req: UrlRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Fetch a web page, index it, and let the chat answer questions about it."""
    if not user_owns_chat(db, user, req.chat_id):
        raise HTTPException(status_code=404, detail="Chat not found")
    try:
        title, text = await run_in_threadpool(fetch_url_text, req.url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        raise HTTPException(status_code=400, detail="Couldn't fetch that URL. Check the link and try again.")

    if not text.strip():
        raise HTTPException(
            status_code=422,
            detail=(
                "This page loads its content with JavaScript (common on sites like Naukri, "
                "LinkedIn, Instagram, X), so there's no readable text to index. Try a direct "
                "article, blog post, documentation, or Wikipedia URL instead."
            ),
        )

    chunks = chunk_text(text)
    try:
        await run_in_threadpool(_ingest, req.chat_id, sanitize_chat_id(req.chat_id), title, chunks)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to index the page. Please try again.")

    return {"message": "Page added", "chat_id": req.chat_id, "source": title}
