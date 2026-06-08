from fastapi import APIRouter, Depends, HTTPException
import os
import shutil

from app.core.security import require_chat_ownership
from app.models import User
from app.services.chroma_service import delete_collection, sanitize_chat_id
from app.services.rag_service import delete_web_context

router = APIRouter()

UPLOAD_DIR = "uploads"


@router.delete("/delete/{chat_id}")
async def delete_chat(chat_id: str, user: User = Depends(require_chat_ownership)):
    """
    Deletes everything associated with a chat session:
      1. The ChromaDB vector collection  (embeddings)
      2. The uploads folder              (PDF files)
      3. The cached live web-search context
    """

    try:

        # 1. Remove vectors from ChromaDB
        delete_collection(chat_id)

        # 2. Remove uploaded PDF files (sanitized id matches the upload path)
        chat_folder = os.path.join(UPLOAD_DIR, sanitize_chat_id(chat_id))

        if os.path.exists(chat_folder):
            shutil.rmtree(chat_folder)

        # 3. Remove cached web-search context for this chat
        delete_web_context(chat_id)

        return {
            "message": "Chat deleted successfully",
            "chat_id": chat_id
        }

    except Exception:

        raise HTTPException(
            status_code=500,
            detail="Failed to delete chat data on the server.",
        )
