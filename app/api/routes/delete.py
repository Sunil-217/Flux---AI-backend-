from fastapi import APIRouter
import os
import shutil

from app.services.chroma_service import delete_collection

router = APIRouter()

UPLOAD_DIR = "uploads"


@router.delete("/delete/{chat_id}")
async def delete_chat(chat_id: str):
    """
    Deletes everything associated with a chat session:
      1. The ChromaDB vector collection  (embeddings)
      2. The uploads folder              (PDF files)
    """

    try:

        # 1. Remove vectors from ChromaDB
        delete_collection(chat_id)

        # 2. Remove uploaded PDF files
        chat_folder = os.path.join(
            UPLOAD_DIR,
            chat_id
        )

        if os.path.exists(chat_folder):
            shutil.rmtree(chat_folder)

        return {
            "message": "Chat deleted successfully",
            "chat_id": chat_id
        }

    except Exception as e:

        return {
            "error": str(e)
        }
