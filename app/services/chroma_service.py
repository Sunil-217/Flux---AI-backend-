
import chromadb
import re

client = chromadb.PersistentClient(
    path="chroma_db"
)


def sanitize_chat_id(chat_id: str):

    sanitized = re.sub(
        r"[^a-zA-Z0-9_-]",
        "_",
        chat_id
    )

    return sanitized


def get_or_create_collection(
    chat_id: str
):

    safe_chat_id = sanitize_chat_id(
        chat_id
    )

    collection_name = (
        f"chat_{safe_chat_id}"
    )

    collection = (
        client.get_or_create_collection(
            name=collection_name
        )
    )

    return collection


def delete_collection(
    chat_id: str
):
    """Delete the ChromaDB collection for a given chat_id."""

    safe_chat_id = sanitize_chat_id(
        chat_id
    )

    collection_name = (
        f"chat_{safe_chat_id}"
    )

    try:
        client.delete_collection(
            name=collection_name
        )
    except Exception:
        # Collection may not exist — safe to ignore
        pass

