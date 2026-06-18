
import logging
import os

# Must be set BEFORE importing chromadb — some versions read this at import
# time when wiring up their telemetry client.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

# ChromaDB 0.4.24 calls posthog.capture() with 3 positional args, but
# posthog 7.x only accepts 1 → every telemetry event raises TypeError and
# ChromaDB logs "Failed to send telemetry event …: capture() takes 1
# positional argument but 3 were given". It's harmless noise from a known
# version mismatch. We don't want a self-hosted app phoning home anyway, so
# we both (a) disable telemetry via Settings and (b) hard-silence the
# telemetry logger so the broken call can't print even if it still fires.
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)

import chromadb
from chromadb.config import Settings
import re

client = chromadb.PersistentClient(
    path="chroma_db",
    settings=Settings(anonymized_telemetry=False),
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


def get_or_create_business_collection(collection_name: str):
    """Get/create an isolated ChromaDB collection for a business tenant."""
    return client.get_or_create_collection(name=collection_name)


def delete_business_document_chunks(collection_name: str, upload_uid: str) -> None:
    """Remove all chunks for one document upload from a business collection."""
    try:
        collection = client.get_collection(name=collection_name)
        collection.delete(where={"upload_uid": {"$eq": upload_uid}})
    except Exception:
        pass


def delete_business_collection(collection_name: str) -> None:
    """Delete an entire business tenant's ChromaDB collection."""
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass

