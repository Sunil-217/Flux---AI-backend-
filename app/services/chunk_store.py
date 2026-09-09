"""Durable copy of every chat's indexed chunks, and the rebuild that uses it.

ChromaDB's persistent client writes to a local directory. On the hosting tier
that directory does not survive a deploy or a spin-down, so without this module
every uploaded document silently vanished on restart — retrieval found an empty
collection and the model answered "I don't have the document" while the chat
still listed it. Measured on production: the same PDF re-uploaded minutes after
a deploy was embedded from scratch instead of being recognised as already
indexed.

Two operations:

  save_chunks      — at index time, mirror the chunks + embeddings to Postgres.
  hydrate          — when a collection is empty, rebuild it from the mirror.

Hydration runs through get_or_create_collection, so every consumer (chat, the
upload route's reuse check, quiz, URL ingest, agent tools) sees a rebuilt
collection without knowing it happened. An in-process memo keeps a chat that is
genuinely empty from paying a database round trip on every message.

Everything here is best-effort: a database hiccup must never turn into a failed
upload or a failed answer. The worst case is the old behaviour.
"""

import json
import logging
import threading
import time

from app.db import SessionLocal
from app.models import ChatChunk

log = logging.getLogger(__name__)

# Chats checked recently, so a chat with no documents is not re-queried on every
# message. Cleared for a chat whenever its chunks change.
_CHECKED_TTL_SECONDS = 600
_checked: dict[str, float] = {}
_checked_lock = threading.Lock()


def _round(vec) -> list:
    # Six decimals is well inside embedding noise and roughly halves the row.
    return [round(float(x), 6) for x in vec]


def forget(chat_id: str) -> None:
    """Drop the memo for a chat so the next collection access re-checks."""
    with _checked_lock:
        _checked.pop(chat_id, None)


def save_chunks(chat_id: str, filename: str, content_hash: str, ids: list, chunks: list, embeddings: list) -> None:
    """Mirror one document's chunks. Replaces any earlier rows for the same
    filename in this chat, matching the upload route's replace semantics."""
    db = SessionLocal()
    try:
        db.query(ChatChunk).filter(
            ChatChunk.chat_id == chat_id, ChatChunk.filename == filename
        ).delete(synchronize_session=False)
        db.add_all(
            ChatChunk(
                chat_id=chat_id,
                chunk_id=cid,
                filename=filename,
                content_hash=content_hash,
                position=i,
                text=text,
                embedding=json.dumps(_round(emb)),
            )
            for i, (cid, text, emb) in enumerate(zip(ids, chunks, embeddings))
        )
        db.commit()
    except Exception:
        db.rollback()
        log.warning("chunk mirror: save failed for chat %s (%s)", chat_id, filename, exc_info=True)
    finally:
        db.close()
    forget(chat_id)


def delete_chunks(chat_id: str, filename: str = None) -> None:
    """Remove the mirror for a chat, or for one document in it."""
    db = SessionLocal()
    try:
        q = db.query(ChatChunk).filter(ChatChunk.chat_id == chat_id)
        if filename is not None:
            q = q.filter(ChatChunk.filename == filename)
        q.delete(synchronize_session=False)
        db.commit()
    except Exception:
        db.rollback()
        log.warning("chunk mirror: delete failed for chat %s", chat_id, exc_info=True)
    finally:
        db.close()
    forget(chat_id)


def hydrate(chat_id: str, collection) -> int:
    """Rebuild an empty collection from the mirror. Returns rows restored.

    A non-empty collection is left alone: it is the live store and the mirror
    follows it, not the other way round."""
    now = time.monotonic()
    with _checked_lock:
        last = _checked.get(chat_id)
        if last is not None and now - last < _CHECKED_TTL_SECONDS:
            return 0
        _checked[chat_id] = now

    try:
        if collection.count() > 0:
            return 0
    except Exception:
        return 0

    db = SessionLocal()
    try:
        rows = (
            db.query(ChatChunk)
            .filter(ChatChunk.chat_id == chat_id)
            .order_by(ChatChunk.filename, ChatChunk.position)
            .all()
        )
    except Exception:
        log.warning("chunk mirror: read failed for chat %s", chat_id, exc_info=True)
        rows = []
    finally:
        db.close()

    if not rows:
        return 0

    try:
        collection.add(
            documents=[r.text for r in rows],
            embeddings=[json.loads(r.embedding) for r in rows],
            ids=[r.chunk_id for r in rows],
            metadatas=[
                {"filename": r.filename, "chat_id": r.chat_id, "content_hash": r.content_hash}
                for r in rows
            ],
        )
    except Exception:
        log.warning("chunk mirror: rebuild failed for chat %s", chat_id, exc_info=True)
        return 0

    log.info("chunk mirror: rebuilt %d chunk(s) for chat %s after a restart", len(rows), chat_id)
    return len(rows)
