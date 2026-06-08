import os
import re
import uuid

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Form,
    Depends,
    HTTPException,
)

from sqlalchemy.orm import Session

from app.core.security import get_current_user, user_owns_chat
from app.db import get_db
from app.models import User

from app.services.pdf_service import (
    extract_text_from_file
)

from app.services.embedding_service import (
    chunk_text,
    create_embeddings
)

from app.services.chroma_service import (
    get_or_create_collection,
    sanitize_chat_id,
)

from starlette.concurrency import run_in_threadpool

router = APIRouter()

UPLOAD_DIR = "uploads"

ALLOWED_EXTENSIONS = [
    ".pdf", ".docx", ".txt", ".md", ".csv", ".json",
    ".py", ".js", ".ts", ".tsx", ".jsx", ".html", ".css",
    ".java", ".c", ".cpp", ".h", ".go", ".rs", ".rb", ".php",
    ".sh", ".yaml", ".yml", ".xml", ".sql",
]

# Cap upload size so a huge file can't exhaust server memory/disk.
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB


def _safe_filename(name: str) -> str:
    """Strip any path components and unsafe characters — prevents path
    traversal (e.g. '../../etc/passwd') via the client-supplied filename."""
    base = os.path.basename(name or "")
    base = base.replace("\\", "_")  # guard Windows-style separators too
    base = re.sub(r"[^A-Za-z0-9._-]", "_", base).strip("._")
    return base or "upload.pdf"


def _embed_and_store(chat_id: str, safe_chat_id: str, filename: str, chunks: list) -> None:
    """Blocking CPU work (embeddings) + ChromaDB write. Run via a threadpool so
    it never blocks the async event loop (which would freeze the whole server)."""
    embeddings = create_embeddings(chunks)
    collection = get_or_create_collection(chat_id)
    # Unique per-upload prefix so a 2nd document's IDs don't collide with the
    # 1st's (which would make Chroma drop them — breaking multiple docs per chat).
    uid = uuid.uuid4().hex[:8]
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"{safe_chat_id}_{uid}_{i}" for i in range(len(chunks))],
        metadatas=[
            {"filename": filename, "chat_id": chat_id}
            for _ in range(len(chunks))
        ],
    )


async def _read_capped(file: UploadFile) -> bytes:
    """Read the upload in chunks, aborting if it exceeds the size cap so we
    never buffer an unbounded body in memory."""
    buf = bytearray()
    while True:
        chunk = await file.read(1024 * 1024)  # 1 MB at a time
        if not chunk:
            break
        buf.extend(chunk)
        if len(buf) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
            )
    return bytes(buf)


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    chat_id: str = Form(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not user_owns_chat(db, user, chat_id):
        raise HTTPException(status_code=404, detail="Chat not found")
    filename = _safe_filename(file.filename)

    # Validate file type
    file_extension = os.path.splitext(filename)[1].lower()
    if file_extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Allowed: PDF, Word (.docx), text, markdown, CSV, and code files.",
        )

    content = await _read_capped(file)
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    # Create chat folder (chat_id sanitized so it can't escape UPLOAD_DIR)
    safe_chat_id = sanitize_chat_id(chat_id)
    chat_folder = os.path.join(UPLOAD_DIR, safe_chat_id)
    os.makedirs(chat_folder, exist_ok=True)

    # Save file
    file_path = os.path.join(chat_folder, filename)
    with open(file_path, "wb") as f:
        f.write(content)

    # Extract text (offloaded — parsing/IO can be slow, especially under load)
    try:
        extracted_text = await run_in_threadpool(
            extract_text_from_file, file_path, file_extension
        )
    except Exception:
        raise HTTPException(
            status_code=422,
            detail="Could not read the file — it may be corrupted or password-protected.",
        )

    # Empty file check
    if not extracted_text.strip():
        raise HTTPException(
            status_code=422,
            detail="No readable text found in the file (scanned/image-only PDFs aren't supported).",
        )

    # Chunk text
    chunks = chunk_text(extracted_text)

    # Embed + store — offloaded to a thread so the CPU-heavy embedding does NOT
    # block the event loop (which would freeze chat/health for everyone).
    try:
        await run_in_threadpool(_embed_and_store, chat_id, safe_chat_id, filename, chunks)
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Failed to index the document. Please try again.",
        )

    return {
        "message": "File uploaded successfully",
        "chat_id": chat_id,
        "filename": filename,
        "total_chunks": len(chunks),
    }
