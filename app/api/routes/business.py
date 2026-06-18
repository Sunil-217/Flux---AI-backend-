"""Business tenant API.

/admin/business/* — admin-only: create / list / revoke tenants
/business/*       — business-key auth: portal (doc upload/list/delete) + public chat

Auth: every /business/* endpoint reads the raw key from the X-Business-Key header,
hashes it, and looks it up — same pattern as user ApiKeys but with bk_ prefix.
"""

import hashlib
import os
import re
import secrets
import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from app.core.security import require_admin
from app.db import get_db
from app.models import BusinessDocument, BusinessTenant
from app.services.chroma_service import (
    delete_business_collection,
    delete_business_document_chunks,
    get_or_create_business_collection,
)
from app.services.embedding_service import chunk_text, create_embeddings
from app.services.pdf_service import extract_text_from_file
from app.services.rag_service import ask_business_question

router = APIRouter()

UPLOAD_DIR = "uploads"
ALLOWED_EXTENSIONS = [
    ".pdf", ".docx", ".xlsx", ".pptx", ".txt", ".md", ".csv", ".json",
    ".py", ".js", ".ts", ".tsx", ".jsx", ".html", ".css",
    ".java", ".c", ".cpp", ".h", ".go", ".rs", ".rb", ".php",
    ".sh", ".yaml", ".yml", ".xml", ".sql",
]
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB


# ── Key helpers ───────────────────────────────────────────────────────────────

def _generate_business_key():
    """Return (raw_key, sha256_hash, display_prefix)."""
    raw = "bk_" + secrets.token_urlsafe(32)
    h = hashlib.sha256(raw.encode()).hexdigest()
    prefix = raw[:10] + "…" + raw[-4:]
    return raw, h, prefix


def _hash_key(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


def _require_business_key(
    x_business_key: Optional[str] = None,
    db: Session = Depends(get_db),
) -> BusinessTenant:
    raise HTTPException(status_code=401, detail="X-Business-Key header required.")


# FastAPI Header dependency — declared inline so the alias works correctly
from fastapi import Header as _Header


def _get_tenant(
    x_business_key: Optional[str] = _Header(None, alias="X-Business-Key"),
    db: Session = Depends(get_db),
) -> BusinessTenant:
    if not x_business_key:
        raise HTTPException(status_code=401, detail="X-Business-Key header required.")
    h = _hash_key(x_business_key)
    tenant = db.query(BusinessTenant).filter_by(api_key_hash=h).first()
    if tenant is None or tenant.revoked:
        raise HTTPException(status_code=401, detail="Invalid or revoked business key.")
    return tenant


# ── File helpers ──────────────────────────────────────────────────────────────

def _safe_filename(name: str) -> str:
    base = os.path.basename(name or "")
    base = base.replace("\\", "_")
    base = re.sub(r"[^A-Za-z0-9._-]", "_", base).strip("._")
    return base or "upload.pdf"


async def _read_capped(file: UploadFile) -> bytes:
    buf = bytearray()
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        buf.extend(chunk)
        if len(buf) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
            )
    return bytes(buf)


def _embed_and_store(collection_name: str, filename: str, upload_uid: str, chunks: list) -> None:
    embeddings = create_embeddings(chunks)
    collection = get_or_create_business_collection(collection_name)
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"{upload_uid}_{i}" for i in range(len(chunks))],
        metadatas=[{"filename": filename, "upload_uid": upload_uid} for _ in chunks],
    )


# ── Admin: tenant CRUD ────────────────────────────────────────────────────────

class CreateTenantRequest(BaseModel):
    business_name: str


@router.post("/admin/business")
def create_tenant(
    req: CreateTenantRequest,
    admin=Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Create a business tenant. The raw API key is returned ONCE — save it."""
    name = req.business_name.strip()[:120]
    if not name:
        raise HTTPException(status_code=400, detail="business_name is required.")

    raw, key_hash, prefix = _generate_business_key()
    collection_name = f"biz_{uuid.uuid4().hex[:16]}"

    tenant = BusinessTenant(
        business_name=name,
        api_key_hash=key_hash,
        api_key_prefix=prefix,
        collection_name=collection_name,
        created_by=admin.id,
    )
    db.add(tenant)
    db.commit()
    db.refresh(tenant)

    return {
        "id": tenant.id,
        "business_name": tenant.business_name,
        "api_key": raw,          # shown ONCE, never stored again
        "api_key_prefix": prefix,
        "created_at": tenant.created_at.isoformat(),
        "warning": "Save this key now — it cannot be retrieved again.",
    }


@router.get("/admin/business")
def list_tenants(
    admin=Depends(require_admin),
    db: Session = Depends(get_db),
):
    tenants = db.query(BusinessTenant).order_by(BusinessTenant.created_at.desc()).all()
    return [
        {
            "id": t.id,
            "business_name": t.business_name,
            "api_key_prefix": t.api_key_prefix,
            "doc_count": t.doc_count,
            "chat_count": t.chat_count,
            "revoked": t.revoked,
            "created_at": t.created_at.isoformat(),
        }
        for t in tenants
    ]


@router.delete("/admin/business/{tenant_id}")
def revoke_tenant(
    tenant_id: int,
    admin=Depends(require_admin),
    db: Session = Depends(get_db),
):
    tenant = db.get(BusinessTenant, tenant_id)
    if tenant is None:
        raise HTTPException(status_code=404, detail="Tenant not found.")
    tenant.revoked = True
    db.commit()
    return {"detail": "Tenant revoked."}


# ── Business portal: identity ─────────────────────────────────────────────────

@router.get("/business/me")
def business_me(
    tenant: BusinessTenant = Depends(_get_tenant),
    db: Session = Depends(get_db),
):
    docs = (
        db.query(BusinessDocument)
        .filter_by(tenant_id=tenant.id)
        .order_by(BusinessDocument.uploaded_at.desc())
        .all()
    )
    return {
        "id": tenant.id,
        "business_name": tenant.business_name,
        "doc_count": tenant.doc_count,
        "chat_count": tenant.chat_count,
        "documents": [
            {
                "id": d.id,
                "filename": d.filename,
                "file_size": d.file_size,
                "chunk_count": d.chunk_count,
                "uploaded_at": d.uploaded_at.isoformat(),
            }
            for d in docs
        ],
    }


# ── Business portal: document management ─────────────────────────────────────

@router.post("/business/documents")
async def upload_document(
    file: UploadFile = File(...),
    tenant: BusinessTenant = Depends(_get_tenant),
    db: Session = Depends(get_db),
):
    filename = _safe_filename(file.filename)
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    content = await _read_capped(file)
    if not content:
        raise HTTPException(status_code=400, detail="File is empty.")

    upload_uid = uuid.uuid4().hex[:12]
    folder = os.path.join(UPLOAD_DIR, f"biz_{tenant.id}")
    os.makedirs(folder, exist_ok=True)

    file_path = os.path.join(folder, f"{upload_uid}_{filename}")
    with open(file_path, "wb") as f:
        f.write(content)

    try:
        extracted = await run_in_threadpool(extract_text_from_file, file_path, ext)
    except Exception:
        raise HTTPException(status_code=422, detail="Could not read the file — it may be corrupt or password-protected.")

    if not extracted.strip():
        raise HTTPException(status_code=422, detail="No readable text found in the file.")

    chunks = chunk_text(extracted)

    try:
        await run_in_threadpool(_embed_and_store, tenant.collection_name, filename, upload_uid, chunks)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to index document.")

    doc = BusinessDocument(
        tenant_id=tenant.id,
        filename=filename,
        file_size=len(content),
        chunk_count=len(chunks),
        upload_uid=upload_uid,
    )
    db.add(doc)
    tenant.doc_count += 1
    db.commit()
    db.refresh(doc)

    return {
        "id": doc.id,
        "filename": filename,
        "chunk_count": len(chunks),
        "file_size": len(content),
        "message": "Document uploaded and indexed successfully.",
    }


@router.get("/business/documents")
def list_documents(
    tenant: BusinessTenant = Depends(_get_tenant),
    db: Session = Depends(get_db),
):
    docs = (
        db.query(BusinessDocument)
        .filter_by(tenant_id=tenant.id)
        .order_by(BusinessDocument.uploaded_at.desc())
        .all()
    )
    return [
        {
            "id": d.id,
            "filename": d.filename,
            "file_size": d.file_size,
            "chunk_count": d.chunk_count,
            "uploaded_at": d.uploaded_at.isoformat(),
        }
        for d in docs
    ]


@router.delete("/business/documents/{doc_id}")
def delete_document(
    doc_id: int,
    tenant: BusinessTenant = Depends(_get_tenant),
    db: Session = Depends(get_db),
):
    doc = db.get(BusinessDocument, doc_id)
    if doc is None or doc.tenant_id != tenant.id:
        raise HTTPException(status_code=404, detail="Document not found.")

    # Remove chunks from ChromaDB
    delete_business_document_chunks(tenant.collection_name, doc.upload_uid)

    # Remove file from disk
    file_path = os.path.join(UPLOAD_DIR, f"biz_{tenant.id}", f"{doc.upload_uid}_{doc.filename}")
    try:
        os.remove(file_path)
    except FileNotFoundError:
        pass

    db.delete(doc)
    if tenant.doc_count > 0:
        tenant.doc_count -= 1
    db.commit()

    return {"detail": "Document deleted."}


# ── Public chat (called from the embedded widget) ─────────────────────────────

class BusinessChatRequest(BaseModel):
    question: str
    history: list = []


@router.post("/business/chat")
def business_chat(
    req: BusinessChatRequest,
    tenant: BusinessTenant = Depends(_get_tenant),
    db: Session = Depends(get_db),
):
    """RAG chat against the tenant's knowledge base. CORS is open (see main.py)
    so this can be called from any origin (the embedded widget's host domain)."""
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required.")

    result = ask_business_question(tenant.collection_name, question, req.history or [])

    tenant.chat_count += 1
    db.commit()

    return {
        "answer": result.get("answer", ""),
        "sources": [
            {"content": s.get("content", ""), "filename": (s.get("metadata") or {}).get("filename", "")}
            for s in result.get("sources", [])
        ],
    }
