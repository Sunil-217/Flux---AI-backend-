"""Knowledge-base RAG apps — built on the existing developer API keys (ck_).

Each ApiKey is one "app/project" with an isolated knowledge base
(ChromaDB collection kb_<api_key_id>). Two access paths:

  Owner (JWT) — manages the app's docs from the developer console:
    GET    /api-keys/{key_id}/kb                  — KB info + plan + docs
    POST   /api-keys/{key_id}/documents           — upload + index a doc
    DELETE /api-keys/{key_id}/documents/{doc_id}  — delete a doc

  End user (widget/secret token) — the embedded chat asks questions:
    POST   /v1/rag/chat                           — RAG over the app's docs

  Public:
    GET    /plans                                 — plan tiers for pricing UI

Auth for /v1/rag/chat accepts either the public widget token
(X-Widget-Token: wk_...) or the secret key (Authorization: Bearer ck_...).
Plans are display-only for now (no payment) — Free allows 1 doc.
"""

import base64
import json
import os
import re
import uuid
from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Header, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.config import ADMIN_EMAILS
from app.services.email_service import send_announcement_email
from starlette.concurrency import run_in_threadpool

from app.api.routes.apikeys import hash_key
from app.core.security import get_current_user
from app.db import get_db
from app.models import ApiKey, KnowledgeDocument, User, WidgetMessage, WidgetLead
from app.services.chroma_service import (
    delete_kb_collection,
    delete_kb_document_chunks,
    get_or_create_kb_collection,
)
from app.services.embedding_service import chunk_text, create_embeddings
from app.services.pdf_service import extract_text_from_file
from app.services.rag_service import ask_kb_question, suggest_starter_questions
from app.services import plan_service

router = APIRouter()

UPLOAD_DIR = "uploads"
ALLOWED_EXTENSIONS = [
    ".pdf", ".docx", ".xlsx", ".pptx", ".txt", ".md", ".csv", ".json",
    ".py", ".js", ".ts", ".tsx", ".jsx", ".html", ".css",
    ".java", ".c", ".cpp", ".h", ".go", ".rs", ".rb", ".php",
    ".sh", ".yaml", ".yml", ".xml", ".sql",
]
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB

# Plan tiers now live in the DB (admin-editable) — see app/services/plan_service.py.
# doc_limit governs how many documents an app's knowledge base may hold and is
# enforced on upload below.


def _collection_name(key_id: int) -> str:
    return f"kb_{key_id}"


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
    collection = get_or_create_kb_collection(collection_name)
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"{upload_uid}_{i}" for i in range(len(chunks))],
        metadatas=[{"filename": filename, "upload_uid": upload_uid} for _ in chunks],
    )


# ── Owner-side (JWT) helpers ──────────────────────────────────────────────────

def _owned_key(key_id: int, user: User, db: Session) -> ApiKey:
    rec = db.get(ApiKey, key_id)
    if rec is None or rec.user_id != user.id:
        raise HTTPException(status_code=404, detail="App not found.")
    return rec


def _ensure_widget_token(rec: ApiKey, db: Session) -> str:
    """Backfill a widget token for keys created before the feature existed."""
    if not rec.widget_token:
        from app.api.routes.apikeys import gen_widget_token
        rec.widget_token = gen_widget_token()
        db.commit()
    return rec.widget_token


def _doc_row(d: KnowledgeDocument) -> dict:
    return {
        "id": d.id,
        "filename": d.filename,
        "file_size": d.file_size,
        "chunk_count": d.chunk_count,
        "uploaded_at": d.uploaded_at.isoformat() if d.uploaded_at else None,
    }


# ── Plans (public) ────────────────────────────────────────────────────────────

@router.get("/plans")
def list_plans(db: Session = Depends(get_db)):
    return {"plans": plan_service.get_plans(db)}


# ── Widget appearance config (owner edits instant fields; embed page reads) ────
# Custom CSS is special: it is NOT applied live on save. It goes through a
# super-admin review (cssStatus pending→approved/rejected) so arbitrary CSS can't
# hit our embed origin unreviewed. Only `customCss` (the approved copy) is ever
# served publicly; `customCssPending` is the under-review submission.
_WIDGET_TEXT_LIMITS = {"title": 60, "subtitle": 40, "greeting": 80, "tagline": 160, "theme": 30}
_HEX_RE = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")
# Instant (no-review) fields the owner may edit directly.
_PUBLIC_FIELDS = ("title", "subtitle", "greeting", "tagline", "accent", "theme", "suggestions", "logoUrl", "customCss", "leadCapture", "leadPrompt")


def _clean_css(css: str) -> str:
    """CSS only — strip anything that could close the <style> tag or smuggle
    markup/script onto our embed origin; clamp size."""
    cleaned = re.sub(r"</\s*style", "", css or "", flags=re.I)
    cleaned = re.sub(r"<\s*script", "", cleaned, flags=re.I)
    return cleaned[:4000]


def _sanitize_widget_config(raw: dict) -> dict:
    """Validate the INSTANT (non-CSS) appearance fields the owner edits directly.
    Custom CSS is handled separately via the review endpoint, never here."""
    if not isinstance(raw, dict):
        return {}
    out: dict = {}
    for field, limit in _WIDGET_TEXT_LIMITS.items():
        if raw.get(field) is not None:
            out[field] = str(raw[field])[:limit]
    accent = raw.get("accent")
    if isinstance(accent, str) and _HEX_RE.match(accent.strip()):
        out["accent"] = accent.strip()
    logo = raw.get("logoUrl")
    if isinstance(logo, str):
        # Large cap so uploaded data: URIs (base64) survive a config save.
        logo = logo.strip()[:400000]
        # Only allow https images or inline data: URIs (no javascript:/http:).
        if logo == "" or logo.startswith("https://") or logo.startswith("data:image/"):
            out["logoUrl"] = logo
    sugg = raw.get("suggestions")
    if isinstance(sugg, list):
        out["suggestions"] = [str(s)[:80] for s in sugg if str(s).strip()][:6]
    if "leadCapture" in raw:
        out["leadCapture"] = bool(raw["leadCapture"])
    if raw.get("leadPrompt") is not None:
        out["leadPrompt"] = str(raw["leadPrompt"])[:160]
    return out


def _read_widget_config(rec: ApiKey) -> dict:
    try:
        cfg = json.loads(rec.widget_config) if rec.widget_config else {}
        return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return {}


def _public_view(cfg: dict, rec: "ApiKey") -> dict:
    """The render-only view served to the embedded widget — approved CSS only,
    never the pending submission. Branding is always on (handled client-side).
    The header title defaults to the app's own name when the dev hasn't set one."""
    out = {k: cfg[k] for k in _PUBLIC_FIELDS if k in cfg}
    if not (out.get("title") or "").strip():
        out["title"] = rec.name
    return out


class WidgetConfigPayload(BaseModel):
    config: dict


class WidgetCssPayload(BaseModel):
    css: str = ""


@router.get("/api-keys/{key_id}/widget")
def get_widget_config(key_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    # Owner sees everything incl. the review state (pending CSS, status, note).
    rec = _owned_key(key_id, user, db)
    return {"config": _read_widget_config(rec)}


@router.put("/api-keys/{key_id}/widget")
def save_widget_config(
    key_id: int,
    payload: WidgetConfigPayload,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Save the instant appearance fields. Merges over existing config so the
    CSS review state (managed elsewhere) is preserved."""
    rec = _owned_key(key_id, user, db)
    existing = _read_widget_config(rec)
    clean = _sanitize_widget_config(payload.config or {})
    merged = {**existing, **clean}
    rec.widget_config = json.dumps(merged)
    db.commit()
    return {"config": merged}


@router.post("/api-keys/{key_id}/widget/css")
def submit_widget_css(
    key_id: int,
    payload: WidgetCssPayload,
    background: BackgroundTasks,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Submit custom CSS for super-admin review. Clearing it (empty) is applied
    immediately; otherwise it is held as 'pending' until an admin approves."""
    rec = _owned_key(key_id, user, db)
    cfg = _read_widget_config(rec)
    css = _clean_css(payload.css or "")
    if not css.strip():
        # Removing styling is always safe — apply right away.
        cfg["customCss"] = ""
        cfg["customCssPending"] = ""
        cfg["cssStatus"] = "none"
        cfg["cssNote"] = ""
    else:
        cfg["customCssPending"] = css
        cfg["cssStatus"] = "pending"
        cfg["cssNote"] = ""
        cfg["cssSubmittedAt"] = datetime.utcnow().isoformat()
        # Notify the super-admin(s) there's something to review (non-blocking).
        subject = "New widget CSS pending review"
        body = (
            f"{user.email} submitted custom CSS for the app “{rec.name}” ({rec.prefix}).\n\n"
            "Review and approve or reject it in Admin → Code Reviews."
        )
        for admin_email in ADMIN_EMAILS:
            background.add_task(send_announcement_email, admin_email, subject, body)
    rec.widget_config = json.dumps(cfg)
    db.commit()
    return {"config": cfg}


@router.get("/v1/rag/config")
def public_widget_config(app: str = "", db: Session = Depends(get_db)):
    """Public appearance config for the embedded widget, resolved by widget token.
    Returns empty (→ frontend defaults) for an unknown or revoked token."""
    if not app:
        return {"config": {}}
    rec = db.query(ApiKey).filter(ApiKey.widget_token == app).first()
    if rec is None or rec.revoked:
        return {"config": {}}
    return {"config": _public_view(_read_widget_config(rec), rec)}


# ── Widget logo upload (owner) ────────────────────────────────────────────────
_LOGO_TYPES = {"image/png", "image/jpeg", "image/svg+xml", "image/webp", "image/gif"}


@router.post("/api-keys/{key_id}/widget/logo")
async def upload_widget_logo(
    key_id: int,
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Upload a logo image; stored inline as a base64 data: URI in the config so
    it serves cross-origin to the embedded widget without separate file hosting."""
    rec = _owned_key(key_id, user, db)
    ct = (file.content_type or "").lower()
    if ct not in _LOGO_TYPES:
        raise HTTPException(400, "Logo must be a PNG, JPEG, SVG, WebP, or GIF image.")
    data = await file.read()
    if len(data) > 150 * 1024:
        raise HTTPException(400, "Logo must be under 150 KB.")
    data_uri = f"data:{ct};base64,{base64.b64encode(data).decode('ascii')}"
    cfg = _read_widget_config(rec)
    cfg["logoUrl"] = data_uri
    rec.widget_config = json.dumps(cfg)
    db.commit()
    return {"logoUrl": data_uri}


# ── Widget insights: analytics + conversations + leads (owner) ────────────────
@router.get("/api-keys/{key_id}/analytics")
def widget_analytics(
    key_id: int,
    days: int = 30,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    rec = _owned_key(key_id, user, db)
    days = max(1, min(days, 90))
    since = datetime.utcnow() - timedelta(days=days)
    user_msgs = db.query(WidgetMessage).filter(
        WidgetMessage.api_key_id == rec.id, WidgetMessage.role == "user"
    )
    total = user_msgs.count()
    sessions = (
        db.query(func.count(func.distinct(WidgetMessage.session_id)))
        .filter(WidgetMessage.api_key_id == rec.id).scalar() or 0
    )
    leads = db.query(func.count(WidgetLead.id)).filter(WidgetLead.api_key_id == rec.id).scalar() or 0
    recent = user_msgs.filter(WidgetMessage.created_at >= since).order_by(WidgetMessage.created_at.asc()).limit(5000).all()

    by_day: dict = {}
    counts: dict = {}
    for m in recent:
        d = m.created_at.date().isoformat() if m.created_at else "?"
        by_day[d] = by_day.get(d, 0) + 1
        k = " ".join(m.content.strip().lower().split())[:120]
        if k:
            counts[k] = counts.get(k, 0) + 1
    top = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:8]

    # Helpfulness from 👍/👎 on assistant answers.
    bot = lambda: db.query(func.count(WidgetMessage.id)).filter(  # noqa: E731
        WidgetMessage.api_key_id == rec.id, WidgetMessage.role == "assistant"
    )
    helpful = bot().filter(WidgetMessage.feedback == 1).scalar() or 0
    unhelpful = bot().filter(WidgetMessage.feedback == -1).scalar() or 0
    unanswered_total = bot().filter(WidgetMessage.answered == False).scalar() or 0  # noqa: E712

    # Content gaps: the visitor questions whose answer came back unanswered.
    gap_bots = (
        db.query(WidgetMessage)
        .filter(WidgetMessage.api_key_id == rec.id, WidgetMessage.role == "assistant", WidgetMessage.answered == False)  # noqa: E712
        .order_by(WidgetMessage.created_at.desc())
        .limit(40)
        .all()
    )
    gap_counts: dict = {}
    for a in gap_bots:
        u = (
            db.query(WidgetMessage)
            .filter(WidgetMessage.api_key_id == rec.id, WidgetMessage.session_id == a.session_id,
                    WidgetMessage.role == "user", WidgetMessage.created_at <= a.created_at)
            .order_by(WidgetMessage.created_at.desc())
            .first()
        )
        if u:
            k = " ".join(u.content.strip().lower().split())[:120]
            if k:
                gap_counts[k] = gap_counts.get(k, 0) + 1
    gaps = sorted(gap_counts.items(), key=lambda x: x[1], reverse=True)[:8]

    return {
        "total_questions": total,
        "window_questions": len(recent),
        "conversations": sessions,
        "leads": leads,
        "days": days,
        "helpful": helpful,
        "unhelpful": unhelpful,
        "unanswered": unanswered_total,
        "by_day": [{"date": d, "count": c} for d, c in sorted(by_day.items())],
        "top_questions": [{"question": q, "count": c} for q, c in top],
        "content_gaps": [{"question": q, "count": c} for q, c in gaps],
    }


@router.post("/api-keys/{key_id}/suggest-questions")
def suggest_questions(
    key_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """LLM-suggest a few starter questions from the app's uploaded documents."""
    rec = _owned_key(key_id, user, db)
    return {"questions": suggest_starter_questions(_collection_name(rec.id))}


@router.get("/api-keys/{key_id}/conversations")
def widget_conversations(
    key_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    rec = _owned_key(key_id, user, db)
    rows = (
        db.query(
            WidgetMessage.session_id,
            func.count(WidgetMessage.id),
            func.min(WidgetMessage.created_at),
            func.max(WidgetMessage.created_at),
        )
        .filter(WidgetMessage.api_key_id == rec.id)
        .group_by(WidgetMessage.session_id)
        .order_by(func.max(WidgetMessage.created_at).desc())
        .limit(60)
        .all()
    )
    out = []
    for sid, cnt, first, last in rows:
        fu = (
            db.query(WidgetMessage.content)
            .filter(WidgetMessage.api_key_id == rec.id, WidgetMessage.session_id == sid, WidgetMessage.role == "user")
            .order_by(WidgetMessage.created_at.asc())
            .first()
        )
        out.append({
            "session_id": sid,
            "messages": cnt,
            "started_at": first.isoformat() if first else None,
            "last_at": last.isoformat() if last else None,
            "preview": (fu[0] if fu else "")[:90],
        })
    return {"conversations": out}


@router.get("/api-keys/{key_id}/conversations/{session_id}")
def widget_transcript(
    key_id: int,
    session_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    rec = _owned_key(key_id, user, db)
    msgs = (
        db.query(WidgetMessage)
        .filter(WidgetMessage.api_key_id == rec.id, WidgetMessage.session_id == session_id)
        .order_by(WidgetMessage.created_at.asc())
        .all()
    )
    return {
        "session_id": session_id,
        "messages": [
            {"role": m.role, "content": m.content, "at": m.created_at.isoformat() if m.created_at else None}
            for m in msgs
        ],
    }


@router.get("/api-keys/{key_id}/leads")
def widget_leads(
    key_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    rec = _owned_key(key_id, user, db)
    rows = (
        db.query(WidgetLead)
        .filter(WidgetLead.api_key_id == rec.id)
        .order_by(WidgetLead.created_at.desc())
        .limit(500)
        .all()
    )
    return {
        "leads": [
            {"id": x.id, "name": x.name, "email": x.email, "message": x.message,
             "at": x.created_at.isoformat() if x.created_at else None}
            for x in rows
        ]
    }


# ── KB info + documents (owner, JWT) ──────────────────────────────────────────

@router.get("/api-keys/{key_id}/kb")
def get_kb(
    key_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    rec = _owned_key(key_id, user, db)
    widget_token = _ensure_widget_token(rec, db)
    docs = (
        db.query(KnowledgeDocument)
        .filter_by(api_key_id=rec.id)
        .order_by(KnowledgeDocument.uploaded_at.desc())
        .all()
    )
    plan = getattr(rec, "plan", None) or "free"
    return {
        "key_id": rec.id,
        "name": rec.name,
        "plan": plan,
        "doc_limit": plan_service.doc_limit_for(db, plan),
        "doc_count": len(docs),
        "widget_token": widget_token,
        "documents": [_doc_row(d) for d in docs],
    }


@router.post("/api-keys/{key_id}/documents")
async def upload_kb_document(
    key_id: int,
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    rec = _owned_key(key_id, user, db)
    plan = getattr(rec, "plan", None) or "free"
    limit = plan_service.doc_limit_for(db, plan)

    current = db.query(KnowledgeDocument).filter_by(api_key_id=rec.id).count()
    if current >= limit:
        plan_label = plan_service.plan_label(db, plan)
        raise HTTPException(
            status_code=403,
            detail=(
                f"{plan_label} plan allows {limit} document{'s' if limit != 1 else ''}. "
                "Upgrade your plan to add more."
            ),
        )

    filename = _safe_filename(file.filename)
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    content = await _read_capped(file)
    if not content:
        raise HTTPException(status_code=400, detail="File is empty.")

    upload_uid = uuid.uuid4().hex[:12]
    folder = os.path.join(UPLOAD_DIR, _collection_name(rec.id))
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
        await run_in_threadpool(_embed_and_store, _collection_name(rec.id), filename, upload_uid, chunks)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to index document.")

    doc = KnowledgeDocument(
        api_key_id=rec.id,
        filename=filename,
        file_size=len(content),
        chunk_count=len(chunks),
        upload_uid=upload_uid,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    return {**_doc_row(doc), "message": "Document uploaded and indexed."}


@router.delete("/api-keys/{key_id}/documents/{doc_id}")
def delete_kb_document(
    key_id: int,
    doc_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    rec = _owned_key(key_id, user, db)
    doc = db.get(KnowledgeDocument, doc_id)
    if doc is None or doc.api_key_id != rec.id:
        raise HTTPException(status_code=404, detail="Document not found.")

    delete_kb_document_chunks(_collection_name(rec.id), doc.upload_uid)
    file_path = os.path.join(UPLOAD_DIR, _collection_name(rec.id), f"{doc.upload_uid}_{doc.filename}")
    try:
        os.remove(file_path)
    except FileNotFoundError:
        pass

    db.delete(doc)
    db.commit()
    return {"detail": "Document deleted."}


# ── RAG chat (end user — widget token or secret key) ──────────────────────────

def _resolve_app(
    x_widget_token: Optional[str] = Header(None, alias="X-Widget-Token"),
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
) -> ApiKey:
    """Resolve the calling app from either the public widget token or the
    secret ck_ key. Both map to one ApiKey → one knowledge base."""
    rec: Optional[ApiKey] = None
    if x_widget_token:
        rec = db.query(ApiKey).filter(ApiKey.widget_token == x_widget_token).first()
    elif authorization and authorization.lower().startswith("bearer "):
        raw = authorization.split(" ", 1)[1].strip()
        if raw.startswith("ck_"):
            rec = db.query(ApiKey).filter(ApiKey.key_hash == hash_key(raw)).first()

    if rec is None or rec.revoked:
        raise HTTPException(status_code=401, detail="Invalid or revoked app token.")
    owner = db.get(User, rec.user_id)
    if owner is None or getattr(owner, "api_blocked", False) or getattr(owner, "is_banned", False):
        raise HTTPException(status_code=403, detail="This assistant is currently unavailable.")
    return rec


class RagChatRequest(BaseModel):
    question: str
    history: list = []
    session_id: Optional[str] = None


def _log_widget_turn(db: Session, key_id: int, session_id: str, question: str, answer: str, answered: bool) -> Optional[int]:
    """Record a widget Q&A turn for the developer's analytics + transcripts.
    Returns the assistant message id (so the widget can attach 👍/👎 feedback).
    Best-effort: never let logging break the answer."""
    try:
        sid = (session_id or "anon")[:64]
        db.add(WidgetMessage(api_key_id=key_id, session_id=sid, role="user", content=question[:4000]))
        bot = WidgetMessage(api_key_id=key_id, session_id=sid, role="assistant",
                            content=(answer or "")[:8000], answered=bool(answered))
        db.add(bot)
        db.commit()
        db.refresh(bot)
        return bot.id
    except Exception:
        db.rollback()
        return None


@router.post("/v1/rag/chat")
def rag_chat(
    req: RagChatRequest,
    app_key: ApiKey = Depends(_resolve_app),
    db: Session = Depends(get_db),
):
    """Answer a question from the app's knowledge base. CORS is opened for this
    path in main.py so it works from any embedding site."""
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required.")

    result = ask_kb_question(_collection_name(app_key.id), question, req.history or [])
    answer = result.get("answer", "")
    sources = result.get("sources", [])

    app_key.usage_count = (app_key.usage_count or 0) + 1
    db.commit()
    # No sources => the KB had nothing relevant (or no docs): a content gap.
    message_id = _log_widget_turn(db, app_key.id, req.session_id or "anon", question, answer, answered=bool(sources))

    return {
        "answer": answer,
        "message_id": message_id,
        "sources": [
            {"content": s.get("content", ""), "filename": (s.get("metadata") or {}).get("filename", "")}
            for s in sources
        ],
    }


class WidgetFeedbackRequest(BaseModel):
    message_id: int
    value: int  # 1 = helpful, -1 = not helpful, 0 = clear


@router.post("/v1/rag/feedback")
def widget_feedback(
    req: WidgetFeedbackRequest,
    app_key: ApiKey = Depends(_resolve_app),
    db: Session = Depends(get_db),
):
    """Record a visitor's 👍/👎 on a widget answer (public, widget token)."""
    msg = db.get(WidgetMessage, req.message_id)
    if msg is None or msg.api_key_id != app_key.id:
        raise HTTPException(status_code=404, detail="Message not found.")
    msg.feedback = 1 if req.value > 0 else (-1 if req.value < 0 else 0)
    db.commit()
    return {"ok": True}


class WidgetLeadRequest(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    message: Optional[str] = None
    session_id: Optional[str] = None


@router.post("/v1/rag/lead")
def submit_widget_lead(
    req: WidgetLeadRequest,
    app_key: ApiKey = Depends(_resolve_app),
    db: Session = Depends(get_db),
):
    """Capture a visitor contact from the embedded widget (public, widget token)."""
    name = (req.name or "").strip()[:120]
    email = (req.email or "").strip()[:200]
    message = (req.message or "").strip()[:1000]
    if not (email or name):
        raise HTTPException(status_code=400, detail="A name or email is required.")
    db.add(WidgetLead(api_key_id=app_key.id, name=name or None, email=email or None,
                      message=message or None, session_id=(req.session_id or None)))
    db.commit()
    return {"ok": True}
