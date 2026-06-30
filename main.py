from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import inspect, text
from slowapi.errors import RateLimitExceeded

from app.core.config import CORS_ORIGINS, ADMIN_EMAILS
from app.core.rate_limit import limiter
from app.db import Base, SessionLocal, engine
from app import models  # noqa: F401  (register models on Base before create_all)

from app.api.routes import upload
from app.api.routes import chat
from app.api.routes import delete
from app.api.routes import auth
from app.api.routes import chats
from app.api.routes import title
from app.api.routes import url
from app.api.routes import assist
from app.api.routes import share
from app.api.routes import generate
from app.api.routes import apikeys
from app.api.routes import public_api
from app.api.routes import research
from app.api.routes import memory
from app.api.routes import quiz
from app.api.routes import media_sources
from app.api.routes import tts
from app.api.routes import admin
from app.api.routes import features
from app.api.routes import broadcast
from app.api.routes import invite
from app.api.routes import kb
from app.api.routes import payments

# Create database tables if they don't exist yet.
Base.metadata.create_all(bind=engine)


def _ensure_schema():
    """Idempotent micro-migration: add columns that create_all() won't add to
    tables that already existed before the column was introduced. Works on both
    SQLite (dev) and Postgres (prod) by reading the live column list via the
    SQLAlchemy inspector and issuing portable ALTER TABLE ADD COLUMN."""
    # (table, column, DDL type + default) — each added only if missing.
    additions = [
        ("otp_codes", "attempts", "INTEGER NOT NULL DEFAULT 0"),
        ("users", "is_admin", "BOOLEAN NOT NULL DEFAULT 0"),
        ("users", "is_banned", "BOOLEAN NOT NULL DEFAULT 0"),
        ("users", "api_blocked", "BOOLEAN NOT NULL DEFAULT 0"),
        ("users", "avatar", "TEXT"),
        # Per-app RAG: plan tier + public widget token on each developer key.
        ("api_keys", "plan", "VARCHAR NOT NULL DEFAULT 'free'"),
        ("api_keys", "widget_token", "VARCHAR"),
        ("api_keys", "widget_config", "TEXT"),
        ("widget_messages", "feedback", "INTEGER NOT NULL DEFAULT 0"),
        ("widget_messages", "answered", "BOOLEAN NOT NULL DEFAULT 1"),
    ]
    try:
        insp = inspect(engine)
        existing_tables = set(insp.get_table_names())
        with engine.connect() as conn:
            for table, column, ddl in additions:
                if table not in existing_tables:
                    continue  # create_all() already made it with the column
                cols = {c["name"] for c in insp.get_columns(table)}
                if column in cols:
                    continue
                # Postgres uses TRUE/FALSE rather than 0/1 for boolean defaults.
                stmt = ddl
                if not engine.url.drivername.startswith("sqlite") and "BOOLEAN" in ddl:
                    stmt = ddl.replace("DEFAULT 0", "DEFAULT FALSE")
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {stmt}"))
            conn.commit()
    except Exception:
        # Non-fatal: a fresh DB already gets every column via create_all().
        pass


def _bootstrap_admins():
    """Grant platform-admin to any verified account whose email is in
    ADMIN_EMAILS. Idempotent — runs every startup, only flips users not yet
    admin. The designated admin just signs up + verifies normally, then becomes
    admin automatically (no manual DB editing, no self-promotion endpoint)."""
    if not ADMIN_EMAILS:
        return
    db = SessionLocal()
    try:
        users = db.query(models.User).filter(models.User.email.in_(ADMIN_EMAILS)).all()
        changed = False
        for u in users:
            if not u.is_admin:
                u.is_admin = True
                changed = True
        if changed:
            db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()


def _seed_plans():
    """Populate the plans table from defaults on first boot (idempotent)."""
    db = SessionLocal()
    try:
        from app.services.plan_service import seed_default_plans
        seed_default_plans(db)
    except Exception:
        db.rollback()
    finally:
        db.close()


_ensure_schema()
_bootstrap_admins()
_seed_plans()

app = FastAPI()

# ── Rate limiting (slowapi) ──
app.state.limiter = limiter


async def _ratelimit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Too many attempts. Please wait a minute and try again."},
    )


app.add_exception_handler(RateLimitExceeded, _ratelimit_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    # Expose the download filename so the browser/frontend can read the
    # LLM-generated title from generated documents (PDF/Excel/Word/PPT).
    expose_headers=["Content-Disposition"],
)


# Open CORS for the embeddable widget endpoints ONLY (everything under /v1/rag/:
# the RAG chat + the public appearance config). The widget is loaded in an iframe
# / called from arbitrary customer domains, and it authenticates with a public,
# RAG-only widget token (not cookies), so reflecting any origin here is safe and
# scoped. Added AFTER CORSMiddleware so it is the OUTERMOST layer — it
# short-circuits the preflight before the stricter global CORS sees it.
_RAG_PUBLIC_PREFIX = "/v1/rag/"


@app.middleware("http")
async def _rag_widget_cors(request: Request, call_next):
    from starlette.responses import Response

    if request.url.path.startswith(_RAG_PUBLIC_PREFIX):
        if request.method == "OPTIONS":
            return Response(
                status_code=200,
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                    "Access-Control-Allow-Headers": "Authorization, Content-Type, X-Widget-Token",
                    "Access-Control-Max-Age": "86400",
                },
            )
        response = await call_next(request)
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response

    return await call_next(request)

app.include_router(auth.router)
app.include_router(chats.router)
app.include_router(upload.router)
app.include_router(chat.router)
app.include_router(delete.router)
app.include_router(title.router)
app.include_router(url.router)
app.include_router(assist.router)
app.include_router(share.router)
app.include_router(generate.router)
app.include_router(apikeys.router)
app.include_router(public_api.router)  # /v1/* — OpenAI-compatible developer API
app.include_router(research.router)    # /research — deep research with citations
app.include_router(memory.router)      # /memory — cross-chat user memory
app.include_router(quiz.router)        # /quiz — quiz generator from docs/content
app.include_router(media_sources.router)  # /upload-youtube, /upload-github
app.include_router(tts.router)         # /tts — neural text-to-speech (edge-tts)
app.include_router(admin.router)       # /admin/* — platform admin control panel
app.include_router(features.router)    # /features — public read of platform feature flags
app.include_router(broadcast.router)   # /broadcast — public read of the active announcement
app.include_router(invite.router)      # /invite/* — invite-link check + accept (onboarding)
app.include_router(kb.router)          # /api-keys/{id}/kb + /v1/rag/chat + /plans — per-app RAG
app.include_router(payments.router)    # /admin/payment-gateways/* — payment gateway config

# Telegram bridge (optional): starts a polling thread when TELEGRAM_BOT_TOKEN is set.
from app.services.telegram_bot import start_telegram_bridge  # noqa: E402

start_telegram_bridge()


@app.get("/")
def home():
    return {"message": "Close AI Backend Running"}
