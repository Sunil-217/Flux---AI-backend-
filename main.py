from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text
from slowapi.errors import RateLimitExceeded

from app.core.config import CORS_ORIGINS
from app.core.rate_limit import limiter
from app.db import Base, engine
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

# Create database tables if they don't exist yet.
Base.metadata.create_all(bind=engine)


def _ensure_sqlite_schema():
    """Idempotent micro-migration: add columns that create_all() won't add to
    tables that already existed before the column was introduced."""
    try:
        with engine.connect() as conn:
            cols = [row[1] for row in conn.execute(text("PRAGMA table_info(otp_codes)"))]
            if cols and "attempts" not in cols:
                conn.execute(
                    text("ALTER TABLE otp_codes ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0")
                )
                conn.commit()
    except Exception:
        # Non-fatal: a fresh DB already gets the column via create_all().
        pass


_ensure_sqlite_schema()

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


@app.get("/")
def home():
    return {"message": "Close AI Backend Running"}
