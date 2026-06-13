"""SQLAlchemy database setup. SQLite by default; set DATABASE_URL to a Postgres
URL for real deployments. You can paste Render's connection string as-is — it
gives `postgres://…`, and we normalise it to the SQLAlchemy + psycopg-v3 form
`postgresql+psycopg://…` automatically (so users never hit a driver mismatch)."""

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

_RAW_DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./flux_ai.db")


def _normalize_db_url(url: str) -> str:
    """Make any common Postgres URL spelling work with the psycopg v3 driver.
    Render/Heroku hand out `postgres://…`; SQLAlchemy needs an explicit driver."""
    if url.startswith("postgres://"):
        return "postgresql+psycopg://" + url[len("postgres://"):]
    # `postgresql://` with no driver → default to psycopg v3 (what we install).
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url[len("postgresql://"):]
    return url


DATABASE_URL = _normalize_db_url(_RAW_DATABASE_URL)

# SQLite needs check_same_thread=False to be used from FastAPI worker threads.
# Postgres / MySQL drivers don't accept that flag, so pass it ONLY for SQLite.
_is_sqlite = DATABASE_URL.startswith("sqlite")
_engine_kwargs = (
    {"connect_args": {"check_same_thread": False}}
    if _is_sqlite
    else {"pool_pre_ping": True}  # drop dead connections in long-running prod
)
engine = create_engine(DATABASE_URL, **_engine_kwargs)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """FastAPI dependency that yields a DB session and always closes it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
