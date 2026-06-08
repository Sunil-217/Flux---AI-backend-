"""SQLAlchemy database setup. SQLite by default; set DATABASE_URL to a Postgres
URL (e.g. postgresql+psycopg://user:pass@host/db) for real deployments."""

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./flux_ai.db")

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
