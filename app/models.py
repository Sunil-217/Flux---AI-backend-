from datetime import datetime

from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text

from app.db import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    phone = Column(String, nullable=True)
    password_hash = Column(String, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class OtpCode(Base):
    """A short-lived, hashed one-time code tied to an email (for sign-up verification)."""

    __tablename__ = "otp_codes"

    id = Column(Integer, primary_key=True)
    email = Column(String, index=True, nullable=False)
    code_hash = Column(String, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    attempts = Column(Integer, default=0, nullable=False)  # wrong-guess counter


class UserChats(Base):
    """Per-user chat sessions stored server-side (the user's full sessions array as JSON)."""

    __tablename__ = "user_chats"

    user_id = Column(Integer, primary_key=True)  # references users.id
    data = Column(Text, default="[]", nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class ChatWebContext(Base):
    """The most recent live web-search results for a chat, so follow-up
    questions in the same conversation stay consistent with that current data
    instead of the model 'forgetting' it. Persisted (not in-memory) so it
    survives uvicorn --reload and works across multiple workers."""

    __tablename__ = "chat_web_context"

    chat_id = Column(String, primary_key=True)
    results = Column(Text, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class UserMemory(Base):
    """Durable facts about a user (preferences, role, projects, language),
    extracted from conversations and remembered across chats. Stored as one
    JSON array of short strings per user."""

    __tablename__ = "user_memory"

    user_id = Column(Integer, primary_key=True)  # references users.id
    facts = Column(Text, default="[]", nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class SharedChat(Base):
    """A read-only public snapshot of a chat, addressable by a short id."""

    __tablename__ = "shared_chats"

    id = Column(String, primary_key=True)  # short share code
    owner_id = Column(Integer, index=True, nullable=False)
    title = Column(String, nullable=False, default="Shared chat")
    data = Column(Text, nullable=False)  # JSON: the messages array
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class ApiKey(Base):
    """A developer API key (platform feature: users call Close AI like OpenAI).

    The raw key (`ck_...`) is shown ONCE at creation and never stored — only its
    SHA-256 hash. `prefix` keeps the first/last characters for display."""

    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True, nullable=False)  # references users.id
    name = Column(String, nullable=False, default="My key")
    key_hash = Column(String, unique=True, index=True, nullable=False)  # sha256 hex
    prefix = Column(String, nullable=False)  # e.g. "ck_3fk9…x2ab" for display
    revoked = Column(Boolean, default=False, nullable=False)
    usage_count = Column(Integer, default=0, nullable=False)  # total requests served
    total_tokens = Column(Integer, default=0, nullable=False)  # completion tokens used
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_used_at = Column(DateTime, nullable=True)
