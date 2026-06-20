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
    # Platform admin — can reach the /admin/* control panel and manage users.
    # Bootstrapped from ADMIN_EMAILS at startup (see main.py).
    is_admin = Column(Boolean, default=False, nullable=False)
    # Banned users keep their row (for audit) but can't authenticate.
    is_banned = Column(Boolean, default=False, nullable=False)
    # Blocked from the developer API: existing keys stop working and no new keys
    # can be created. Independent of is_banned (a user can keep app access but
    # lose API access).
    api_blocked = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    # Optional profile avatar: a small base64 data URL (uploaded photo) or a
    # "preset:<id>" token. Null = the default monogram.
    avatar = Column(Text, nullable=True)


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
    # Subscription plan governing this app's knowledge-base doc limit. No payment
    # yet — everyone is "free" (1 doc) and paid tiers are display-only upgrades.
    plan = Column(String, default="free", nullable=False)
    # Public, RAG-only token (wk_...) safe to embed in a customer-facing iframe.
    # Unlike the secret ck_ key, it ONLY answers from this app's uploaded docs —
    # it can't call the general LLM API, so leaking it can't burn the owner's
    # quota. Stored in the clear (it is public by design) and recoverable so the
    # embed code can always be re-shown. Generated lazily for pre-existing keys.
    widget_token = Column(String, unique=True, index=True, nullable=True)
    # JSON appearance config for the embeddable chat widget (title, theme, accent,
    # greeting, suggested questions, custom CSS). Owner-editable; null = defaults.
    widget_config = Column(Text, nullable=True)


class FeatureFlag(Base):
    """A platform-wide feature switch, toggled by admins. Only keys present in
    DEFAULT_FEATURES are honoured; a missing row means "use the default" (on)."""

    __tablename__ = "feature_flags"

    key = Column(String, primary_key=True)
    enabled = Column(Boolean, default=True, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class AuditLog(Base):
    """An append-only record of privileged admin actions (who did what, to whom,
    when). Enterprise requirement and a safety net — every mutating /admin call
    writes one row so actions are traceable and reversible to investigate."""

    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    actor_id = Column(Integer, index=True, nullable=False)   # admin who acted
    actor_email = Column(String, nullable=False)             # denormalised for display
    action = Column(String, nullable=False)                  # e.g. "user.ban", "user.delete"
    target_id = Column(Integer, nullable=True)               # affected user id (if any)
    target_email = Column(String, nullable=True)             # affected user email (if any)
    detail = Column(Text, nullable=True)                     # free-text / JSON context
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class Broadcast(Base):
    """A platform-wide announcement banner set by an admin and shown to every
    user. At most one is active at a time (posting a new one deactivates the
    rest); dismissal is per-user and kept client-side, so a brand-new broadcast
    re-appears even after a previous one was dismissed."""

    __tablename__ = "broadcasts"

    id = Column(Integer, primary_key=True, index=True)
    message = Column(Text, nullable=False)
    level = Column(String, nullable=False, default="info")  # info | warning | success
    active = Column(Boolean, default=True, nullable=False)
    created_by = Column(String, nullable=True)              # admin email (display only)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class Invite(Base):
    """A one-time onboarding link an admin sends to a specific email. The
    recipient opens it and sets a password — no OTP needed (the admin vouched
    for them). `token` is the secret carried in the link."""

    __tablename__ = "invites"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, index=True, nullable=False)
    token = Column(String, unique=True, index=True, nullable=False)
    invited_by = Column(String, nullable=True)              # admin email (display only)
    accepted = Column(Boolean, default=False, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class Webhook(Base):
    """An admin-registered outbound webhook. When a subscribed platform event
    fires, the server POSTs signed JSON to `url`. `events` is a JSON array of
    event names; `secret` signs each payload (HMAC-SHA256) so receivers can
    verify the call really came from us."""

    __tablename__ = "webhooks"

    id = Column(Integer, primary_key=True, index=True)
    url = Column(String, nullable=False)
    secret = Column(String, nullable=False)
    events = Column(Text, nullable=False, default="[]")     # JSON array of event names
    enabled = Column(Boolean, default=True, nullable=False)
    created_by = Column(String, nullable=True)              # admin email (display only)
    last_status = Column(String, nullable=True)             # last delivery result, e.g. "200"
    last_triggered_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class KnowledgeDocument(Base):
    """A document uploaded into one API key's RAG knowledge base.

    Each ApiKey (ck_ key) represents one "app/project" and owns an isolated
    ChromaDB collection (kb_<api_key_id>). The key owner uploads their app's
    docs (PDF/Word/etc.); their end users then query them via the embedded chat
    widget or the RAG API. `upload_uid` is the prefix used for this document's
    ChromaDB chunk IDs, so a single document can be deleted independently."""

    __tablename__ = "knowledge_documents"

    id = Column(Integer, primary_key=True, index=True)
    api_key_id = Column(Integer, index=True, nullable=False)  # references api_keys.id
    filename = Column(String, nullable=False)
    file_size = Column(Integer, default=0, nullable=False)
    chunk_count = Column(Integer, default=0, nullable=False)
    upload_uid = Column(String, nullable=False)             # prefix used for ChromaDB chunk IDs
    uploaded_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class Plan(Base):
    """A subscription plan tier for developer apps (knowledge-base RAG).

    Editable by admins from the Plans tab — price, document limit, API rate
    limit, and the list of services it provides. Seeded from defaults on first
    boot (see plan_service.seed_default_plans) so the table is never empty.
    `features` is a JSON array of service strings shown on the pricing card.
    `doc_limit` is ENFORCED on upload; `rate_limit` is ENFORCED on the public API."""

    __tablename__ = "plans"

    id = Column(Integer, primary_key=True, index=True)
    key = Column(String, unique=True, index=True, nullable=False)  # free, go, pro…
    label = Column(String, nullable=False)
    price = Column(String, nullable=False, default="₹0")           # display string
    doc_limit = Column(Integer, nullable=False, default=1)
    rate_limit = Column(Integer, nullable=False, default=20)       # API requests / minute
    blurb = Column(String, nullable=True)
    features = Column(Text, nullable=False, default="[]")          # JSON array of strings
    sort_order = Column(Integer, nullable=False, default=0)
    active = Column(Boolean, nullable=False, default=True)         # hidden from pricing when false
    highlighted = Column(Boolean, nullable=False, default=False)   # "most popular" badge
    updated_at = Column(DateTime, default=datetime.utcnow, nullable=False)
