"""Password hashing (bcrypt) + JWT tokens + the current-user dependency."""

import json
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from app.core.config import JWT_SECRET, JWT_EXPIRE_MINUTES
from app.db import get_db
from app.models import User, UserChats

_ALGORITHM = "HS256"
_bearer = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    # bcrypt only uses the first 72 bytes — truncate for consistency.
    pw = password.encode("utf-8")[:72]
    return bcrypt.hashpw(pw, bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8")[:72], password_hash.encode("utf-8"))
    except Exception:
        return False


def create_access_token(user_id: int) -> str:
    payload = {
        "sub": str(user_id),
        "exp": datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=_ALGORITHM)


def decode_token(token: str) -> dict:
    return jwt.decode(token, JWT_SECRET, algorithms=[_ALGORITHM])


def get_current_user(
    creds: HTTPAuthorizationCredentials = Depends(_bearer),
    db: Session = Depends(get_db),
) -> User:
    if creds is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Not authenticated")
    try:
        payload = decode_token(creds.credentials)
        user_id = int(payload["sub"])
    except Exception:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token")

    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "User not found")
    if getattr(user, "is_banned", False):
        # A banned account keeps its row (for audit) but cannot use the API.
        raise HTTPException(status.HTTP_403_FORBIDDEN, "This account has been suspended.")
    return user


def require_admin(user: User = Depends(get_current_user)) -> User:
    """FastAPI dependency for /admin/* routes: 403 unless the caller is a
    platform admin. Layered on get_current_user, so it also enforces a valid,
    non-banned token first."""
    if not getattr(user, "is_admin", False):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Admin access required.")
    return user


def user_owns_chat(db: Session, user: User, chat_id: str) -> bool:
    """True iff `chat_id` appears in this user's saved sessions blob.

    We store all of a user's chats as one JSON blob in UserChats. Chat IDs are
    client-generated UUIDs (server doesn't index them), so on any per-chat
    action we have to consult the blob to confirm ownership.
    """
    rec = db.get(UserChats, user.id)
    if rec is None:
        return False
    try:
        sessions = json.loads(rec.data) or []
    except Exception:
        return False
    return any(isinstance(s, dict) and s.get("id") == chat_id for s in sessions)


def require_chat_ownership(
    chat_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> User:
    """FastAPI dependency: 404 if the caller doesn't own this chat.

    Returns 404 (not 403) on purpose — don't leak existence of other users'
    chat IDs.
    """
    if not user_owns_chat(db, user, chat_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Chat not found")
    return user
