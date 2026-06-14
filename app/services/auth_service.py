"""Sign-up / OTP-verify / sign-in business logic."""

import hashlib
import hmac
import secrets
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy.orm import Session

from app.core.config import OTP_TTL_MINUTES, ADMIN_EMAILS
from app.core.security import hash_password, verify_password
from app.models import User, OtpCode, Invite

# After this many wrong guesses the code is invalidated and a new one must be
# requested — so a 6-digit code can't be brute-forced within its TTL window.
MAX_OTP_ATTEMPTS = 5


def _generate_otp() -> str:
    """Cryptographically secure 6-digit code."""
    return f"{secrets.randbelow(1_000_000):06d}"


def _hash_otp(code: str) -> str:
    """OTPs are short-lived 6-digit codes guarded by a TTL + attempt limit, so a
    fast SHA-256 is the right tool — bcrypt here just made signup/verify slow on
    constrained hosts (two bcrypt hashes per signup) for no security gain."""
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def _verify_otp_hash(code: str, stored: str) -> bool:
    return hmac.compare_digest(_hash_otp(code), stored)


def _check_code_or_raise(db, rec, code: str, wrong_msg: str) -> None:
    """Validate a code against an OTP record, enforcing an attempt limit.
    Raises ValueError on any failure (and counts/locks wrong guesses)."""
    if rec.attempts >= MAX_OTP_ATTEMPTS:
        db.query(OtpCode).filter(OtpCode.email == rec.email).delete()
        db.commit()
        raise ValueError("Too many incorrect attempts. Please request a new code.")
    if not _verify_otp_hash(code, rec.code_hash):
        rec.attempts = (rec.attempts or 0) + 1
        db.commit()
        raise ValueError(wrong_msg)


def issue_otp(db: Session, email: str) -> str:
    """Create a fresh OTP record and RETURN the code. Delivery is the caller's
    job — endpoints send it in a FastAPI background task so the request never
    blocks on SMTP (a slow/blocked mail server can't 502 the signup anymore)."""
    code = _generate_otp()
    db.query(OtpCode).filter(OtpCode.email == email).delete()
    db.add(
        OtpCode(
            email=email,
            code_hash=_hash_otp(code),
            expires_at=datetime.utcnow() + timedelta(minutes=OTP_TTL_MINUTES),
        )
    )
    db.commit()
    return code


def signup(db: Session, name: str, email: str, password: str, phone: str):
    """Returns (user, auto_verified, otp_code). For a designated admin email the
    account is created admin AND verified with NO OTP (auto_verified=True,
    code=None), so the critical admin login never depends on email. Everyone else
    gets an OTP code back (auto_verified=False) which the endpoint emails in the
    background."""
    existing = db.query(User).filter(User.email == email).first()

    if existing and existing.is_verified:
        raise ValueError("This email is already registered. Please sign in.")

    # A designated admin email becomes admin (and verified) the moment it signs
    # up — no restart, no bootstrap pass, no email round-trip needed.
    is_admin_email = email.lower() in ADMIN_EMAILS

    if existing:
        # Unverified account re-registering — refresh details.
        existing.name = name
        existing.phone = phone
        existing.password_hash = hash_password(password)
        if is_admin_email:
            existing.is_admin = True
            existing.is_verified = True
        user = existing
    else:
        user = User(
            name=name,
            email=email,
            phone=phone,
            password_hash=hash_password(password),
            is_verified=is_admin_email,  # admins are verified on creation
            is_admin=is_admin_email,
        )
        db.add(user)
    db.commit()
    db.refresh(user)

    if is_admin_email:
        return user, True, None  # skip OTP entirely for the admin
    code = issue_otp(db, email)
    return user, False, code


def accept_invite(db: Session, token: str, name: str, password: str, phone: str) -> User:
    """Consume a valid invite: create (or finalise) a PRE-VERIFIED account with
    the chosen password. No OTP — the admin already vouched for this email.
    Raises ValueError on an invalid / used / expired token."""
    inv = db.query(Invite).filter(Invite.token == token).first()
    if inv is None:
        raise ValueError("This invite link is invalid.")
    if inv.accepted:
        raise ValueError("This invite has already been used. Please sign in.")
    if inv.expires_at < datetime.utcnow():
        raise ValueError("This invite has expired. Ask your admin for a new one.")

    email = inv.email.lower()
    existing = db.query(User).filter(User.email == email).first()
    if existing and existing.is_verified:
        raise ValueError("This email is already registered. Please sign in.")

    is_admin_email = email in ADMIN_EMAILS
    if existing:
        existing.name = name
        existing.phone = phone
        existing.password_hash = hash_password(password)
        existing.is_verified = True
        if is_admin_email:
            existing.is_admin = True
        user = existing
    else:
        user = User(
            name=name,
            email=email,
            phone=phone,
            password_hash=hash_password(password),
            is_verified=True,           # invited users are pre-verified
            is_admin=is_admin_email,
        )
        db.add(user)
    inv.accepted = True
    db.commit()
    db.refresh(user)
    return user


def verify_otp(db: Session, email: str, code: str) -> User:
    rec = db.query(OtpCode).filter(OtpCode.email == email).first()
    if rec is None or rec.expires_at < datetime.utcnow():
        raise ValueError("Code expired or not found. Please request a new one.")
    _check_code_or_raise(db, rec, code, "Incorrect verification code.")

    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise ValueError("Account not found.")

    user.is_verified = True
    db.query(OtpCode).filter(OtpCode.email == email).delete()
    db.commit()
    db.refresh(user)
    return user


def resend_otp(db: Session, email: str) -> str:
    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise ValueError("No sign-up found for this email.")
    if user.is_verified:
        raise ValueError("This email is already verified. Please sign in.")
    return issue_otp(db, email)


def ensure_admin_status(db: Session, user: User) -> User:
    """Self-heal admin rights: if this user's email is designated in
    ADMIN_EMAILS but the row isn't admin yet, promote it now. This removes the
    dependence on the startup bootstrap pass — an account added to ADMIN_EMAILS
    (or created before the admin feature shipped) becomes admin on its next
    sign-in / session check, with no restart needed. No-op otherwise."""
    if user is None:
        return user
    if (user.email or "").lower() in ADMIN_EMAILS and not getattr(user, "is_admin", False):
        user.is_admin = True
        if not user.is_verified:
            user.is_verified = True  # a designated admin never needs OTP
        db.commit()
        db.refresh(user)
    return user


def signin(db: Session, email: str, password: str) -> User:
    user = db.query(User).filter(User.email == email).first()
    if user is None or not verify_password(password, user.password_hash):
        raise ValueError("Invalid email or password.")
    if getattr(user, "is_banned", False):
        raise ValueError("This account has been suspended. Contact support.")
    if not user.is_verified:
        raise ValueError("Please verify your email before signing in.")
    return ensure_admin_status(db, user)


def request_password_reset(db: Session, email: str) -> Optional[str]:
    """Return a reset code if a verified account exists, else None. The endpoint
    emails it (in the background) and stays generic either way, so we never reveal
    whether an email is registered."""
    user = db.query(User).filter(User.email == email).first()
    if user is not None and user.is_verified:
        return issue_otp(db, email)
    return None


def reset_password(db: Session, email: str, code: str, new_password: str) -> User:
    rec = db.query(OtpCode).filter(OtpCode.email == email).first()
    if rec is None or rec.expires_at < datetime.utcnow():
        raise ValueError("Code expired or not found. Please request a new one.")
    _check_code_or_raise(db, rec, code, "Incorrect reset code.")

    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise ValueError("Account not found.")

    user.password_hash = hash_password(new_password)
    db.query(OtpCode).filter(OtpCode.email == email).delete()
    db.commit()
    db.refresh(user)
    return user


def update_profile(db: Session, user: User, name: str, phone: str | None = None) -> User:
    name = name.strip()
    if not name:
        raise ValueError("Name cannot be empty.")
    user.name = name
    if phone is not None:
        user.phone = phone.strip()
    db.commit()
    db.refresh(user)
    return user


def change_password(db: Session, user: User, current_password: str, new_password: str) -> None:
    if not verify_password(current_password, user.password_hash):
        raise ValueError("Current password is incorrect.")
    if len(new_password) < 8:
        raise ValueError("New password must be at least 8 characters.")
    user.password_hash = hash_password(new_password)
    db.commit()
