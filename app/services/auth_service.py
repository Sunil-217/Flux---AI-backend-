"""Sign-up / OTP-verify / sign-in business logic."""

import secrets
from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from app.core.config import OTP_TTL_MINUTES
from app.core.security import hash_password, verify_password
from app.models import User, OtpCode
from app.services.email_service import send_otp_email

# After this many wrong guesses the code is invalidated and a new one must be
# requested — so a 6-digit code can't be brute-forced within its TTL window.
MAX_OTP_ATTEMPTS = 5


def _generate_otp() -> str:
    """Cryptographically secure 6-digit code."""
    return f"{secrets.randbelow(1_000_000):06d}"


def _check_code_or_raise(db, rec, code: str, wrong_msg: str) -> None:
    """Validate a code against an OTP record, enforcing an attempt limit.
    Raises ValueError on any failure (and counts/locks wrong guesses)."""
    if rec.attempts >= MAX_OTP_ATTEMPTS:
        db.query(OtpCode).filter(OtpCode.email == rec.email).delete()
        db.commit()
        raise ValueError("Too many incorrect attempts. Please request a new code.")
    if not verify_password(code, rec.code_hash):
        rec.attempts = (rec.attempts or 0) + 1
        db.commit()
        raise ValueError(wrong_msg)


def issue_otp(db: Session, email: str) -> None:
    code = _generate_otp()
    db.query(OtpCode).filter(OtpCode.email == email).delete()
    db.add(
        OtpCode(
            email=email,
            code_hash=hash_password(code),
            expires_at=datetime.utcnow() + timedelta(minutes=OTP_TTL_MINUTES),
        )
    )
    db.commit()
    send_otp_email(email, code)


def signup(db: Session, name: str, email: str, password: str, phone: str) -> None:
    existing = db.query(User).filter(User.email == email).first()

    if existing and existing.is_verified:
        raise ValueError("This email is already registered. Please sign in.")

    if existing:
        # Unverified account re-registering — refresh details.
        existing.name = name
        existing.phone = phone
        existing.password_hash = hash_password(password)
    else:
        db.add(
            User(
                name=name,
                email=email,
                phone=phone,
                password_hash=hash_password(password),
                is_verified=False,
            )
        )
    db.commit()
    issue_otp(db, email)


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


def resend_otp(db: Session, email: str) -> None:
    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise ValueError("No sign-up found for this email.")
    if user.is_verified:
        raise ValueError("This email is already verified. Please sign in.")
    issue_otp(db, email)


def signin(db: Session, email: str, password: str) -> User:
    user = db.query(User).filter(User.email == email).first()
    if user is None or not verify_password(password, user.password_hash):
        raise ValueError("Invalid email or password.")
    if not user.is_verified:
        raise ValueError("Please verify your email before signing in.")
    return user


def request_password_reset(db: Session, email: str) -> None:
    """Send a reset code if a verified account exists. Stays silent either way
    (so we never reveal whether an email is registered)."""
    user = db.query(User).filter(User.email == email).first()
    if user is not None and user.is_verified:
        issue_otp(db, email)


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
