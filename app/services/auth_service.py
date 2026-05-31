"""Sign-up / OTP-verify / sign-in business logic."""

import secrets
from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from app.core.config import OTP_TTL_MINUTES
from app.core.security import hash_password, verify_password
from app.models import User, OtpCode
from app.services.email_service import send_otp_email


def _generate_otp() -> str:
    """Cryptographically secure 6-digit code."""
    return f"{secrets.randbelow(1_000_000):06d}"


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
    if not verify_password(code, rec.code_hash):
        raise ValueError("Incorrect verification code.")

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
