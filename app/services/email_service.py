"""
Sends the OTP via SMTP email.

If SMTP isn't configured (no SMTP_HOST/USER/PASS in .env), the code is logged
to the server console instead — so the sign-up flow works out of the box and
upgrades to real email the moment you add credentials.
"""

import logging
import smtplib
from email.message import EmailMessage

from app.core.config import (
    IS_PRODUCTION,
    SMTP_HOST,
    SMTP_PORT,
    SMTP_USER,
    SMTP_PASS,
    SMTP_FROM,
)

log = logging.getLogger("close_ai.email")


def _smtp_configured() -> bool:
    return bool(SMTP_HOST and SMTP_USER and SMTP_PASS)


def _dev_log_code(to_email: str, code: str, reason: str) -> None:
    """Print the OTP to the server console in DEV ONLY. In production we must
    not log secret codes — that's a credential leak; we surface a hard error
    so the operator notices the missing/broken SMTP config instead."""
    if IS_PRODUCTION:
        log.error("OTP delivery failed in production (%s) for %s — SMTP must be configured.",
                  reason, to_email)
        raise RuntimeError("OTP delivery failed: SMTP is not configured/working in production.")
    print(f"[OTP] {reason} — code for {to_email}: {code}", flush=True)


def send_otp_email(to_email: str, code: str) -> None:
    subject = "Your Close AI verification code"
    body = (
        f"Welcome to Close AI!\n\n"
        f"Your verification code is: {code}\n\n"
        f"It expires in 10 minutes. If you didn't request this, you can ignore this email."
    )

    if not _smtp_configured():
        _dev_log_code(to_email, code, "SMTP not configured")
        return

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = SMTP_FROM or SMTP_USER
    msg["To"] = to_email
    msg.set_content(body)

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
    except Exception as exc:
        # In dev, fall back to console so sign-up still works.
        # In production this raises so the operator can fix SMTP fast.
        _dev_log_code(to_email, code, f"Email send failed ({type(exc).__name__})")
