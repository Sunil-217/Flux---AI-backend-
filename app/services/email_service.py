"""
Sends the OTP via SMTP email.

If SMTP isn't configured (no SMTP_HOST/USER/PASS in .env), the code is logged
to the server console instead — so the sign-up flow works out of the box and
upgrades to real email the moment you add credentials.
"""

import smtplib
from email.message import EmailMessage

from app.core.config import (
    SMTP_HOST,
    SMTP_PORT,
    SMTP_USER,
    SMTP_PASS,
    SMTP_FROM,
)


def _smtp_configured() -> bool:
    return bool(SMTP_HOST and SMTP_USER and SMTP_PASS)


def send_otp_email(to_email: str, code: str) -> None:
    subject = "Your Close AI verification code"
    body = (
        f"Welcome to Close AI!\n\n"
        f"Your verification code is: {code}\n\n"
        f"It expires in 10 minutes. If you didn't request this, you can ignore this email."
    )

    if not _smtp_configured():
        print(f"[OTP] SMTP not configured — code for {to_email}: {code}", flush=True)
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
        # Never block sign-up on email failure — fall back to console.
        print(f"[OTP] Email send failed ({exc}) — code for {to_email}: {code}", flush=True)
