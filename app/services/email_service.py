"""
Sends the OTP via SMTP email (Brevo).

Cloud hosts (incl. some Render setups) often block outbound SMTP on port 587/25
but allow Brevo's alternative port 2525 — so we try several ports before giving
up. If SMTP isn't configured at all, the code is logged to the server console
in dev so the flow still works.
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


def _send_via(host: str, port: int, user: str, pwd: str, msg: EmailMessage) -> None:
    """Send through one host:port. Port 465 = implicit SSL; others = STARTTLS.
    Short timeout so a blocked port fails fast and we can try the next one."""
    if int(port) == 465:
        with smtplib.SMTP_SSL(host, port, timeout=6) as server:
            server.login(user, pwd)
            server.send_message(msg)
    else:
        with smtplib.SMTP(host, port, timeout=6) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(user, pwd)
            server.send_message(msg)


def send_otp_email(to_email: str, code: str) -> None:
    """Email the OTP. Tries the configured port, then common fallbacks. Raises
    ValueError (a clean, user-facing 400) in production if every attempt fails —
    NEVER an unhandled 500 (which would strip CORS headers from the response)."""
    subject = "Your Close AI verification code"
    body = (
        f"Welcome to Close AI!\n\n"
        f"Your verification code is: {code}\n\n"
        f"It expires in 10 minutes. If you didn't request this, you can ignore this email."
    )

    if not _smtp_configured():
        if IS_PRODUCTION:
            log.error("OTP email not sent for %s — SMTP is not configured.", to_email)
            raise ValueError(
                "We couldn't send the verification code — email isn't set up on the server yet. "
                "Please try again later."
            )
        print(f"[OTP] SMTP not configured — code for {to_email}: {code}", flush=True)
        return

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = SMTP_FROM or SMTP_USER
    msg["To"] = to_email
    msg.set_content(body)

    # Try the configured port first, then cloud-friendly fallbacks. 2525 is
    # Brevo's alternative port (usually open when 587 is blocked); 465 is SSL.
    ports: list = []
    for p in (SMTP_PORT, 2525, 587, 465):
        try:
            p = int(p)
        except (TypeError, ValueError):
            continue
        if p not in ports:
            ports.append(p)

    tried = []
    last_exc = None
    for port in ports:
        try:
            _send_via(SMTP_HOST, port, SMTP_USER, SMTP_PASS, msg)
            return  # sent successfully
        except Exception as exc:  # noqa: BLE001 — try the next port
            last_exc = exc
            tried.append(f"{port}={type(exc).__name__}")
            continue

    # Every port failed.
    log.error("OTP email failed for %s via all ports [%s]: %s", to_email, ", ".join(tried), last_exc)
    if IS_PRODUCTION:
        raise ValueError(
            "We couldn't send the verification code right now (mail server unreachable). "
            "Please try again in a moment."
        )
    print(f"[OTP] all SMTP ports failed ({last_exc}) — code for {to_email}: {code}", flush=True)
