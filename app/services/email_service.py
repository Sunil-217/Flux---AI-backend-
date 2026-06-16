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
    FRONTEND_URL,
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


def send_invite_email(to_email: str, link: str, invited_by: str | None = None) -> None:
    """Email an invite link. Best-effort and NEVER raises: the admin always gets
    the link in the API response, so a blocked/missing SMTP must not fail the
    invite-creation request (it would also strip CORS headers on a 500)."""
    subject = "You're invited to Close AI"
    inviter = f" by {invited_by}" if invited_by else ""
    body = (
        f"You've been invited{inviter} to join Close AI.\n\n"
        f"Set your password and get started here:\n{link}\n\n"
        f"This invite link expires in 7 days. If you weren't expecting this, you can ignore this email."
    )

    if not _smtp_configured():
        if not IS_PRODUCTION:
            print(f"[INVITE] SMTP not configured — link for {to_email}: {link}", flush=True)
        else:
            log.warning("Invite email not sent for %s — SMTP is not configured.", to_email)
        return

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = SMTP_FROM or SMTP_USER
    msg["To"] = to_email
    msg.set_content(body)

    ports: list = []
    for p in (SMTP_PORT, 2525, 587, 465):
        try:
            p = int(p)
        except (TypeError, ValueError):
            continue
        if p not in ports:
            ports.append(p)

    last_exc = None
    for port in ports:
        try:
            _send_via(SMTP_HOST, port, SMTP_USER, SMTP_PASS, msg)
            return  # sent
        except Exception as exc:  # noqa: BLE001 — try the next port
            last_exc = exc
            continue
    log.error("Invite email failed for %s via all ports: %s", to_email, last_exc)


# ── Announcements (admin → all users) ────────────────────────────────────────
def _deliver_best_effort(msg: EmailMessage) -> bool:
    """Try the configured SMTP port, then cloud-friendly fallbacks. Returns True
    on the first success, False if every attempt failed. Never raises."""
    ports: list = []
    for p in (SMTP_PORT, 2525, 587, 465):
        try:
            p = int(p)
        except (TypeError, ValueError):
            continue
        if p not in ports:
            ports.append(p)
    for port in ports:
        try:
            _send_via(SMTP_HOST, port, SMTP_USER, SMTP_PASS, msg)
            return True
        except Exception:  # noqa: BLE001 — try the next port
            continue
    return False


def _announcement_text(subject: str, message: str) -> str:
    """Plain-text fallback for clients that don't render HTML."""
    return (
        f"{subject}\n\n"
        f"{message}\n\n"
        f"Open Close AI: {FRONTEND_URL}\n\n"
        f"— The Close AI team\n"
        f"You're receiving this because you have a Close AI account."
    )


def _announcement_html(subject: str, message: str) -> str:
    """A branded, responsive HTML announcement. Table-based + inline styles only
    (the only thing email clients render reliably). No <style> blocks."""
    import html as _html

    safe_subject = _html.escape(subject)
    safe_body = _html.escape(message).replace("\n", "<br>")
    return (
        '<!DOCTYPE html>'
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f'<title>{safe_subject}</title></head>'
        '<body style="margin:0;padding:0;background:#f4f4f7;'
        '-webkit-font-smoothing:antialiased;'
        'font-family:-apple-system,BlinkMacSystemFont,\'Segoe UI\',Roboto,Helvetica,Arial,sans-serif;">'
        '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" '
        'style="background:#f4f4f7;padding:32px 16px;"><tr><td align="center">'
        # ── Card ──
        '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" '
        'style="max-width:560px;width:100%;background:#ffffff;border-radius:16px;'
        'overflow:hidden;box-shadow:0 12px 40px -12px rgba(0,0,0,0.18);">'
        # Accent bar
        '<tr><td style="height:5px;background:linear-gradient(90deg,#fca5a5,#f87171,#ef4444);"></td></tr>'
        # Header
        '<tr><td style="padding:30px 36px 8px;">'
        '<span style="font-size:20px;font-weight:700;color:#ef4444;letter-spacing:-0.02em;">Close AI</span>'
        '<span style="font-size:12px;color:#9ca3af;margin-left:8px;">Announcement</span>'
        '</td></tr>'
        # Subject
        f'<tr><td style="padding:8px 36px 0;"><h1 style="margin:0;font-size:22px;'
        f'line-height:1.3;font-weight:700;color:#111827;letter-spacing:-0.01em;">{safe_subject}</h1></td></tr>'
        # Body
        f'<tr><td style="padding:16px 36px 4px;font-size:15px;line-height:1.65;color:#374151;">{safe_body}</td></tr>'
        # CTA
        '<tr><td style="padding:24px 36px 8px;">'
        f'<a href="{FRONTEND_URL}" style="display:inline-block;background:#ef4444;color:#ffffff;'
        'text-decoration:none;font-size:14px;font-weight:600;padding:12px 26px;border-radius:10px;">'
        'Open Close AI</a></td></tr>'
        # Footer
        '<tr><td style="padding:24px 36px 30px;border-top:1px solid #f0f0f3;margin-top:12px;">'
        '<p style="margin:16px 0 0;font-size:12px;line-height:1.5;color:#9ca3af;">'
        'You\'re receiving this because you have a Close AI account.<br>'
        'Powered by <span style="color:#ef4444;font-weight:600;">Fluxera</span>.</p>'
        '</td></tr>'
        '</table></td></tr></table></body></html>'
    )


def send_announcement_email(to_email: str, subject: str, message: str) -> None:
    """Email one user a branded announcement (multipart text + HTML). Best-effort
    and NEVER raises — a bulk blast must not stop because one address bounces."""
    if not _smtp_configured():
        if not IS_PRODUCTION:
            print(f"[ANNOUNCE] SMTP not configured — would email {to_email}: {subject}", flush=True)
        return
    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = SMTP_FROM or SMTP_USER
        msg["To"] = to_email
        msg.set_content(_announcement_text(subject, message))
        msg.add_alternative(_announcement_html(subject, message), subtype="html")
        if not _deliver_best_effort(msg):
            log.warning("Announcement email failed for %s (all ports).", to_email)
    except Exception as exc:  # noqa: BLE001 — never propagate from a bulk send
        log.warning("Announcement email error for %s: %s", to_email, exc)
