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
    msg.add_alternative(_otp_html(code), subtype="html")

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
    msg.add_alternative(_invite_html(link, invited_by), subtype="html")

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


def _email_shell(
    eyebrow: str,
    heading: str,
    intro_html: str = "",
    hero_html: str = "",
    cta_label: str = "",
    cta_url: str = "",
    preheader: str = "",
) -> str:
    """The shared premium email frame — dark, on-brand, fully table-based and
    bulletproof (Outlook VML button, hidden preheader, solid fallbacks behind
    every gradient, no <style> blocks or external assets). EVERY transactional and
    announcement email renders through this, so they all look identical.

    `eyebrow` / `heading` / `preheader` are PLAIN text (escaped here);
    `intro_html` / `hero_html` are trusted HTML the caller already assembled."""
    import html as _html

    font = "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif"
    safe_eyebrow = _html.escape(eyebrow)
    safe_heading = _html.escape(heading)
    safe_pre = _html.escape(preheader or heading)
    pad = "&#847;&zwnj;&nbsp;" * 24  # invisible spacer so the inbox preview shows only the preheader

    intro_block = (
        f'<tr><td style="padding:22px 40px 0;color:#b9b9c0;font-size:15.5px;line-height:1.7;font-family:{font};">{intro_html}</td></tr>'
        if intro_html
        else ""
    )
    hero_block = f'<tr><td style="padding:22px 40px 0;">{hero_html}</td></tr>' if hero_html else ""
    cta_block = ""
    if cta_label and cta_url:
        safe_label = _html.escape(cta_label)
        cta_block = f"""<tr><td style="padding:30px 40px 6px;">
  <!--[if mso]>
  <v:roundrect xmlns:v="urn:schemas-microsoft-com:vml" xmlns:w="urn:schemas-microsoft-com:office:word" href="{cta_url}" style="height:48px;v-text-anchor:middle;width:240px;" arcsize="24%" strokecolor="#ef4444" fillcolor="#ef4444">
  <w:anchorlock/>
  <center style="color:#ffffff;font-family:{font};font-size:15px;font-weight:bold;">{safe_label}</center>
  </v:roundrect>
  <![endif]-->
  <!--[if !mso]><!-->
  <a href="{cta_url}" style="display:inline-block;background:linear-gradient(135deg,#fb7185,#ef4444);background-color:#ef4444;color:#ffffff;text-decoration:none;font-size:15px;font-weight:700;line-height:48px;padding:0 32px;border-radius:12px;font-family:{font};">{safe_label}&nbsp;&nbsp;&rarr;</a>
  <!--<![endif]-->
</td></tr>"""

    return f"""<!DOCTYPE html>
<html lang="en" xmlns:v="urn:schemas-microsoft-com:vml" xmlns:o="urn:schemas-microsoft-com:office:office">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="X-UA-Compatible" content="IE=edge">
<meta name="color-scheme" content="dark light">
<meta name="supported-color-schemes" content="dark light">
<title>{safe_heading}</title>
<!--[if mso]><noscript><xml><o:OfficeDocumentSettings><o:PixelsPerInch>96</o:PixelsPerInch></o:OfficeDocumentSettings></xml></noscript><![endif]-->
</head>
<body style="margin:0;padding:0;background-color:#07070a;-webkit-font-smoothing:antialiased;font-family:{font};">
<div style="display:none;max-height:0;overflow:hidden;opacity:0;color:#07070a;font-size:1px;line-height:1px;">{safe_pre}{pad}</div>
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" bgcolor="#07070a" style="background-color:#07070a;">
<tr><td align="center" style="padding:36px 14px;">

<table role="presentation" width="600" cellpadding="0" cellspacing="0" border="0" style="width:600px;max-width:600px;background-color:#111114;border:1px solid #26262b;border-radius:18px;overflow:hidden;">

<tr><td height="4" style="height:4px;font-size:0;line-height:0;background:linear-gradient(90deg,#fca5a5,#f87171,#ef4444);">&nbsp;</td></tr>

<tr><td bgcolor="#141015" style="background:linear-gradient(180deg,#1c1013,#121215);padding:34px 40px 26px;">
  <table role="presentation" cellpadding="0" cellspacing="0" border="0"><tr>
    <td width="46" height="46" align="center" valign="middle" bgcolor="#ef4444" style="width:46px;height:46px;background:linear-gradient(135deg,#fb7185,#ef4444);border-radius:13px;color:#ffffff;font-family:{font};font-size:23px;font-weight:800;text-align:center;line-height:46px;">C</td>
    <td valign="middle" style="padding-left:14px;">
      <div style="color:#ffffff;font-size:19px;font-weight:700;letter-spacing:-0.02em;line-height:1.1;">Close AI</div>
      <div style="color:#8a8a93;font-size:12px;line-height:1.4;margin-top:3px;">Document intelligence</div>
    </td>
  </tr></table>
  <div style="height:22px;line-height:22px;font-size:0;">&nbsp;</div>
  <div style="color:#fb7185;font-size:11px;font-weight:700;letter-spacing:0.18em;text-transform:uppercase;">{safe_eyebrow}</div>
  <h1 style="margin:9px 0 0;color:#f6f6f7;font-size:25px;line-height:1.3;font-weight:800;letter-spacing:-0.02em;">{safe_heading}</h1>
</td></tr>
{intro_block}
{hero_block}
{cta_block}
<tr><td style="padding:30px 40px 0;"><table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0"><tr><td height="1" style="height:1px;font-size:0;line-height:0;background-color:#26262b;">&nbsp;</td></tr></table></td></tr>

<tr><td style="padding:20px 40px 34px;">
  <p style="margin:0;color:#6f6f78;font-size:12px;line-height:1.6;font-family:{font};">This is an automated message from Close AI. If you weren&rsquo;t expecting it, you can safely ignore this email.</p>
  <p style="margin:6px 0 0;color:#6f6f78;font-size:12px;line-height:1.6;font-family:{font};">Powered by <span style="color:#fb7185;font-weight:600;">Fluxera</span></p>
</td></tr>

</table>

<table role="presentation" width="600" cellpadding="0" cellspacing="0" border="0" style="width:600px;max-width:600px;"><tr><td align="center" style="padding:18px 16px 0;">
  <p style="margin:0;color:#4a4a52;font-size:11px;line-height:1.5;font-family:{font};">&copy; Close AI &middot; Powered by Fluxera</p>
</td></tr></table>

</td></tr>
</table>
</body>
</html>"""


def _announcement_html(subject: str, message: str) -> str:
    """Admin announcement — rendered through the shared shell."""
    import html as _html
    import re as _re

    safe_body = _html.escape(message).replace("\n", "<br>")
    snippet = _re.sub(r"\s+", " ", message or "").strip()[:110]
    return _email_shell(
        eyebrow="Announcement",
        heading=subject,
        intro_html=safe_body,
        cta_label="Open Close AI",
        cta_url=FRONTEND_URL,
        preheader=snippet or subject,
    )


def _otp_html(code: str) -> str:
    """Verification / password-reset code email — shared shell + a big code box.
    One template serves both signup verification and password reset."""
    import html as _html

    safe_code = _html.escape(code)
    code_box = (
        '<table role="presentation" cellpadding="0" cellspacing="0" border="0"><tr>'
        '<td style="background-color:#1a1a1f;border:1px solid #2e2e35;border-radius:14px;'
        'padding:16px 28px;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;'
        f'font-size:34px;font-weight:700;letter-spacing:0.32em;color:#f6f6f7;text-align:center;">{safe_code}</td>'
        "</tr></table>"
    )
    return _email_shell(
        eyebrow="Verification",
        heading="Your verification code",
        intro_html='Enter this code to continue. It expires in <strong style="color:#d8d8de;font-weight:600;">10 minutes</strong>.',
        hero_html=code_box,
        preheader=f"Your Close AI code: {safe_code}",
    )


def _invite_html(link: str, invited_by: str | None = None) -> str:
    """Admin invite email — shared shell + an Accept button."""
    import html as _html

    inviter = f" by {_html.escape(invited_by)}" if invited_by else ""
    intro = (
        f"You&rsquo;ve been invited{inviter} to join "
        '<strong style="color:#d8d8de;font-weight:600;">Close AI</strong> — your AI workspace for '
        "documents, research, and code. Set a password to get started. This invite link expires in 7 days."
    )
    return _email_shell(
        eyebrow="Invitation",
        heading="You're invited to Close AI",
        intro_html=intro,
        cta_label="Accept invitation",
        cta_url=link,
        preheader="You've been invited to Close AI",
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


def _open_smtp(timeout: int = 8):
    """Open + authenticate a SINGLE SMTP connection, trying the configured port
    then fallbacks. Returns a live, logged-in smtplib server, or None if every
    port failed. Short timeout so a blocked port fails fast instead of hanging."""
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
            if port == 465:
                server = smtplib.SMTP_SSL(SMTP_HOST, port, timeout=timeout)
            else:
                server = smtplib.SMTP(SMTP_HOST, port, timeout=timeout)
                server.ehlo()
                server.starttls()
                server.ehlo()
            server.login(SMTP_USER, SMTP_PASS)
            return server
        except Exception:  # noqa: BLE001 — try the next port
            continue
    return None


def send_announcement_bulk(recipients: list, subject: str, message: str) -> int:
    """Send the announcement to many recipients over ONE reused SMTP connection
    (connect + log in once, then a single send per recipient). This is what makes
    a blast go out in ~1-2s instead of re-connecting per email — the old path paid
    a full TCP+TLS+AUTH handshake (seconds) for every single recipient. Returns the
    count actually sent. Best-effort — one bad address never stops the rest."""
    if not recipients:
        return 0
    if not _smtp_configured():
        if not IS_PRODUCTION:
            print(f"[ANNOUNCE] SMTP not configured — would email {len(recipients)} users: {subject}", flush=True)
        return 0

    # The body is identical for everyone — render it ONCE; only the envelope
    # (the "To" line) changes per recipient.
    html = _announcement_html(subject, message)
    text = _announcement_text(subject, message)
    sender = SMTP_FROM or SMTP_USER

    server = _open_smtp()
    if server is None:
        log.warning("Announcement blast: could not open an SMTP connection (all ports failed).")
        return 0

    sent = 0
    try:
        for to_email in recipients:
            try:
                msg = EmailMessage()
                msg["Subject"] = subject
                msg["From"] = sender
                msg["To"] = to_email
                msg.set_content(text)
                msg.add_alternative(html, subtype="html")
                server.send_message(msg)
                sent += 1
            except Exception as exc:  # noqa: BLE001 — skip a bad address, keep going
                log.warning("Announcement send failed for %s: %s", to_email, exc)
                continue
    finally:
        try:
            server.quit()
        except Exception:
            pass
    return sent
