from dotenv import load_dotenv
import os
import sys

load_dotenv()

# ── App mode ── set APP_ENV=production in real deployments. Used as a switch
# for safety checks below (fail-closed on missing/weak production secrets).
APP_ENV = (os.getenv("APP_ENV") or "development").lower().strip()
IS_PRODUCTION = APP_ENV == "production"

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")   # Chat / Router / Plan (faster inference)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
# Pollinations.ai (kept as a fallback option for /generate/video).
POLLINATIONS_API_KEY = os.getenv("POLLINATIONS_API_KEY")

# Hugging Face Inference (primary path for /video as of 2026 — most generous
# truly-free signup). Get a token at https://huggingface.co/settings/tokens
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

# ── Telegram bridge (optional) — when set, main.py starts a daemon polling
#    thread that answers Telegram messages via the RAG chat pipeline. ──
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# ── Auth ──
_DEFAULT_JWT_SECRET = "dev-secret-change-me-in-production"
JWT_SECRET = os.getenv("JWT_SECRET", _DEFAULT_JWT_SECRET)
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", str(60 * 24 * 7)))  # 7 days
OTP_TTL_MINUTES = int(os.getenv("OTP_TTL_MINUTES", "10"))

# ── Admin bootstrap ── comma-separated emails granted platform-admin on startup.
# These accounts (once signed up + verified) can reach the /admin control panel.
# Override in production via the ADMIN_EMAILS env var.
_ADMIN_EMAILS_RAW = os.getenv("ADMIN_EMAILS", "fluxera.noreply@gmail.com")
ADMIN_EMAILS = {e.strip().lower() for e in _ADMIN_EMAILS_RAW.split(",") if e.strip()}

# ── CORS: comma-separated list of allowed frontend origins ──
_CORS_DEFAULT = "http://localhost:3000,http://127.0.0.1:3000"
_CORS_RAW = os.getenv("CORS_ORIGINS", _CORS_DEFAULT)
CORS_ORIGINS = [o.strip() for o in _CORS_RAW.split(",") if o.strip()]

# ── Public base URL of the frontend (used to build invite links) ──
# Defaults to the first configured CORS origin, then localhost. Set FRONTEND_URL
# in production to your real site (e.g. https://close-ai.vercel.app).
FRONTEND_URL = (
    os.getenv("FRONTEND_URL")
    or (CORS_ORIGINS[0] if CORS_ORIGINS else "http://localhost:3000")
).rstrip("/")

# ── Email (SMTP) for OTP delivery. If unset, OTPs are logged to the server
#    console as a dev fallback so the flow still works without SMTP setup. ──
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
SMTP_FROM = os.getenv("SMTP_FROM") or SMTP_USER


# ── Production safety checks ─────────────────────────────────────────────────
# In production (APP_ENV=production) we FAIL FAST on insecure defaults rather
# than silently running with a publicly-known JWT secret or open CORS — both
# are critical security holes, easy to miss in deploy scripts.
def _enforce_production_safety() -> None:
    problems = []
    if JWT_SECRET == _DEFAULT_JWT_SECRET or len(JWT_SECRET) < 32:
        problems.append(
            "JWT_SECRET must be set to a strong random value (>=32 chars) in production. "
            "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
        )
    if _CORS_RAW == _CORS_DEFAULT:
        problems.append(
            "CORS_ORIGINS must be set to your real frontend URL(s) in production "
            "(currently defaulting to localhost — the backend will be unreachable)."
        )
    if not NVIDIA_API_KEY:
        problems.append("NVIDIA_API_KEY is required.")
    if problems:
        msg = "\n  - ".join(["Production startup blocked:"] + problems)
        print(msg, file=sys.stderr, flush=True)
        raise RuntimeError(msg)


if IS_PRODUCTION:
    _enforce_production_safety()

