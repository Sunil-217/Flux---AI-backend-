from dotenv import load_dotenv
import os

load_dotenv()

NVIDIA_API_KEY = os.getenv(
    "NVIDIA_API_KEY"
)

TAVILY_API_KEY = os.getenv(
    "TAVILY_API_KEY"
)

# ── Auth ──
JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-me-in-production")
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", str(60 * 24 * 7)))  # 7 days
OTP_TTL_MINUTES = int(os.getenv("OTP_TTL_MINUTES", "10"))

# ── Email (SMTP) for OTP delivery. If unset, OTPs are logged to the server
#    console as a dev fallback so the flow still works without SMTP setup. ──
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
SMTP_FROM = os.getenv("SMTP_FROM") or SMTP_USER

