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
JINA_API_KEY   = os.getenv("JINA_API_KEY")   # Embeddings (RAG). See embedding_service.
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
# Pollinations.ai (kept as a fallback option for /generate/video).
POLLINATIONS_API_KEY = os.getenv("POLLINATIONS_API_KEY")

# Hugging Face Inference (primary path for /video as of 2026 — most generous
# truly-free signup). Get a token at https://huggingface.co/settings/tokens
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

# ── Telegram bridge (optional) — when set, main.py starts a daemon polling
#    thread that answers Telegram messages via the RAG chat pipeline. ──
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# ── Fluxway PSP orchestration (optional) ──
# When all three are set, the Payment Gateways admin can pull the live PSP
# catalog from Fluxway and onboard a PSP at brand level (both Fluxway + Close AI).
# Unset → the page works in local-only mode (manual add), no Fluxway calls.
#   FLUXWAY_BASE_URL      e.g. http://localhost:7111/fluxway/v2
#   FLUXWAY_SECRET_TOKEN  a brand environment secret (X-SECRET-TOKEN); bound to
#                         one brand+environment, so onboarding lands brand-level.
#   FLUXWAY_PSP_FLOW_TYPE the FlowType name whose targets are the PSP catalog.
FLUXWAY_BASE_URL = (os.getenv("FLUXWAY_BASE_URL") or "").rstrip("/")
FLUXWAY_SECRET_TOKEN = os.getenv("FLUXWAY_SECRET_TOKEN") or ""
FLUXWAY_PSP_FLOW_TYPE = os.getenv("FLUXWAY_PSP_FLOW_TYPE") or "PSP"
FLUXWAY_TIMEOUT = int(os.getenv("FLUXWAY_TIMEOUT", "20"))  # seconds

# ── LLM models ────────────────────────────────────────────────────────────────
# Every model id is env-overridable. The defaults are the ones proven in
# production; see the routing note in rag_service for why the short utility
# calls use a non-reasoning model. Change a model without a redeploy of code by
# setting the matching variable.
#
# Groq — the primary provider. Everything works with only GROQ_API_KEY set.
MODEL        = os.getenv("MODEL")        or "openai/gpt-oss-120b"   # chat / RAG
PLAN_MODEL   = os.getenv("PLAN_MODEL")   or "openai/gpt-oss-120b"   # JSON planning
CODE_MODEL   = os.getenv("CODE_MODEL")   or "openai/gpt-oss-120b"   # code edit / Q&A
VISION_MODEL = os.getenv("VISION_MODEL") or "qwen/qwen3.8-27b"      # image / screenshot
ROUTER_MODEL = os.getenv("ROUTER_MODEL") or "qwen/qwen3.8-27b"      # short utility calls
# The critic emits a small JSON verdict, not prose, so it uses the same
# non-reasoning model the other short structured calls use. A reasoning model
# spends completion budget thinking before it writes — invisible tokens that
# still count against the 8,000/min ceiling, on a call whose whole output is
# four fields. Measured on four drafts with known-correct verdicts: identical
# 4/4 judgements, 1121ms -> 376ms average.
CRITIC_MODEL = os.getenv("CRITIC_MODEL") or "qwen/qwen3.8-27b"

# Second Groq model, tried when the primary errors.
FALLBACK_CHAT_MODEL = os.getenv("FALLBACK_CHAT_MODEL") or "qwen/qwen3.8-27b"

# NVIDIA NIM — optional secondary provider (https://build.nvidia.com).
# Picked for agentic reasoning and planning rather than by name recognition:
# Nemotron is NVIDIA's own instruction/reasoning line and `-super-120b-a12b` is
# a mixture-of-experts checkpoint (≈12B active of 120B total), so it reasons at
# large-model quality without large-model latency. NVIDIA_MODEL overrides it.
# `meta/llama-3.3-70b-instruct` is deliberately NOT a default — it was retired
# by NVIDIA (410, end of life 2026-08-26) and is absent from the live catalog.
NVIDIA_MODEL      = os.getenv("NVIDIA_MODEL")      or "nvidia/nemotron-3-super-120b-a12b"
NVIDIA_PLAN_MODEL = os.getenv("NVIDIA_PLAN_MODEL") or NVIDIA_MODEL
NVIDIA_CODE_MODEL = os.getenv("NVIDIA_CODE_MODEL") or NVIDIA_MODEL

# ── Provider routing per role ─────────────────────────────────────────────────
# "groq" | "nvidia" | "auto". "auto" prefers NVIDIA when NVIDIA_API_KEY is set
# and falls back to Groq; an unset key makes it plain Groq. Every role keeps a
# fallback chain, so a provider outage degrades instead of failing.
#
# Defaults keep the hot path (chat, routing, code) on Groq — it is measurably
# faster and is what the app is tuned for — and send only the deliberative roles
# (planning, orchestration) to NVIDIA, which is what an extra provider is worth
# paying latency for.
CHAT_PROVIDER         = (os.getenv("CHAT_PROVIDER")         or "groq").lower().strip()
ROUTER_PROVIDER       = (os.getenv("ROUTER_PROVIDER")       or "groq").lower().strip()
CODE_PROVIDER         = (os.getenv("CODE_PROVIDER")         or "groq").lower().strip()
VISION_PROVIDER       = (os.getenv("VISION_PROVIDER")       or "groq").lower().strip()
PLANNER_PROVIDER      = (os.getenv("PLANNER_PROVIDER")      or "auto").lower().strip()
ORCHESTRATOR_PROVIDER = (os.getenv("ORCHESTRATOR_PROVIDER") or "auto").lower().strip()

# ── Autonomous orchestration limits ───────────────────────────────────────────
# Bounds on the agent loop. Every one of these exists to make runaway behaviour
# impossible rather than unlikely: without them a critic that is never satisfied
# retries forever and a plan that decomposes itself fans out without limit.
MAX_AGENT_RETRIES  = int(os.getenv("MAX_AGENT_RETRIES", "2"))   # per task, after the first try
MAX_PLAN_STEPS     = int(os.getenv("MAX_PLAN_STEPS", "8"))      # hard cap on decomposition
MAX_PARALLEL_STEPS = int(os.getenv("MAX_PARALLEL_STEPS", "4"))  # concurrent agent workers
AGENT_STEP_TIMEOUT = int(os.getenv("AGENT_STEP_TIMEOUT", "90")) # seconds per step
# Turn the autonomous path off entirely without a redeploy.
AGENT_ORCHESTRATION_ENABLED = (
    os.getenv("AGENT_ORCHESTRATION_ENABLED", "true").lower().strip() not in ("0", "false", "no")
)

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
    # Guard the keys the SELECTED CONFIGURATION actually needs. This used to
    # require NVIDIA_API_KEY, which stopped being the chat provider — so a
    # deploy with no GROQ_API_KEY started cleanly and then failed on every
    # single message, which is the opposite of failing fast.
    #
    # Groq is required unconditionally: it is the last attempt in EVERY role's
    # fallback chain (see llm_provider.plan_attempts), so without it a single
    # NVIDIA outage takes the whole app down. NVIDIA is optional — the app must
    # never depend on a provider whose account entitlement it does not control.
    if not GROQ_API_KEY:
        problems.append(
            "GROQ_API_KEY is required — chat, routing, code and vision all run on it, "
            "and it is the fallback for every other role. "
            "Free key: https://console.groq.com/keys"
        )

    # ...unless NVIDIA has been named explicitly as a role's provider. "auto"
    # does not count: auto means "use it if it is there", which is exactly the
    # case where booting without it is correct.
    explicit_nvidia = [
        name for name, value in (
            ("CHAT_PROVIDER", CHAT_PROVIDER), ("ROUTER_PROVIDER", ROUTER_PROVIDER),
            ("CODE_PROVIDER", CODE_PROVIDER), ("VISION_PROVIDER", VISION_PROVIDER),
            ("PLANNER_PROVIDER", PLANNER_PROVIDER),
            ("ORCHESTRATOR_PROVIDER", ORCHESTRATOR_PROVIDER),
        ) if value == "nvidia"
    ]
    if explicit_nvidia and not NVIDIA_API_KEY:
        problems.append(
            f"NVIDIA_API_KEY is required because {', '.join(explicit_nvidia)}="
            "nvidia. Set the key, or use 'auto' (prefer NVIDIA when present, "
            "fall back to Groq) or 'groq'."
        )

    # Document Q&A needs an embedding provider. With neither key set, uploads
    # and retrieval fail at request time while the app reports healthy — the
    # same silent-failure shape the Groq check above exists to prevent.
    if not JINA_API_KEY and not NVIDIA_API_KEY:
        problems.append(
            "An embedding key is required for document Q&A: set JINA_API_KEY "
            "(preferred — https://jina.ai/embeddings/) or NVIDIA_API_KEY."
        )

    if problems:
        msg = "\n  - ".join(["Production startup blocked:"] + problems)
        print(msg, file=sys.stderr, flush=True)
        raise RuntimeError(msg)


if IS_PRODUCTION:
    _enforce_production_safety()

