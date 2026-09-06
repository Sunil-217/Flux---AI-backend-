"""Production startup validation.

Run in SUBPROCESSES, because the check runs at import time and the module is
already imported by the rest of the suite. Each case is a real interpreter
boot with a real environment — the same thing Render does.
"""

import subprocess
import sys

import pytest

BACKEND_DIR = str(__import__("pathlib").Path(__file__).resolve().parent.parent)

_VALID = {
    "APP_ENV": "production",
    "JWT_SECRET": "x" * 40,
    "CORS_ORIGINS": "https://close-ai-ai.vercel.app",
    "DATABASE_URL": "sqlite://",
    "GROQ_API_KEY": "g",
    "JINA_API_KEY": "j",
    "NVIDIA_API_KEY": "",
    # Every *_PROVIDER is cleared so a developer's own .env cannot decide the
    # outcome of a test about configuration.
    "CHAT_PROVIDER": "", "ROUTER_PROVIDER": "", "CODE_PROVIDER": "",
    "VISION_PROVIDER": "", "PLANNER_PROVIDER": "", "ORCHESTRATOR_PROVIDER": "",
}


def boot(**overrides):
    """Import the config in a fresh interpreter. Returns (booted, stderr)."""
    import os

    env = {k: v for k, v in os.environ.items()
           if not k.endswith("_PROVIDER") and k not in _VALID}
    env.update(_VALID)
    env.update({k: str(v) for k, v in overrides.items()})
    p = subprocess.run(
        [sys.executable, "-c", "import app.core.config; print('BOOTED')"],
        env=env, capture_output=True, text=True, cwd=BACKEND_DIR,
    )
    return "BOOTED" in p.stdout, p.stderr


# ── Groq is what the app actually needs ──────────────────────────────────────

def test_a_valid_production_config_boots():
    booted, err = boot()
    assert booted, err


def test_missing_groq_key_aborts_startup():
    """Groq is the last attempt in EVERY role's chain. Booting without it means
    reporting healthy and then failing on every single message."""
    booted, err = boot(GROQ_API_KEY="")
    assert not booted
    assert "GROQ_API_KEY is required" in err


def test_nvidia_is_optional():
    """The app must never depend on a provider whose account entitlement it does
    not control — NVIDIA inference is currently 403 on this very account."""
    assert boot(NVIDIA_API_KEY="")[0]


@pytest.mark.parametrize("var", [
    "CHAT_PROVIDER", "ROUTER_PROVIDER", "CODE_PROVIDER",
    "VISION_PROVIDER", "PLANNER_PROVIDER", "ORCHESTRATOR_PROVIDER",
])
def test_naming_nvidia_explicitly_makes_its_key_required(var):
    booted, err = boot(**{var: "nvidia", "NVIDIA_API_KEY": ""})
    assert not booted
    assert "NVIDIA_API_KEY is required" in err and var in err


@pytest.mark.parametrize("value", ["auto", "groq", ""])
def test_auto_and_groq_do_not_require_the_nvidia_key(value):
    """"auto" means "use it if it is there" — which is exactly the case where
    booting without it is correct."""
    assert boot(PLANNER_PROVIDER=value, NVIDIA_API_KEY="")[0]


# ── Embeddings ───────────────────────────────────────────────────────────────

def test_no_embedding_key_at_all_aborts_startup():
    """Document Q&A would fail at request time while the app reported healthy —
    the same silent-failure shape the Groq check exists to prevent."""
    booted, err = boot(JINA_API_KEY="", NVIDIA_API_KEY="")
    assert not booted
    assert "embedding key" in err


def test_either_embedding_provider_is_enough():
    assert boot(JINA_API_KEY="j", NVIDIA_API_KEY="")[0]
    assert boot(JINA_API_KEY="", NVIDIA_API_KEY="n")[0]


# ── Pre-existing guards still hold ───────────────────────────────────────────

def test_default_jwt_secret_aborts_startup():
    booted, err = boot(JWT_SECRET="dev-secret-change-me-in-production")
    assert not booted and "JWT_SECRET" in err


def test_short_jwt_secret_aborts_startup():
    assert not boot(JWT_SECRET="tooshort")[0]


def test_localhost_cors_aborts_startup():
    booted, err = boot(CORS_ORIGINS="http://localhost:3000,http://127.0.0.1:3000")
    assert not booted and "CORS_ORIGINS" in err


def test_development_never_blocks_startup():
    """The checks are a production guard, not a local-development obstacle."""
    assert boot(APP_ENV="development", GROQ_API_KEY="", JWT_SECRET="short",
                JINA_API_KEY="", NVIDIA_API_KEY="")[0]


def test_the_failure_message_never_echoes_a_key_value():
    booted, err = boot(GROQ_API_KEY="", NVIDIA_API_KEY="nvapi-supersecret-value-000")
    assert not booted
    assert "nvapi-supersecret-value-000" not in err
