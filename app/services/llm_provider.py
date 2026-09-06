"""Central LLM provider layer — Groq (primary) + NVIDIA NIM (secondary).

Every LLM call in the app goes through `complete()` or `stream()` here, so a
caller names a ROLE ("chat", "plan", "code", …) and never an API. That is what
lets an agent pick a provider without re-implementing client construction,
model selection, error handling and fallback in each call site.

Three things this layer owns that the old bare-client code did not:

1. **Error classification.** "It failed" is not actionable. A 401 means the
   deployment is misconfigured and retrying is pointless; a 429 means retrying
   later is exactly right. `classify_error` turns a provider exception into one
   of the `ErrorKind` values, and the retry policy is written in terms of those
   rather than in terms of guesses.

2. **A bounded fallback chain.** Each role resolves to an ordered list of
   (provider, model) attempts. The list is finite and built once per call, so
   there is no path on which retrying loops — the chain is walked at most once.

3. **A circuit breaker for dead credentials.** An auth failure is a property of
   the account, not of the request: NVIDIA's `/v1/models` answers 200 for a key
   with no inference entitlement while every inference endpoint answers 403. If
   we merely "tried NVIDIA and fell back" we would pay that round-trip on every
   single request forever. An auth failure disables the provider for a window
   (BREAKER_RETRY_AFTER_SECONDS) rather than permanently, so the cost is one
   probe every 15 minutes — and the day the entitlement is granted the app
   picks the provider back up without needing a redeploy. A retired model is
   condemned on its own; the account it lives on stays usable.

Secrets never appear here in any output. Keys are read from config, held only
inside the SDK client, and every log line and user-facing message is built from
the exception CLASS and status code — never from the provider's response body,
which can echo back the request or the key.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Iterable, Optional

from openai import OpenAI

from app.core.config import (
    CHAT_PROVIDER,
    CODE_MODEL,
    CODE_PROVIDER,
    CRITIC_MODEL,
    FALLBACK_CHAT_MODEL,
    MODEL_OUTPUT_CAPS,
    GROQ_API_KEY,
    MODEL,
    NVIDIA_API_KEY,
    NVIDIA_CODE_MODEL,
    NVIDIA_MODEL,
    NVIDIA_PLAN_MODEL,
    NVIDIA_VISION_MODEL,
    ORCHESTRATOR_PROVIDER,
    PLAN_MODEL,
    PLANNER_PROVIDER,
    ROUTER_MODEL,
    ROUTER_PROVIDER,
    VISION_MODEL,
    VISION_PROVIDER,
)

GROQ = "groq"
NVIDIA = "nvidia"

_GROQ_BASE_URL = "https://api.groq.com/openai/v1"
_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"


# ── Error classification ─────────────────────────────────────────────────────

class ErrorKind:
    AUTH = "auth"                            # 401/403 — wrong or unentitled key
    RATE_LIMIT = "rate_limit"                # 429 — back off, try elsewhere
    TIMEOUT = "timeout"                      # network stall
    MODEL_UNAVAILABLE = "model_unavailable"  # 404/410 — retired or wrong id
    SERVER_ERROR = "server_error"            # 5xx — provider's fault, transient
    INVALID_REQUEST = "invalid_request"      # 400/422 — OUR bug, never retry
    UNKNOWN = "unknown"


# Kinds where trying a different provider/model can plausibly succeed. An
# INVALID_REQUEST is a malformed request of ours and will be malformed for
# every provider, so retrying it just multiplies the latency of a certain
# failure. AUTH is not retried against the SAME provider (the breaker below
# removes it) but the next provider in the chain is still worth trying.
_RETRYABLE = frozenset(
    {ErrorKind.AUTH, ErrorKind.RATE_LIMIT, ErrorKind.TIMEOUT,
     ErrorKind.MODEL_UNAVAILABLE, ErrorKind.SERVER_ERROR, ErrorKind.UNKNOWN}
)


def classify_error(exc: BaseException) -> str:
    """Map a provider exception onto an ErrorKind.

    Prefers the HTTP status, which is unambiguous. Falls back to the exception
    class name — never to the response body, which may contain the request or
    credentials.
    """
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)

    if status in (401, 403):
        return ErrorKind.AUTH
    if status in (404, 410):
        return ErrorKind.MODEL_UNAVAILABLE
    if status == 429:
        return ErrorKind.RATE_LIMIT
    if status in (400, 422):
        return ErrorKind.INVALID_REQUEST
    if isinstance(status, int) and status >= 500:
        return ErrorKind.SERVER_ERROR

    name = exc.__class__.__name__.lower()
    if "timeout" in name:
        return ErrorKind.TIMEOUT
    if "connection" in name or "apiconnection" in name:
        return ErrorKind.SERVER_ERROR
    if "authentication" in name or "permissiondenied" in name:
        return ErrorKind.AUTH
    if "notfound" in name:
        return ErrorKind.MODEL_UNAVAILABLE
    if "ratelimit" in name:
        return ErrorKind.RATE_LIMIT
    if "badrequest" in name or "unprocessable" in name:
        return ErrorKind.INVALID_REQUEST
    return ErrorKind.UNKNOWN


class AllProvidersFailed(RuntimeError):
    """Every attempt in the chain failed. Carries the LAST cause + its kind so
    callers can render an accurate message without inspecting provider guts."""

    def __init__(self, role: str, kind: str, cause: Optional[BaseException]):
        super().__init__(f"All providers failed for role={role} ({kind})")
        self.role = role
        self.kind = kind
        self.cause = cause


# ── Providers ────────────────────────────────────────────────────────────────

# How long a tripped breaker stays open before ONE probe is allowed through.
#
# Not permanent, because the failure it guards against is usually temporary in
# the way that matters: NVIDIA's 403 is an account entitlement, and the day
# billing is fixed the app should start using NVIDIA without needing a
# redeploy. Not short either — the point is to stop paying a dead round-trip on
# every request. One probe per 15 minutes per process is the compromise.
BREAKER_RETRY_AFTER_SECONDS = 900


@dataclass
class _Provider:
    name: str
    client: Optional[OpenAI]
    # Models this provider has failed on permanently (retired ids). Guarded by
    # the module lock; both this and `disabled_until` are process-local.
    dead_models: set = field(default_factory=set)
    disabled_until: float = 0.0
    disabled_reason: str = ""

    @property
    def disabled(self) -> bool:
        return self.disabled_until > time.time()

    @property
    def available(self) -> bool:
        return self.client is not None and not self.disabled


# The OpenAI SDK retries twice by default, INSIDE a single call. That silently
# turns each of our attempts into three, multiplies the timeout we configured by
# three, and does it where our classification, logging and breakers cannot see
# it. Measured: a throttled vision request took 92s against a 30s timeout, and
# the chain looked like one attempt the whole time.
#
# Retrying is this module's job and it is bounded and observable. The SDK's copy
# is turned off so the timeout means what it says.
_SDK_RETRIES = 0

_lock = threading.Lock()

_providers: dict[str, _Provider] = {
    GROQ: _Provider(
        GROQ,
        OpenAI(base_url=_GROQ_BASE_URL, api_key=GROQ_API_KEY, timeout=30.0,
               max_retries=_SDK_RETRIES)
        if GROQ_API_KEY else None,
    ),
    NVIDIA: _Provider(
        NVIDIA,
        # NVIDIA's larger reasoning checkpoints think for a while before the
        # first token, so they get a longer ceiling than Groq's.
        OpenAI(base_url=_NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY, timeout=90.0,
               max_retries=_SDK_RETRIES)
        if NVIDIA_API_KEY else None,
    ),
}


def provider_status() -> dict:
    """Non-secret snapshot for diagnostics and tests. Never includes keys."""
    with _lock:
        return {
            name: {
                "configured": p.client is not None,
                "available": p.available,
                "disabled_reason": p.disabled_reason,
                "retry_in_seconds": max(0, int(p.disabled_until - time.time())),
                "dead_models": sorted(p.dead_models),
            }
            for name, p in _providers.items()
        }


def reset_breakers() -> None:
    """Re-enable every provider. Used by tests, and safe to call at runtime."""
    with _lock:
        for p in _providers.values():
            p.disabled_until = 0.0
            p.disabled_reason = ""
            p.dead_models.clear()


def _trip_breaker(provider: str, model: str, kind: str) -> None:
    """Stop paying for a known-bad attempt on every request.

    An auth failure opens the whole provider for a while: it is a property of
    the account, not of this request, so the next request would fail the same
    way. A retired model condemns that one id — the account is fine and every
    other model it serves still works.
    """
    with _lock:
        p = _providers.get(provider)
        if p is None:
            return
        if kind == ErrorKind.AUTH:
            p.disabled_until = time.time() + BREAKER_RETRY_AFTER_SECONDS
            p.disabled_reason = "credentials rejected"
        elif kind == ErrorKind.MODEL_UNAVAILABLE:
            p.dead_models.add(model)


# ── Role → attempt chain ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class Attempt:
    provider: str
    model: str
    # Caps this attempt's output budget below whatever the caller asked for.
    # Comes from MODEL_OUTPUT_CAPS: some models carry a much smaller per-minute
    # output allowance than the primary, and asking for more is refused outright
    # rather than merely queued.
    max_tokens: Optional[int] = None

    def budget(self, requested: int) -> int:
        return min(requested, self.max_tokens) if self.max_tokens else requested


# role -> (preference, groq model, nvidia model). The preference string is the
# configured *_PROVIDER value: "groq", "nvidia" or "auto".
_ROLES: dict[str, tuple[str, str, str]] = {
    "chat":        (CHAT_PROVIDER, MODEL, NVIDIA_MODEL),
    "router":      (ROUTER_PROVIDER, ROUTER_MODEL, NVIDIA_MODEL),
    "code":        (CODE_PROVIDER, CODE_MODEL, NVIDIA_CODE_MODEL),
    "vision":      (VISION_PROVIDER, VISION_MODEL, NVIDIA_VISION_MODEL),
    "plan":        (PLANNER_PROVIDER, PLAN_MODEL, NVIDIA_PLAN_MODEL),
    "orchestrate": (ORCHESTRATOR_PROVIDER, PLAN_MODEL, NVIDIA_PLAN_MODEL),
    # Self-evaluation: a short JSON verdict, not deliberation. See CRITIC_MODEL.
    "critic":      (ORCHESTRATOR_PROVIDER, CRITIC_MODEL, NVIDIA_PLAN_MODEL),
}


def _order_for(preference: str) -> list[str]:
    if preference == NVIDIA:
        return [NVIDIA, GROQ]
    if preference == GROQ:
        return [GROQ, NVIDIA]
    # "auto" (and anything unrecognised): prefer NVIDIA when it is configured,
    # otherwise Groq. Unrecognised values fall here on purpose — a typo in an
    # env var should degrade to a working default, not take the app down.
    return [NVIDIA, GROQ] if _providers[NVIDIA].client is not None else [GROQ, NVIDIA]


def plan_attempts(role: str, model_override: Optional[str] = None) -> list[Attempt]:
    """Build the finite, ordered attempt chain for a role.

    Skips providers whose breaker has tripped and models known to be retired,
    so a chain never contains an attempt we already know will fail.
    """
    preference, groq_model, nvidia_model = _ROLES.get(role, _ROLES["chat"])
    by_provider = {GROQ: groq_model, NVIDIA: nvidia_model}

    attempts: list[Attempt] = []
    with _lock:
        for name in _order_for(preference):
            p = _providers[name]
            if not p.available:
                continue
            model = model_override if (model_override and name == GROQ) else by_provider[name]
            if model in p.dead_models:
                continue
            attempts.append(Attempt(name, model, max_tokens=MODEL_OUTPUT_CAPS.get(model)))

        # Same-provider second model, so a single bad Groq model still has a way
        # through even when NVIDIA is not configured at all. Vision is excluded:
        # the fallback is text-only and would reject the image.
        if role != "vision":
            g = _providers[GROQ]
            if g.available and FALLBACK_CHAT_MODEL not in g.dead_models:
                extra = Attempt(GROQ, FALLBACK_CHAT_MODEL,
                                max_tokens=MODEL_OUTPUT_CAPS.get(FALLBACK_CHAT_MODEL))
                if not any(a.provider == extra.provider and a.model == extra.model
                           for a in attempts):
                    attempts.append(extra)
    return attempts


# ── Observability ────────────────────────────────────────────────────────────

def log_event(event: str, **fields) -> None:
    """One structured line per notable step: task, agent, provider, model,
    duration, retries, outcome.

    Values are stringified and newline-stripped so a multi-line provider
    message cannot forge extra log records. Nothing here ever receives a key —
    callers pass identifiers and outcomes, and the only exception data admitted
    is the class name and status code.
    """
    parts = [f"event={event}"]
    for k, v in fields.items():
        if v is None:
            continue
        parts.append(f"{k}={str(v).replace(chr(10), ' ')[:200]}")
    print("[agent] " + " ".join(parts), flush=True)


# ── Public API ───────────────────────────────────────────────────────────────

def _call(attempt: Attempt, messages: list, temperature: float, max_tokens: int,
          stream: bool, extra: dict):
    p = _providers[attempt.provider]
    return p.client.chat.completions.create(
        model=attempt.model,
        messages=messages,
        temperature=temperature,
        max_tokens=attempt.budget(max_tokens),
        stream=stream,
        **extra,
    )


def _run_chain(role: str, messages: list, temperature: float, max_tokens: int,
               stream: bool, model_override: Optional[str], task_id: Optional[str],
               extra: dict):
    """Walk the attempt chain once. Returns (result, attempt) or raises."""
    attempts = plan_attempts(role, model_override)
    if not attempts:
        raise AllProvidersFailed(role, ErrorKind.AUTH, None)

    last_exc: Optional[BaseException] = None
    last_kind = ErrorKind.UNKNOWN

    for i, attempt in enumerate(attempts):
        started = time.time()
        try:
            result = _call(attempt, messages, temperature, max_tokens, stream, extra)
            log_event(
                "llm_call", task_id=task_id, role=role, provider=attempt.provider,
                model=attempt.model, retry=i, ok=True,
                ms=int((time.time() - started) * 1000),
            )
            return result, attempt
        except Exception as exc:  # noqa: BLE001 — classified immediately below
            kind = classify_error(exc)
            last_exc, last_kind = exc, kind
            log_event(
                "llm_call", task_id=task_id, role=role, provider=attempt.provider,
                model=attempt.model, retry=i, ok=False, kind=kind,
                error=exc.__class__.__name__,
                ms=int((time.time() - started) * 1000),
            )
            if kind in (ErrorKind.AUTH, ErrorKind.MODEL_UNAVAILABLE):
                _trip_breaker(attempt.provider, attempt.model, kind)
            if kind not in _RETRYABLE:
                break

    raise AllProvidersFailed(role, last_kind, last_exc)


def complete(role: str, messages: list, temperature: float = 0.3,
             max_tokens: int = 1024, model: Optional[str] = None,
             task_id: Optional[str] = None, **extra) -> str:
    """Run a non-streaming completion for `role`. Returns the text content.

    Raises AllProvidersFailed when the whole chain is exhausted — callers that
    prefer to degrade rather than fail should catch it explicitly, so that a
    silent empty string can never be mistaken for a real answer.
    """
    resp, _ = _run_chain(role, messages, temperature, max_tokens, False, model, task_id, extra)
    try:
        return resp.choices[0].message.content or ""
    except (IndexError, AttributeError):
        return ""


def stream(role: str, messages: list, temperature: float = 0.3,
           max_tokens: int = 4096, model: Optional[str] = None,
           task_id: Optional[str] = None, **extra) -> tuple[Iterable, Attempt]:
    """Open a streaming completion. Returns (chunk_iterator, winning_attempt).

    Only the stream OPEN is retried across the chain. Once tokens are flowing a
    failure is mid-answer, and silently restarting on another provider would
    duplicate text the reader has already seen.
    """
    return _run_chain(role, messages, temperature, max_tokens, True, model, task_id, extra)


def friendly_error(kind: str) -> str:
    """A user-facing sentence per failure class. Says enough for an operator to
    know which of these it is, without exposing any provider payload."""
    if kind == ErrorKind.AUTH:
        return "The AI service rejected our credentials. The server needs attention."
    if kind == ErrorKind.MODEL_UNAVAILABLE:
        return "The configured AI model is no longer available. The server needs attention."
    if kind == ErrorKind.RATE_LIMIT:
        return "The AI service is rate-limited right now. Please try again in a moment."
    if kind == ErrorKind.TIMEOUT:
        return "The AI service took too long to respond. Please try again."
    if kind == ErrorKind.SERVER_ERROR:
        return "The AI service is temporarily unavailable. Please try again."
    if kind == ErrorKind.INVALID_REQUEST:
        return "That request could not be processed. Please rephrase and try again."
    return "Failed to start the response. Please try again."
