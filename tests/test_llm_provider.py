"""Provider layer: classification, chain order, fallback, breakers, secrecy."""

import time

import pytest

from app.services import llm_provider as lp
from app.services.llm_provider import (
    AllProvidersFailed,
    Attempt,
    ErrorKind,
    classify_error,
    complete,
    plan_attempts,
)


class _HttpError(Exception):
    """Stands in for an SDK error. The real ones expose `status_code`, which is
    the only thing classification is allowed to read from them."""

    def __init__(self, status_code, message="upstream said no"):
        super().__init__(message)
        self.status_code = status_code


@pytest.fixture
def calls(monkeypatch):
    """Record every (provider, model) attempt and script the outcomes."""
    seen = []
    script = {"fail": {}, "text": "ok"}

    def fake_call(attempt, messages, temperature, max_tokens, stream, extra):
        seen.append((attempt.provider, attempt.model))
        exc = script["fail"].get((attempt.provider, attempt.model)) or script["fail"].get(attempt.provider)
        if exc:
            raise exc

        class _M:
            content = script["text"]

        class _C:
            message = _M()

        class _R:
            choices = [_C()]

        return _R()

    monkeypatch.setattr(lp, "_call", fake_call)
    return {"seen": seen, "script": script}


# ── Classification ───────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "status,kind",
    [
        (401, ErrorKind.AUTH),
        (403, ErrorKind.AUTH),
        (404, ErrorKind.MODEL_UNAVAILABLE),
        (410, ErrorKind.MODEL_UNAVAILABLE),
        (429, ErrorKind.RATE_LIMIT),
        (400, ErrorKind.INVALID_REQUEST),
        (422, ErrorKind.INVALID_REQUEST),
        (500, ErrorKind.SERVER_ERROR),
        (503, ErrorKind.SERVER_ERROR),
    ],
)
def test_classify_by_status(status, kind):
    assert classify_error(_HttpError(status)) == kind


def test_classify_falls_back_to_exception_class():
    """No status code — a timeout still has to be told apart from a bad key,
    because one is worth retrying and the other never is."""

    class APITimeoutError(Exception):
        pass

    class AuthenticationError(Exception):
        pass

    assert classify_error(APITimeoutError()) == ErrorKind.TIMEOUT
    assert classify_error(AuthenticationError()) == ErrorKind.AUTH
    assert classify_error(ValueError("who knows")) == ErrorKind.UNKNOWN


# ── Chain construction ───────────────────────────────────────────────────────

def test_groq_first_for_chat_then_nvidia_then_second_groq_model():
    chain = plan_attempts("chat")
    assert chain[0] == Attempt(lp.GROQ, lp.MODEL)
    assert lp.NVIDIA in [a.provider for a in chain]
    assert chain[-1].model == lp.FALLBACK_CHAT_MODEL


def test_plan_role_prefers_nvidia_under_auto():
    """PLANNER_PROVIDER defaults to "auto", and auto means NVIDIA when it is
    configured — the whole point of adding a second provider is that the
    deliberative roles can use it."""
    chain = plan_attempts("plan")
    assert chain[0].provider == lp.NVIDIA
    assert lp.GROQ in [a.provider for a in chain]


def test_vision_chain_has_no_text_only_fallback():
    """A text model handed an image request rejects it. Retrying there converts
    a clean failure into a confusing one.

    Asserted as "no EXTRA Groq attempt is appended" rather than "no attempt uses
    FALLBACK_CHAT_MODEL": today VISION_MODEL and FALLBACK_CHAT_MODEL happen to
    be the same id, so a string comparison would pass for the wrong reason and
    keep passing after the fallback was wrongly re-added.
    """
    vision = plan_attempts("vision")
    chat = plan_attempts("chat")
    assert [a.provider for a in vision].count(lp.GROQ) == 1
    assert len(vision) < len(chat)


def test_unknown_role_and_bad_preference_still_produce_a_chain():
    """A typo in an env var must degrade, not take the app down."""
    assert plan_attempts("no-such-role")
    assert lp._order_for("nonsense")


# ── Fallback behaviour ───────────────────────────────────────────────────────

def test_falls_through_to_the_next_provider_on_rate_limit(calls):
    calls["script"]["fail"][lp.GROQ] = _HttpError(429)
    calls["script"]["text"] = "second provider answered"

    assert complete("chat", [{"role": "user", "content": "hi"}]) == "second provider answered"
    assert [p for p, _ in calls["seen"]][:2] == [lp.GROQ, lp.NVIDIA]


def test_raises_when_every_attempt_fails_and_reports_the_last_kind(calls):
    """Exhausting the chain raises, carrying the FINAL failure's class — that is
    what the user-facing message is built from, so it has to be the one that
    actually ended the attempt rather than whichever failed first."""
    calls["script"]["fail"][(lp.GROQ, lp.MODEL)] = _HttpError(500)
    calls["script"]["fail"][(lp.NVIDIA, lp.NVIDIA_MODEL)] = _HttpError(500)
    calls["script"]["fail"][(lp.GROQ, lp.FALLBACK_CHAT_MODEL)] = _HttpError(429)

    with pytest.raises(AllProvidersFailed) as err:
        complete("chat", [{"role": "user", "content": "hi"}])
    assert len(calls["seen"]) == 3
    assert err.value.kind == ErrorKind.RATE_LIMIT


def test_invalid_request_is_not_retried_anywhere(calls):
    """A 400 is our bug. Every provider will reject it identically, so retrying
    only multiplies the latency of a failure that is already certain."""
    calls["script"]["fail"][lp.GROQ] = _HttpError(400)

    with pytest.raises(AllProvidersFailed) as err:
        complete("chat", [{"role": "user", "content": "hi"}])
    assert err.value.kind == ErrorKind.INVALID_REQUEST
    assert len(calls["seen"]) == 1


def test_chain_is_walked_at_most_once(calls):
    """No configuration may produce an unbounded retry loop."""
    calls["script"]["fail"][lp.GROQ] = _HttpError(500)
    calls["script"]["fail"][lp.NVIDIA] = _HttpError(500)

    with pytest.raises(AllProvidersFailed):
        complete("chat", [{"role": "user", "content": "hi"}])
    assert len(calls["seen"]) == len(set(calls["seen"]))


# ── Circuit breakers ─────────────────────────────────────────────────────────

def test_auth_failure_disables_that_provider_for_later_requests(calls):
    """NVIDIA answers 200 on /v1/models for a key with no inference
    entitlement and 403 on every inference call. Without a breaker the app pays
    that dead round-trip on every single request, forever."""
    calls["script"]["fail"][lp.NVIDIA] = _HttpError(403)

    complete("plan", [{"role": "user", "content": "hi"}])  # nvidia 403 → groq answers
    assert lp.provider_status()[lp.NVIDIA]["available"] is False

    calls["seen"].clear()
    complete("plan", [{"role": "user", "content": "hi"}])
    assert all(p != lp.NVIDIA for p, _ in calls["seen"])


def test_retired_model_is_skipped_but_the_provider_survives(calls):
    """A 410 condemns one model id, not the account. Disabling the whole
    provider would throw away every other model it still serves."""
    calls["script"]["fail"][(lp.GROQ, lp.MODEL)] = _HttpError(410)

    complete("chat", [{"role": "user", "content": "hi"}])
    status = lp.provider_status()[lp.GROQ]
    assert lp.MODEL in status["dead_models"]
    assert status["available"] is True

    calls["seen"].clear()
    complete("chat", [{"role": "user", "content": "hi"}])
    assert all(m != lp.MODEL for _, m in calls["seen"])


def test_all_providers_dead_raises_instead_of_returning_empty(calls):
    """An empty string is indistinguishable from a real (bad) answer. Failure
    has to be loud enough that a caller cannot mistake it for content."""
    lp._trip_breaker(lp.GROQ, "", ErrorKind.AUTH)
    lp._trip_breaker(lp.NVIDIA, "", ErrorKind.AUTH)

    assert plan_attempts("chat") == []
    with pytest.raises(AllProvidersFailed):
        complete("chat", [{"role": "user", "content": "hi"}])


# ── Secrecy ──────────────────────────────────────────────────────────────────

def test_status_and_logs_never_expose_credentials(calls, capsys):
    blob = repr(lp.provider_status())
    assert "test-groq-key" not in blob and "test-nvidia-key" not in blob

    calls["script"]["fail"][lp.GROQ] = _HttpError(401, "invalid api key sk-supersecret-000")
    complete("chat", [{"role": "user", "content": "hi"}])

    out = capsys.readouterr().out
    assert "sk-supersecret-000" not in out       # never echo the provider body
    assert "test-groq-key" not in out
    assert "kind=auth" in out                    # but do say what actually broke


def test_log_event_cannot_be_forged_by_a_multiline_value(capsys):
    """Provider text reaches logs. A newline in it must not be able to fake a
    second, attacker-chosen log record."""
    lp.log_event("x", note="line1\nevent=fake_success ok=True")
    assert len(capsys.readouterr().out.strip().splitlines()) == 1


def test_friendly_error_is_specific_but_leaks_nothing():
    assert "credentials" in lp.friendly_error(ErrorKind.AUTH)
    assert "rate-limited" in lp.friendly_error(ErrorKind.RATE_LIMIT)
    assert "no longer available" in lp.friendly_error(ErrorKind.MODEL_UNAVAILABLE)
    for kind in vars(ErrorKind).values():
        if isinstance(kind, str) and not kind.startswith("_"):
            assert lp.friendly_error(kind).endswith((".", "!"))


# ── Breaker recovery ─────────────────────────────────────────────────────────

def test_a_tripped_breaker_reopens_after_its_window(calls, monkeypatch):
    """The 403 the breaker guards is an account entitlement. The day billing is
    fixed the app should start using the provider again on its own — a breaker
    that never reopens would require a redeploy to notice."""
    monkeypatch.setattr(lp, "BREAKER_RETRY_AFTER_SECONDS", 0.05)
    calls["script"]["fail"][lp.NVIDIA] = _HttpError(403)

    complete("plan", [{"role": "user", "content": "hi"}])
    assert lp.provider_status()[lp.NVIDIA]["available"] is False

    time.sleep(0.06)
    assert lp.provider_status()[lp.NVIDIA]["available"] is True

    calls["seen"].clear()
    del calls["script"]["fail"][lp.NVIDIA]
    complete("plan", [{"role": "user", "content": "hi"}])
    assert calls["seen"][0][0] == lp.NVIDIA


def test_the_window_costs_one_probe_not_one_per_request(calls, monkeypatch):
    """The whole point is to stop paying a dead round-trip on every request."""
    monkeypatch.setattr(lp, "BREAKER_RETRY_AFTER_SECONDS", 60)
    calls["script"]["fail"][lp.NVIDIA] = _HttpError(403)

    for _ in range(5):
        complete("plan", [{"role": "user", "content": "hi"}])

    assert sum(1 for p, _ in calls["seen"] if p == lp.NVIDIA) == 1


def test_status_reports_how_long_until_the_next_probe(calls, monkeypatch):
    monkeypatch.setattr(lp, "BREAKER_RETRY_AFTER_SECONDS", 900)
    calls["script"]["fail"][lp.NVIDIA] = _HttpError(403)
    complete("plan", [{"role": "user", "content": "hi"}])

    st = lp.provider_status()[lp.NVIDIA]
    assert 0 < st["retry_in_seconds"] <= 900
    assert st["disabled_reason"] == "credentials rejected"
    assert lp.provider_status()[lp.GROQ]["retry_in_seconds"] == 0


# ── Per-attempt output budget ────────────────────────────────────────────────

def test_the_fallback_attempt_asks_for_a_budget_its_model_can_serve():
    """Measured against the live account: the fallback model carries an
    output-tokens-per-minute cap of 1,000, so asking it for the normal 4,096 is
    refused outright with "Request too large ... reduce max_tokens" — every
    time, whatever budget remains. The fallback existed, was tried, and could
    never succeed."""
    chain = plan_attempts("chat")
    primary, fallback = chain[0], chain[-1]

    assert primary.max_tokens is None
    assert primary.budget(4096) == 4096
    assert fallback.model == lp.FALLBACK_CHAT_MODEL
    assert fallback.budget(4096) == lp.FALLBACK_MAX_TOKENS
    assert lp.FALLBACK_MAX_TOKENS < 4096


def test_a_budget_cap_never_raises_a_smaller_request():
    """A cap is a ceiling, not a target — a caller asking for 200 gets 200."""
    capped = Attempt(lp.GROQ, "m", max_tokens=900)
    assert capped.budget(200) == 200
    assert capped.budget(4096) == 900


def test_the_cap_is_applied_to_the_actual_call(calls):
    calls["script"]["fail"][(lp.GROQ, lp.MODEL)] = _HttpError(429)
    calls["script"]["fail"][(lp.NVIDIA, lp.NVIDIA_MODEL)] = _HttpError(429)

    sent = []
    original = lp._call

    def spy(attempt, messages, temperature, max_tokens, stream, extra):
        sent.append((attempt.model, attempt.budget(max_tokens)))
        return original(attempt, messages, temperature, max_tokens, stream, extra)

    lp._call = spy
    try:
        complete("chat", [{"role": "user", "content": "hi"}], max_tokens=4096)
    finally:
        lp._call = original

    assert sent[0] == (lp.MODEL, 4096)
    assert sent[-1] == (lp.FALLBACK_CHAT_MODEL, lp.FALLBACK_MAX_TOKENS)
