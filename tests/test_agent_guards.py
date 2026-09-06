"""The rules the autonomous path is NOT allowed to relax.

An orchestrated answer that quietly sourced itself from pretrained knowledge
reads exactly like a well-researched one — it is fluent, structured and
confident, and the reader has no way to tell. That is why these are asserted as
behaviour here rather than left to a line in a prompt: a planner is an LLM, and
"the prompt says not to" is not a control.
"""

import json

import pytest

from app.agents import orchestrator, registry
from app.agents.state import SubTask, TaskState
from app.services import rag_service


def _events(**kw):
    kw.setdefault("chat_id", "c1")
    return list(orchestrator.run_task(**kw))


def _text(events):
    return "".join(e.get("content", "") for e in events if e.get("type") == "token")


def _types(events):
    return [e.get("type") for e in events]


# ── Web off + no document: refuse rather than answer from training data ──────

TIME_SENSITIVE = (
    "Who is the current CEO of OpenAI, compare their strategy with the previous "
    "one and write a short report"
)


def test_web_off_and_no_document_refuses_before_any_planning(monkeypatch):
    """The guard runs BEFORE the planner. If it ran after, a plan would already
    exist that reached around it."""
    planned = []
    monkeypatch.setattr(orchestrator, "make_plan",
                        lambda *a, **k: (planned.append(1), ("simple", []))[1])

    events = _events(goal=TIME_SENSITIVE, web_enabled=False, has_documents=False)

    assert rag_service.NO_GROUNDED_SOURCE_MESSAGE in _text(events)
    assert planned == []
    assert _types(events)[-1] == "done"


def test_web_on_lets_the_same_question_through(monkeypatch):
    monkeypatch.setattr(orchestrator, "make_plan", lambda *a, **k: ("simple", []))
    events = _events(goal=TIME_SENSITIVE, web_enabled=True, has_documents=False)
    assert rag_service.NO_GROUNDED_SOURCE_MESSAGE not in _text(events)


def test_small_talk_is_handed_back_to_ordinary_chat():
    """A greeting is not a task. Answering it here would bypass the language
    mirroring and style rules the chat prompt carries."""
    events = _events(goal="hi")
    assert _types(events) == ["delegate"]


# ── A direct image request is never turned into an essay ─────────────────────

@pytest.mark.parametrize("goal", [
    "generate an image of a red sports car in the desert",
    "draw a picture of a cat wearing sunglasses",
    "create an image of a mountain at sunrise please",
])
def test_direct_image_requests_short_circuit_to_the_image_agent(monkeypatch, goal):
    monkeypatch.setattr(orchestrator, "make_plan",
                        lambda *a, **k: pytest.fail("a direct image request must never be planned"))
    monkeypatch.setattr(orchestrator, "_run_step",
                        lambda st, sub: (sub.id, "data:image/png;base64,AAAA", [], "", ""))

    events = _events(goal=goal)
    assert "image" in _types(events)
    assert next(e for e in events if e["type"] == "image")["image"].startswith("data:image/")
    assert _text(events) == ""     # no textual answer instead of the picture


def test_a_question_about_images_is_still_answered_as_text(monkeypatch):
    """"How do image models work" is a question, not a request for a picture."""
    monkeypatch.setattr(orchestrator, "make_plan", lambda *a, **k: ("simple", []))
    events = _events(goal="How does an image generation model actually work internally")
    assert "image" not in _types(events)


def test_image_failure_reports_instead_of_inventing_a_reply(monkeypatch):
    monkeypatch.setattr(orchestrator, "_run_step", lambda st, sub: (sub.id, "", [], "provider down", ""))
    events = _events(goal="draw a picture of a dog")
    assert _types(events) == ["status", "error"]


# ── Document grounding inside the agent path ─────────────────────────────────

def _rag_state(**kw):
    st = TaskState(user_goal="what does the contract say about termination", chat_id="c1", **kw)
    return st


def test_rag_agent_refuses_when_nothing_in_the_document_is_relevant(monkeypatch, fake_collection):
    """Empty retrieval is a real answer: the document cannot support this.
    Handing the question to the model anyway is how document Q&A starts
    returning pretrained facts wearing a citation."""
    fake_collection.count.return_value = 5
    monkeypatch.setattr(rag_service, "_retrieve_relevant", lambda *a, **k: ("", []))
    monkeypatch.setattr(registry, "complete",
                        lambda *a, **k: pytest.fail("the model must not be consulted"))

    out, sources = registry.resolve("rag").run(
        _rag_state(), SubTask(id=1, agent="rag", task="what is the termination clause"))

    assert out == rag_service.DOC_NOT_FOUND_MESSAGE
    assert sources == []


def test_rag_agent_passes_the_document_selection_through(monkeypatch, fake_collection):
    """The agent path must not be a second, laxer way into the same corpus: a
    deselected document has to stay out of reach here too."""
    seen = {}
    fake_collection.count.return_value = 5
    monkeypatch.setattr(rag_service, "_retrieve_relevant",
                        lambda col, q, docs=None: (seen.update(docs=docs), ("chunk", []))[1])
    monkeypatch.setattr(registry, "complete", lambda *a, **k: "answer")

    registry.resolve("rag").run(
        _rag_state(active_docs=["only-this.pdf"]),
        SubTask(id=1, agent="rag", task="q"))

    assert seen["docs"] == ["only-this.pdf"]


def test_rag_agent_uses_the_same_similarity_gate_as_chat(monkeypatch, fake_collection):
    """Asserted by delegation: the agent calls rag_service._retrieve_relevant
    rather than querying Chroma itself, so the threshold cannot drift apart
    from the one ordinary chat enforces."""
    fake_collection.count.return_value = 5
    calls = []
    monkeypatch.setattr(rag_service, "_retrieve_relevant",
                        lambda *a, **k: (calls.append(a), ("ctx", []))[1])
    monkeypatch.setattr(registry, "complete", lambda *a, **k: "answer")

    registry.resolve("rag").run(_rag_state(), SubTask(id=1, agent="rag", task="q"))
    assert len(calls) == 1


# ── Research grounding ───────────────────────────────────────────────────────

def test_research_agent_returns_nothing_when_web_search_is_off(monkeypatch):
    """A research step that answers from training data is the exact failure the
    web-off guard exists to prevent, and it is far harder to spot buried in a
    multi-step report."""
    monkeypatch.setattr(registry, "complete",
                        lambda *a, **k: pytest.fail("the model must not be consulted"))
    st = TaskState(user_goal="g", chat_id="c1", web_enabled=False)
    assert registry.resolve("research").run(st, SubTask(id=1, agent="research", task="q")) == ("", [])


def test_research_agent_returns_nothing_when_the_search_finds_nothing(monkeypatch):
    # Patched at the service, not on the Tool: Tool is a frozen dataclass on
    # purpose, so a permission or an implementation cannot be swapped at runtime.
    monkeypatch.setattr("app.services.web_search_service.is_search_available", lambda: True)
    monkeypatch.setattr("app.services.web_search_service.web_search", lambda *a, **k: "")
    monkeypatch.setattr(registry, "complete",
                        lambda *a, **k: pytest.fail("the model must not be consulted"))

    st = TaskState(user_goal="g", chat_id="c1", web_enabled=True)
    assert registry.resolve("research").run(st, SubTask(id=1, agent="research", task="q")) == ("", [])


# ── Bounded retry ────────────────────────────────────────────────────────────

def test_the_critic_cannot_retry_forever(monkeypatch):
    """An evaluator asked "is this good enough?" will nearly always find
    something. Without a budget the task never finishes."""
    from app.core.config import MAX_AGENT_RETRIES

    monkeypatch.setattr(orchestrator, "make_plan", lambda *a, **k: ("complex", [
        SubTask(id=1, agent="chat", task="a"),
        SubTask(id=2, agent="chat", task="b", depends_on=[1]),
    ]))
    monkeypatch.setattr(orchestrator, "_run_step", lambda st, sub: (sub.id, f"out{sub.id}", [], "", ""))

    aggregations = []
    monkeypatch.setattr(orchestrator, "_aggregate",
                        lambda st, corr="": (aggregations.append(corr), "draft")[1])
    monkeypatch.setattr(
        "app.agents.critic.complete",
        lambda *a, **k: json.dumps({"complete": False, "grounded": False,
                                    "issues": ["never happy"], "verdict": "retry"}),
    )

    events = _events(goal="Research the market, compare vendors and write a full report")

    # The loop terminates and is bounded. Whether it stops at the budget or
    # earlier (see the no-progress guard below) is a separate question — what
    # this test pins is that "never happy" can never mean "never finish".
    assert 1 <= len(aggregations) <= MAX_AGENT_RETRIES + 1
    assert _types(events)[-1] == "done"
    assert "never happy" not in _text(events)     # critic notes are internal


def test_a_correction_round_is_told_what_to_fix(monkeypatch):
    monkeypatch.setattr(orchestrator, "make_plan", lambda *a, **k: ("complex", [
        SubTask(id=1, agent="chat", task="a"),
        SubTask(id=2, agent="chat", task="b", depends_on=[1]),
    ]))
    monkeypatch.setattr(orchestrator, "_run_step", lambda st, sub: (sub.id, f"out{sub.id}", [], "", ""))

    corrections = []
    monkeypatch.setattr(orchestrator, "_aggregate",
                        lambda st, corr="": (corrections.append(corr), "draft")[1])

    verdicts = iter([
        {"complete": False, "grounded": True, "issues": ["pricing is missing"], "verdict": "retry"},
        {"complete": True, "grounded": True, "issues": [], "verdict": "pass"},
    ])
    monkeypatch.setattr("app.agents.critic.complete", lambda *a, **k: json.dumps(next(verdicts)))

    _events(goal="Research the market, compare vendors and write a full report")

    assert corrections[0] == ""
    assert "pricing is missing" in corrections[1]


# ── Failure reporting ────────────────────────────────────────────────────────

def test_a_task_where_every_step_failed_says_so(monkeypatch):
    """Never silently fabricate a result from an empty run."""
    monkeypatch.setattr(orchestrator, "make_plan", lambda *a, **k: ("complex", [
        SubTask(id=1, agent="research", task="a"),
        SubTask(id=2, agent="research", task="b"),
    ]))
    monkeypatch.setattr(orchestrator, "_run_step", lambda st, sub: (sub.id, "", [], "provider down", ""))
    monkeypatch.setattr(orchestrator, "_aggregate",
                        lambda *a, **k: pytest.fail("must not aggregate nothing into an answer"))

    events = _events(goal="Research the market, compare vendors and write a full report")
    assert _types(events)[-1] == "error"


def test_status_events_never_leak_reasoning(monkeypatch):
    """The user sees WHAT is happening, never the model's internal reasoning."""
    monkeypatch.setattr(orchestrator, "make_plan", lambda *a, **k: ("complex", [
        SubTask(id=1, agent="research", task="secret internal step description"),
        SubTask(id=2, agent="analyse", task="b", depends_on=[1]),
    ]))
    monkeypatch.setattr(orchestrator, "_run_step", lambda st, sub: (sub.id, "out", [], "", ""))
    monkeypatch.setattr(orchestrator, "_aggregate", lambda *a, **k: "final answer")
    monkeypatch.setattr("app.agents.critic.complete",
                        lambda *a, **k: json.dumps({"complete": True, "grounded": True,
                                                    "issues": [], "verdict": "pass"}))

    statuses = [e for e in _events(goal="Research the market, compare vendors and write a report")
                if e.get("type") == "status"]
    assert statuses
    assert all(s["label"] in orchestrator._STAGE_LABELS.values() for s in statuses)
    assert all("<think>" not in json.dumps(s) for s in statuses)


def test_a_retry_that_changes_nothing_stops_early(monkeypatch):
    """A critic handed the same draft raises the same objection. Spending the
    rest of the budget to hear it again costs the user tens of seconds for a
    word-for-word identical outcome."""
    monkeypatch.setattr(orchestrator, "make_plan", lambda *a, **k: ("complex", [
        SubTask(id=1, agent="chat", task="a"),
        SubTask(id=2, agent="chat", task="b", depends_on=[1]),
    ]))
    monkeypatch.setattr(orchestrator, "_run_step", lambda st, sub: (sub.id, f"out{sub.id}", [], "", ""))

    rounds = []
    monkeypatch.setattr(orchestrator, "_aggregate",
                        lambda st, corr="": (rounds.append(corr), "identical draft")[1])
    monkeypatch.setattr(
        "app.agents.critic.complete",
        lambda *a, **k: json.dumps({"complete": False, "grounded": True,
                                    "issues": ["Same complaint"], "verdict": "retry"}),
    )

    events = _events(goal="Research the market, compare vendors and write a full report")

    # One retry to see whether it helps, then stop — not the full budget.
    assert len(rounds) == 2
    assert _types(events)[-1] == "done"


def test_different_criticism_each_round_still_uses_the_budget(monkeypatch):
    """The early stop must trigger on REPEATED criticism only — a critic that
    finds something new each time is making progress and deserves its budget."""
    from app.core.config import MAX_AGENT_RETRIES

    monkeypatch.setattr(orchestrator, "make_plan", lambda *a, **k: ("complex", [
        SubTask(id=1, agent="chat", task="a"),
        SubTask(id=2, agent="chat", task="b", depends_on=[1]),
    ]))
    monkeypatch.setattr(orchestrator, "_run_step", lambda st, sub: (sub.id, f"out{sub.id}", [], "", ""))

    rounds = []
    monkeypatch.setattr(orchestrator, "_aggregate",
                        lambda st, corr="": (rounds.append(corr), "draft")[1])

    counter = iter(range(99))
    monkeypatch.setattr(
        "app.agents.critic.complete",
        lambda *a, **k: json.dumps({"complete": False, "grounded": True,
                                    "issues": [f"issue {next(counter)}"], "verdict": "retry"}),
    )

    _events(goal="Research the market, compare vendors and write a full report")
    assert len(rounds) == MAX_AGENT_RETRIES + 1


# ── Rate-limit exhaustion must never become a fabricated answer ──────────────

def test_total_provider_exhaustion_reports_instead_of_inventing(monkeypatch):
    """Groq free tier is 8,000 tokens/minute and 200,000/day, and one agent task
    costs ~10,800. Running out is a NORMAL operating condition here, not an
    edge case — and the one outcome that must never follow from it is a
    confident answer the model made up because no source was reachable."""
    from app.services.llm_provider import AllProvidersFailed, ErrorKind

    monkeypatch.setattr(orchestrator, "make_plan", lambda *a, **k: ("complex", [
        SubTask(id=1, agent="research", task="gather"),
        SubTask(id=2, agent="analyse", task="compare", depends_on=[1]),
    ]))
    monkeypatch.setattr(orchestrator, "_run_step", lambda st, sub: (sub.id, "material", [], "", ""))

    def rate_limited(*a, **k):
        raise AllProvidersFailed("chat", ErrorKind.RATE_LIMIT, None)

    monkeypatch.setattr(orchestrator, "_aggregate", rate_limited)

    events = _events(goal="Research the market, compare vendors and write a full report")

    assert _text(events) == ""                      # no invented answer
    assert _types(events)[-1] == "error"
    assert "rate-limited" in events[-1]["message"]  # and it says why


def test_a_rate_limited_step_is_reported_as_a_gap_not_filled_in(monkeypatch):
    """A step that produced nothing must reach the final answer as a stated gap.
    Quietly substituting model knowledge is how a half-failed research task
    starts reading like a complete one."""
    from app.services.llm_provider import ErrorKind

    monkeypatch.setattr(orchestrator, "make_plan", lambda *a, **k: ("complex", [
        SubTask(id=1, agent="research", task="find competitor pricing"),
        SubTask(id=2, agent="research", task="find competitor features"),
    ]))

    def half_fail(st, sub):
        # _run_step classifies and RETURNS; it never raises to its caller, so a
        # single failing step cannot take the task down with it.
        if sub.id == 1:
            return sub.id, "", [], f"provider unavailable ({ErrorKind.RATE_LIMIT})", ErrorKind.RATE_LIMIT
        return sub.id, "features: A, B", [], "", ""

    monkeypatch.setattr(orchestrator, "_run_step", half_fail)

    seen = {}
    monkeypatch.setattr(orchestrator, "_aggregate",
                        lambda st, corr="": (seen.update(prompt=orchestrator._AGGREGATOR_SYSTEM,
                                                         failures=list(st.failures)), "answer")[1])
    monkeypatch.setattr("app.agents.critic.complete",
                        lambda *a, **k: json.dumps({"complete": True, "grounded": True,
                                                    "issues": [], "verdict": "pass"}))

    _events(goal="Research the market, compare vendors and write a full report")

    assert seen["failures"], "a failed step must be recorded on the task state"
    assert "say plainly what could not be determined" in seen["prompt"]
    assert "do not pretend it succeeded" in seen["prompt"].lower()


def test_a_uniformly_rate_limited_task_says_so(monkeypatch):
    """On a free tier, "rate-limited, try again in a moment" is actionable and
    "please try again" is not — the difference between waiting a minute and
    concluding the feature is broken."""
    from app.services.llm_provider import ErrorKind

    monkeypatch.setattr(orchestrator, "make_plan", lambda *a, **k: ("complex", [
        SubTask(id=1, agent="research", task="a"),
        SubTask(id=2, agent="research", task="b"),
    ]))
    monkeypatch.setattr(orchestrator, "_run_step", lambda st, sub: (
        sub.id, "", [], f"provider unavailable ({ErrorKind.RATE_LIMIT})", ErrorKind.RATE_LIMIT))

    events = _events(goal="Research the market, compare vendors and write a full report")
    assert _types(events)[-1] == "error"
    assert "rate-limited" in events[-1]["message"]


def test_mixed_failure_causes_stay_generic(monkeypatch):
    """Picking one cause out of several would be a guess presented as a fact."""
    from app.services.llm_provider import ErrorKind

    monkeypatch.setattr(orchestrator, "make_plan", lambda *a, **k: ("complex", [
        SubTask(id=1, agent="research", task="a"),
        SubTask(id=2, agent="research", task="b"),
    ]))

    def mixed(st, sub):
        if sub.id == 1:
            return sub.id, "", [], "rate limited", ErrorKind.RATE_LIMIT
        return sub.id, "", [], "timed out", ErrorKind.TIMEOUT

    monkeypatch.setattr(orchestrator, "_run_step", mixed)

    events = _events(goal="Research the market, compare vendors and write a full report")
    assert events[-1]["message"] == "I couldn't complete any part of that. Please try again."


def test_a_non_provider_failure_stays_generic(monkeypatch):
    """A crashed agent is not a provider outage and must not be reported as one."""
    monkeypatch.setattr(orchestrator, "make_plan", lambda *a, **k: ("complex", [
        SubTask(id=1, agent="research", task="a"),
        SubTask(id=2, agent="research", task="b"),
    ]))
    monkeypatch.setattr(orchestrator, "_run_step",
                        lambda st, sub: (sub.id, "", [], "ValueError", ""))

    events = _events(goal="Research the market, compare vendors and write a full report")
    assert events[-1]["message"] == "I couldn't complete any part of that. Please try again."
