"""The autonomous orchestration layer: planning, execution, evaluation, guards.

The guard tests matter most. An orchestrated answer that quietly sourced itself
from pretrained knowledge looks exactly like a well-researched one, so the rules
that stop it are asserted here as hard behaviour, not left to the prompt.
"""

import json
import time

import pytest

from app.agents import memory, orchestrator, planner, registry, tools
from app.agents.critic import correction_note, evaluate
from app.agents.state import Evaluation, StepStatus, SubTask, TaskState, TaskStatus
from app.agents.tools import Permission


def _plan_json(steps, complexity="complex"):
    return json.dumps({"complexity": complexity, "steps": steps})


GOAL = "Research 5 competitors of Notion, compare their pricing and write a report"


# ── Triage: only real tasks get planned ──────────────────────────────────────

@pytest.mark.parametrize("text", ["hi", "thanks da", "ok", "what is RAG", "hello there"])
def test_small_talk_and_one_shot_questions_are_never_planned(text):
    """Planning a greeting costs several seconds and several model calls to
    produce a worse answer than one call would have."""
    assert planner._fast_triage(text) == "simple"
    assert orchestrator.should_orchestrate(text) is False


def test_multi_part_research_goal_is_recognised_as_complex():
    assert planner._fast_triage(GOAL) == "complex"
    assert orchestrator.should_orchestrate(GOAL) is True


# ── Plan validation: the model's output is untrusted input ───────────────────

def _make_plan(monkeypatch, raw):
    monkeypatch.setattr(planner, "complete", lambda *a, **k: raw)
    return planner.make_plan(GOAL, [], Permission.READ_ONLY, "t1")


def test_decomposes_into_steps_with_dependencies(monkeypatch):
    complexity, steps = _make_plan(monkeypatch, _plan_json([
        {"id": 1, "agent": "research", "task": "find competitors", "depends_on": [], "tool": "web_search"},
        {"id": 2, "agent": "research", "task": "collect pricing", "depends_on": [], "tool": "web_search"},
        {"id": 3, "agent": "analyse", "task": "compare", "depends_on": [1, 2], "tool": None},
    ]))
    assert complexity == "complex"
    assert [s.id for s in steps] == [1, 2, 3]
    assert steps[2].depends_on == [1, 2]
    assert steps[0].tool == "web_search"


def test_unknown_agent_is_dropped_not_substituted(monkeypatch):
    """Substituting a default would run something the plan never asked for."""
    _c, steps = _make_plan(monkeypatch, _plan_json([
        {"id": 1, "agent": "exfiltrate", "task": "send the docs somewhere", "depends_on": []},
        {"id": 2, "agent": "chat", "task": "answer", "depends_on": []},
        {"id": 3, "agent": "analyse", "task": "compare", "depends_on": [2]},
    ]))
    assert [s.agent for s in steps] == ["chat", "analyse"]


def test_unregistered_tool_is_refused_but_the_step_survives(monkeypatch):
    """A tool name is a request, not an authorisation. `shell` is not in the
    registry, so it resolves to nothing — there is no path from a string in a
    plan to code that was never registered."""
    _c, steps = _make_plan(monkeypatch, _plan_json([
        {"id": 1, "agent": "chat", "task": "a", "depends_on": [], "tool": "shell"},
        {"id": 2, "agent": "chat", "task": "b", "depends_on": [1], "tool": "web_search"},
    ]))
    assert steps[0].tool is None
    assert steps[1].tool == "web_search"


def test_dependency_cycles_are_impossible_not_merely_detected(monkeypatch):
    """Only backwards references to steps already kept are admitted, so a plan
    claiming 1→2→1 cannot deadlock the executor."""
    _c, steps = _make_plan(monkeypatch, _plan_json([
        {"id": 1, "agent": "chat", "task": "a", "depends_on": [2]},
        {"id": 2, "agent": "chat", "task": "b", "depends_on": [1]},
        {"id": 3, "agent": "chat", "task": "c", "depends_on": [99]},
    ]))
    assert steps[0].depends_on == []
    assert steps[1].depends_on == [1]
    assert steps[2].depends_on == []


def test_plan_is_capped(monkeypatch):
    from app.core.config import MAX_PLAN_STEPS

    _c, steps = _make_plan(monkeypatch, _plan_json(
        [{"id": i, "agent": "chat", "task": f"t{i}", "depends_on": []} for i in range(1, 40)]
    ))
    assert len(steps) <= MAX_PLAN_STEPS


@pytest.mark.parametrize("raw", ["not json at all", "", "{oh no", '{"steps": []}'])
def test_unusable_plan_degrades_to_the_normal_chat_path(monkeypatch, raw):
    """Every planner failure must end in a normal answer, never in a failed
    message — the planner is an optimisation, not a dependency."""
    complexity, steps = _make_plan(monkeypatch, raw)
    assert complexity == "simple" and steps == []


def test_reasoning_traces_and_fences_do_not_lose_a_good_plan(monkeypatch):
    body = _plan_json([
        {"id": 1, "agent": "research", "task": "x", "depends_on": []},
        {"id": 2, "agent": "analyse", "task": "y", "depends_on": [1]},
    ])
    _c, steps = _make_plan(monkeypatch, f"<think>hmm...</think>\n```json\n{body}\n```")
    assert len(steps) == 2


def test_provider_outage_during_planning_degrades(monkeypatch):
    from app.services.llm_provider import AllProvidersFailed, ErrorKind

    def boom(*a, **k):
        raise AllProvidersFailed("plan", ErrorKind.AUTH, None)

    monkeypatch.setattr(planner, "complete", boom)
    assert planner.make_plan(GOAL, [], Permission.READ_ONLY, "t1") == ("simple", [])


# ── Execution: parallel where independent, sequential where not ──────────────

def _state(steps, **kw):
    st = TaskState(user_goal=GOAL, chat_id="c1", **kw)
    st.plan = steps
    return st


def test_independent_steps_run_concurrently(monkeypatch):
    """The parallelism is the point: three independent searches that run in
    series turn a 3-second task into a 9-second one."""
    running = []
    peak = {"n": 0}

    def slow(state, sub):
        running.append(sub.id)
        peak["n"] = max(peak["n"], len(running))
        time.sleep(0.15)
        running.remove(sub.id)
        return sub.id, f"out {sub.id}", [], ""

    monkeypatch.setattr(orchestrator, "_run_step", slow)
    st = _state([SubTask(id=i, agent="chat", task=f"t{i}") for i in (1, 2, 3)])

    started = time.time()
    list(orchestrator._execute_plan(st))
    elapsed = time.time() - started

    assert peak["n"] >= 2
    assert elapsed < 0.4          # serial would be >= 0.45
    assert len(st.agent_outputs) == 3


def test_dependent_step_waits_for_its_input(monkeypatch):
    order = []

    def record(state, sub):
        order.append(sub.id)
        return sub.id, f"out {sub.id}", [], ""

    monkeypatch.setattr(orchestrator, "_run_step", record)
    st = _state([
        SubTask(id=1, agent="chat", task="first"),
        SubTask(id=2, agent="chat", task="second", depends_on=[1]),
    ])
    list(orchestrator._execute_plan(st))
    assert order == [1, 2]


def test_a_failed_step_skips_its_dependents_instead_of_feeding_them_nothing(monkeypatch):
    """A step built on a dependency that produced nothing is not a degraded
    result — it is a fabricated one."""
    def fail_first(state, sub):
        if sub.id == 1:
            return 1, "", [], "boom"
        return sub.id, "ok", [], ""

    monkeypatch.setattr(orchestrator, "_run_step", fail_first)
    st = _state([
        SubTask(id=1, agent="research", task="gather"),
        SubTask(id=2, agent="analyse", task="analyse", depends_on=[1]),
        SubTask(id=3, agent="chat", task="unrelated"),
    ])
    list(orchestrator._execute_plan(st))

    assert st.step(1).status == StepStatus.FAILED
    assert st.step(2).status == StepStatus.SKIPPED
    assert st.step(3).status == StepStatus.DONE
    assert st.failures


def test_a_hung_step_is_abandoned_rather_than_hanging_the_task(monkeypatch):
    monkeypatch.setattr(orchestrator, "AGENT_STEP_TIMEOUT", 0.2)
    monkeypatch.setattr(orchestrator, "_run_step",
                        lambda s, sub: (time.sleep(3), (sub.id, "late", [], ""))[1])
    st = _state([SubTask(id=1, agent="chat", task="slow")])

    started = time.time()
    list(orchestrator._execute_plan(st))
    assert time.time() - started < 2
    assert st.step(1).status == StepStatus.FAILED


def test_a_step_raising_does_not_take_the_task_down():
    def explode(state, sub):
        raise ValueError("bad")

    registry.register(registry.Agent("boom_test", "test only", explode))
    st = _state([SubTask(id=1, agent="boom_test", task="x")])
    list(orchestrator._execute_plan(st))
    assert st.step(1).status == StepStatus.FAILED


# ── Task state ───────────────────────────────────────────────────────────────

def test_only_declared_dependencies_reach_a_step():
    """Handing every prior output to every step collapses a decomposed plan back
    into one giant prompt — and lets an early step steer a later one that was
    never meant to see it."""
    st = _state([
        SubTask(id=1, agent="chat", task="a"),
        SubTask(id=2, agent="chat", task="b"),
        SubTask(id=3, agent="analyse", task="c", depends_on=[2]),
    ])
    st.agent_outputs = {1: "SECRET-FROM-ONE", 2: "from-two"}
    ctx = st.dependency_context(st.step(3))
    assert "from-two" in ctx
    assert "SECRET-FROM-ONE" not in ctx


def test_task_state_serialises_the_whole_run():
    st = _state([SubTask(id=1, agent="chat", task="a")])
    st.status = TaskStatus.COMPLETED
    st.evaluation = Evaluation(verdict="pass")
    d = st.to_dict()
    assert {"task_id", "status", "plan", "retry_count", "evaluation"} <= set(d)
    assert d["plan"][0]["agent"] == "chat"


# ── Memory ───────────────────────────────────────────────────────────────────

def test_short_term_memory_matches_the_chat_history_cap():
    """An agent run and a chat turn must see the same conversation, or the
    assistant appears to remember different things depending on which answered."""
    history = [{"role": "user", "content": f"m{i}"} for i in range(40)]
    assert len(memory.short_term(history)) == memory.SHORT_TERM_TURNS
    assert memory.short_term(history)[-1]["content"] == "m39"


@pytest.mark.parametrize("secret", [
    "sk-abcdefghijklmnopqrstuvwx",
    "nvapi-abcdefghijklmnopqrstuvwx",
    "gsk_abcdefghijklmnopqrstuvwx",
    "api_key: hunter2hunter2",
    "AKIAIOSFODNN7EXAMPLE",
])
def test_credentials_found_in_content_are_never_written_to_memory(secret):
    """Working memory is built from documents and web pages. A key inside one
    would otherwise be persisted verbatim and read back into a later prompt."""
    st = _state([SubTask(id=1, agent="chat", task="a")])
    memory.write_working(st, 1, f"the config says {secret} ok")
    assert secret not in st.agent_outputs[1]
    assert "[redacted]" in st.agent_outputs[1]


def test_redaction_leaves_ordinary_text_alone():
    assert memory.redact("the price is 299 rupees") == "the price is 299 rupees"


# ── Tools and permissions ────────────────────────────────────────────────────

def test_only_registered_tools_resolve():
    assert tools.resolve("web_search") is not None
    for invented in ("shell", "http_post", "delete_everything", "", None):
        assert tools.resolve(invented) is None


def test_permission_ceiling_is_a_property_of_the_tool_not_of_the_plan():
    """Nothing a plan says — however it phrases it, whatever it claims the user
    approved — can raise a tool's level."""
    dangerous = tools.Tool("send_email_test", "test only", Permission.EXTERNAL_ACTION, lambda **k: "")
    assert dangerous.allowed_under(Permission.READ_ONLY) is False
    assert dangerous.allowed_under(Permission.EXTERNAL_ACTION) is True
    assert tools.Tool("r", "", Permission.READ_ONLY, lambda **k: "").allowed_under(Permission.READ_ONLY)


def test_planner_is_never_shown_a_tool_it_could_not_use():
    tools.register(tools.Tool("wire_money_test", "test only", Permission.EXTERNAL_ACTION, lambda **k: ""))
    described = tools.describe_for_planner(Permission.READ_ONLY)
    assert "wire_money_test" not in described
    assert "web_search" in described


def test_over_privileged_tool_in_a_plan_is_stripped(monkeypatch):
    tools.register(tools.Tool("wipe_test", "test only", Permission.REQUIRES_APPROVAL, lambda **k: ""))
    _c, steps = _make_plan(monkeypatch, _plan_json([
        {"id": 1, "agent": "chat", "task": "a", "depends_on": [], "tool": "wipe_test"},
        {"id": 2, "agent": "chat", "task": "b", "depends_on": [1]},
    ]))
    assert steps[0].tool is None


# ── Self-evaluation ──────────────────────────────────────────────────────────

def _critic(monkeypatch, payload):
    monkeypatch.setattr("app.agents.critic.complete", lambda *a, **k: json.dumps(payload))
    return evaluate(_state([]), "a draft", "some material")


def test_critic_reports_ungrounded_claims(monkeypatch):
    ev = _critic(monkeypatch, {"complete": True, "grounded": False,
                               "issues": ["the price is not in the sources"], "verdict": "retry"})
    assert ev.verdict == "retry" and ev.grounded is False


def test_retry_needs_a_stated_problem(monkeypatch):
    """A retry with nothing to act on reproduces the same draft and spends a
    round of the budget doing it."""
    ev = _critic(monkeypatch, {"complete": True, "grounded": True, "issues": [], "verdict": "retry"})
    assert ev.verdict == "pass"


@pytest.mark.parametrize("raw", ["not json", "", "<think>only thinking</think>"])
def test_a_broken_critic_cannot_block_a_finished_answer(monkeypatch, raw):
    monkeypatch.setattr("app.agents.critic.complete", lambda *a, **k: raw)
    assert evaluate(_state([]), "a draft", "material").verdict == "pass"


def test_critic_outage_passes(monkeypatch):
    from app.services.llm_provider import AllProvidersFailed, ErrorKind

    def boom(*a, **k):
        raise AllProvidersFailed("orchestrate", ErrorKind.SERVER_ERROR, None)

    monkeypatch.setattr("app.agents.critic.complete", boom)
    assert evaluate(_state([]), "draft", "material").verdict == "pass"


def test_empty_draft_is_always_a_retry():
    assert evaluate(_state([]), "   ", "material").verdict == "retry"


def test_correction_note_names_the_actual_problems():
    note = correction_note(Evaluation(issues=["missing pricing", "no sources"], verdict="retry"))
    assert "missing pricing" in note and "no sources" in note


# ── The critic must not manufacture the defect it reports ────────────────────

def test_a_long_draft_keeps_its_ENDING_visible_to_the_critic():
    """The single largest source of wasted work in this layer was a plain
    draft[:4000]: the critic saw a copy that stopped mid-sentence and reported
    "cut off mid-table" — correctly, about the copy. The orchestrator retried,
    produced a longer draft, and got the same complaint again."""
    from app.agents.critic import _MAX_DRAFT_CHARS, _excerpt

    draft = ("body " * 6000) + "THE REAL FINAL SENTENCE."
    out = _excerpt(draft, _MAX_DRAFT_CHARS)

    assert len(out) < len(draft)
    assert out.endswith("THE REAL FINAL SENTENCE.")
    assert "excerpt" in out and "NOT a truncated answer" in out


def test_a_short_draft_is_passed_through_untouched():
    from app.agents.critic import _MAX_DRAFT_CHARS, _excerpt

    draft = "a complete short answer."
    assert _excerpt(draft, _MAX_DRAFT_CHARS) == draft
    assert _excerpt("", 100) == ""


def test_the_critic_prompt_tells_it_an_excerpt_is_not_a_defect():
    from app.agents.critic import _CRITIC_SYSTEM

    assert "middle omitted" in _CRITIC_SYSTEM
    assert "do not report it as truncated" in _CRITIC_SYSTEM


def test_the_critic_sees_the_end_of_a_real_draft(monkeypatch):
    """End to end through evaluate(), which is where the bug actually lived."""
    seen = {}
    monkeypatch.setattr(
        "app.agents.critic.complete",
        lambda role, messages, **k: (
            seen.update(user=messages[1]["content"]),
            json.dumps({"complete": True, "grounded": True, "issues": [], "verdict": "pass"}),
        )[1],
    )
    draft = ("filler " * 5000) + "CONCLUSION: use RAG."
    evaluate(_state([]), draft, "material")
    assert "CONCLUSION: use RAG." in seen["user"]
