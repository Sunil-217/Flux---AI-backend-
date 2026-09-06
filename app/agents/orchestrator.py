"""The autonomous orchestration layer.

    USER GOAL → intent → complexity → plan → decomposition → agent + tool
    selection → parallel/sequential execution → aggregation → self-evaluation →
    bounded retry → final answer

This is an AGI-INSPIRED autonomous multi-agent system: a planner, specialised
agents, permissioned tools, four memory tiers, explicit task state, a critic and
bounded recovery. It is not general intelligence and does not claim to be — it
is an orchestration layer over ordinary LLMs, and every one of its decisions is
inspectable in the task state it emits.

Two rules constrain everything below, and both exist because an autonomous path
is exactly where they would otherwise be lost:

**Grounding is not negotiable.** The same guards ordinary chat applies are
applied here, on the same switches, before any planning happens. Web off and no
document means a time-sensitive question is refused, not routed to an agent that
would answer it from training data. A selected document that cannot support a
step produces the refusal string, not a graceful paraphrase of pretrained
knowledge. There is no plan a planner can write that reaches around this.

**A direct image request is never planned.** It short-circuits to the image
agent. A planner handed "draw a dog" will happily produce a thoughtful essay
about dogs, which is precisely the regression the direct image routing exists to
prevent.
"""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

from app.agents import memory, registry, store
from app.agents.critic import correction_note, evaluate
from app.agents.planner import make_plan
from app.agents.state import StepStatus, SubTask, TaskState, TaskStatus
from app.agents.tools import Permission
from app.core.config import (
    AGENT_STEP_TIMEOUT,
    MAX_AGENT_RETRIES,
    MAX_PARALLEL_STEPS,
)
from app.services.llm_provider import AllProvidersFailed, complete, friendly_error, log_event

# Safe, high-level labels. The user sees what the system is DOING, never how it
# is reasoning — a chain of thought shown to the reader is both a leak and a
# distraction, and it is frequently wrong in ways the final answer is not.
_STAGE_LABELS = {
    "planning": "Planning",
    "research": "Researching",
    "rag": "Reading your documents",
    "code": "Writing code",
    "analyse": "Analyzing",
    "translate": "Translating",
    "image": "Creating the image",
    "chat": "Thinking",
    "aggregating": "Putting it together",
    "evaluating": "Verifying",
    "retrying": "Correcting",
    "completed": "Completed",
}

# A direct image request. Kept intentionally narrow — it only has to catch the
# unambiguous cases, because the frontend already routes image intent before a
# message ever reaches chat; this is the backend's own safety net.
_DIRECT_IMAGE = re.compile(
    r"^\s*(?:please\s+)?(?:can you\s+|could you\s+)?"
    r"(?:draw|sketch|paint|generate|create|make|show me|design)\b"
    r".{0,80}?\b(?:image|picture|photo|pic|drawing|painting|artwork|logo|wallpaper|poster)\b",
    re.I,
)


def _sse(payload: dict) -> dict:
    return payload


def _persist(state: TaskState) -> None:
    """Checkpoint the task. Never raises — see store.save; a good answer must
    not be lost because a write failed."""
    store.save(state, state.user_id)


def _status(stage: str, state: TaskState, **extra) -> dict:
    return {
        "type": "status",
        "stage": stage,
        "label": _STAGE_LABELS.get(stage, "Working"),
        "task_id": state.task_id,
        "status": state.status,
        **extra,
    }


# ── Execution ────────────────────────────────────────────────────────────────

def _run_step(state: TaskState, sub: SubTask) -> tuple[int, str, list, str]:
    """Execute one step in a worker thread.

    Returns a tuple rather than mutating shared state. Workers touching
    `state.agent_outputs` directly would be a data race waiting to be
    introduced by the next person who adds a read; returning results and letting
    the single coordinating thread do every write removes the question.
    """
    agent = registry.resolve(sub.agent)
    if agent is None:
        return sub.id, "", [], f"unknown agent: {sub.agent}"
    try:
        output, sources = agent.run(state, sub)
        return sub.id, output or "", sources or [], ""
    except AllProvidersFailed as exc:
        return sub.id, "", [], f"provider unavailable ({exc.kind})"
    except Exception as exc:  # noqa: BLE001 — one step failing must not kill the task
        return sub.id, "", [], exc.__class__.__name__


def _ready_steps(state: TaskState) -> list[SubTask]:
    """Steps whose dependencies have all finished. This is the whole parallelism
    rule: everything ready runs together, everything else waits for its inputs."""
    done = {s.id for s in state.plan if s.status in (StepStatus.DONE, StepStatus.SKIPPED)}
    return [
        s for s in state.plan
        if s.status == StepStatus.PENDING and all(d in done for d in s.depends_on)
    ]


def _execute_plan(state: TaskState):
    """Run the plan wave by wave, yielding a status event per wave.

    A failed step is recorded and its dependents are skipped rather than fed an
    empty input — a step built on a dependency that produced nothing is not a
    degraded result, it is a fabricated one.
    """
    workers = max(1, min(MAX_PARALLEL_STEPS, len(state.plan)))
    # NOT a `with` block: its __exit__ calls shutdown(wait=True), which blocks
    # on any thread still running — so a step we already gave up on at
    # AGENT_STEP_TIMEOUT would still hold the request open for as long as the
    # underlying call took. A hung step must cost its timeout and no more.
    # Abandoned threads finish in the background against their own client
    # timeouts, exactly as the Tavily search pool already does.
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="agent")
    try:
        while True:
            ready = _ready_steps(state)
            if not ready:
                break

            wave = ready[:MAX_PARALLEL_STEPS]
            for sub in wave:
                sub.status = StepStatus.RUNNING
                sub.started_at = time.time()
                sub.attempts += 1
            state.current_step = wave[0].id
            state.status = TaskStatus.RUNNING

            yield _status(wave[0].agent, state, steps=[s.id for s in wave],
                          detail=wave[0].task[:120])

            futures = {pool.submit(_run_step, state, s): s for s in wave}
            for future, sub in futures.items():
                try:
                    _sid, output, sources, error = future.result(timeout=AGENT_STEP_TIMEOUT)
                except FuturesTimeout:
                    output, sources, error = "", [], "timed out"
                except Exception as exc:  # noqa: BLE001
                    output, sources, error = "", [], exc.__class__.__name__

                sub.finished_at = time.time()
                if error or not output.strip():
                    sub.status = StepStatus.FAILED
                    sub.error = error or "produced no output"
                    state.failures.append(f"step {sub.id} ({sub.agent}): {sub.error}")
                else:
                    sub.status = StepStatus.DONE
                    sub.output = output
                    memory.write_working(state, sub.id, output)
                    if sources:
                        sub.sources = sources
                        state.sources.extend(sources)

                log_event("step", task_id=state.task_id, step=sub.id, agent=sub.agent,
                          tool=sub.tool, status=sub.status, ms=sub.duration_ms,
                          attempts=sub.attempts)

            # Checkpoint per wave, not per step: a wave is the unit of progress,
            # and one write per step would multiply DB round-trips for no extra
            # recoverable detail.
            _persist(state)

            # Dependents of a failed step cannot run honestly.
            failed = {s.id for s in state.plan if s.status == StepStatus.FAILED}
            for s in state.plan:
                if s.status == StepStatus.PENDING and any(d in failed for d in s.depends_on):
                    s.status = StepStatus.SKIPPED
                    s.error = "a step it depended on failed"
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


# ── Aggregation ──────────────────────────────────────────────────────────────

_AGGREGATOR_SYSTEM = """You are writing the final answer for the user.

You are given the user's goal and the results of the steps that were run to
satisfy it. Write the answer they asked for — directly, in their own language and
script, well structured, with no meta-commentary about steps, agents or plans.

Rules:
- Use ONLY the step results as your source of facts. Never add specifics
  (numbers, names, dates, prices, quotes) that are not in them.
- If a step failed or produced nothing, say plainly what could not be determined.
  Do not fill the gap from your own knowledge and do not pretend it succeeded.
- Keep any source URLs that the step results cited.
- FINISH. Never stop mid-sentence, mid-table or mid-list. If the material is
  large, write a shorter answer that is COMPLETE rather than a long one that is
  cut off — a truncated report is worse than a brief one.
"""


def _aggregate(state: TaskState, correction: str = "") -> str:
    material = memory.working_summary(state)
    failed = [s for s in state.plan if s.status in (StepStatus.FAILED, StepStatus.SKIPPED)]
    gaps = (
        "\n\nSteps that did not produce a result (report these honestly as gaps):\n"
        + "\n".join(f"- {s.task}: {s.error}" for s in failed)
    ) if failed else ""

    user = f"USER GOAL:\n{state.user_goal[:1500]}\n\nSTEP RESULTS:\n{material}{gaps}"
    if correction:
        user += f"\n\n{correction}"

    return complete(
        "chat",
        [{"role": "system", "content": _AGGREGATOR_SYSTEM}, {"role": "user", "content": user}],
        temperature=0.3,
        # 4096, matching the streaming chat path. 2400 truncated real reports
        # mid-table: the model is a reasoning one, so part of the budget goes on
        # thinking before any output is written. The critic caught it correctly
        # and asked for a retry — which produced another truncated draft under
        # the same ceiling, so the loop could never converge and simply spent
        # the whole retry budget.
        max_tokens=4096,
        task_id=state.task_id,
    )


def _emit_text(text: str, size: int = 90):
    """Hand the finished answer to the client in chunks so it renders
    progressively, like the streaming chat path it sits beside."""
    for i in range(0, len(text), size):
        yield _sse({"type": "token", "content": text[i:i + size]})


# ── Entry point ──────────────────────────────────────────────────────────────

def should_orchestrate(goal: str) -> bool:
    """Cheap pre-check so the caller can skip this layer entirely.

    Mirrors the planner's triage: anything the planner would call simple should
    never have paid for an import, a state object and a status event.
    """
    from app.agents.planner import _fast_triage
    return _fast_triage(goal) != "simple"


def run_task(
    goal: str,
    chat_id: str = "",
    history: list | None = None,
    web_enabled: bool = True,
    active_docs: list | None = None,
    has_documents: bool = False,
    ceiling: str = Permission.READ_ONLY,
    user_id: int | None = None,
):
    """Generator of event dicts for the autonomous path.

    Events: status | sources | image | token | task | done | error.
    The shape is a superset of the /chat stream, so a client that already
    understands chat needs to learn only `status`, `image` and `task`.
    """
    state = TaskState(
        user_goal=(goal or "").strip(),
        chat_id=chat_id or "",
        web_enabled=bool(web_enabled),
        active_docs=list(active_docs or []),
        has_documents=bool(has_documents),
        user_id=user_id,
    )
    history = history or []

    log_event("task_start", task_id=state.task_id, chars=len(state.user_goal),
              web=state.web_enabled, docs=state.has_documents)

    try:
        # ── Grounding guards, applied BEFORE any planning ────────────────────
        # Identical to the ordinary chat path. Running them first means no plan
        # can be written that reaches around them.
        from app.services.rag_service import (
            NO_GROUNDED_SOURCE_MESSAGE,
            _is_conversational,
            _might_need_fresh_info,
        )

        if not state.has_documents and not state.web_enabled and _might_need_fresh_info(state.user_goal):
            yield from _emit_text(NO_GROUNDED_SOURCE_MESSAGE)
            state.status = TaskStatus.COMPLETED
            yield _sse({"type": "done"})
            return

        if _is_conversational(state.user_goal):
            # Small talk is not a task. Say so and let the caller fall through
            # to ordinary chat rather than answering it here.
            yield _sse({"type": "delegate", "reason": "conversational"})
            return

        # ── Direct image intent — never planned ─────────────────────────────
        if _DIRECT_IMAGE.match(state.user_goal):
            state.plan = [SubTask(id=1, agent="image", task=state.user_goal)]
            yield _status("image", state)
            _sid, out, _src, err = _run_step(state, state.plan[0])
            if err or not out:
                yield _sse({"type": "error", "message": "Image generation failed. Please try again."})
                return
            yield _sse({"type": "image", "image": out})
            state.status = TaskStatus.COMPLETED
            yield _sse({"type": "done"})
            return

        # ── Plan ────────────────────────────────────────────────────────────
        state.status = TaskStatus.PLANNING
        yield _status("planning", state)

        complexity, steps = make_plan(
            state.user_goal, history, ceiling, state.task_id,
            has_documents=state.has_documents, web_enabled=state.web_enabled,
        )

        if complexity == "simple" or not steps:
            # Not worth a plan — the ordinary chat path answers this better and
            # faster. Tell the caller instead of running a one-step plan that
            # would only re-implement it.
            yield _sse({"type": "delegate", "reason": "simple"})
            return

        state.plan = steps
        state.status = TaskStatus.RUNNING
        _persist(state)
        yield _status("planning", state, plan=[s.to_dict() for s in steps])

        # ── Execute ─────────────────────────────────────────────────────────
        yield from _execute_plan(state)

        if not state.agent_outputs:
            state.status = TaskStatus.FAILED
            _persist(state)
            log_event("task_end", task_id=state.task_id, status=state.status,
                      ms=state.duration_ms, failures=len(state.failures))
            yield _sse({"type": "error",
                        "message": "I couldn't complete any part of that. Please try again."})
            return

        if state.sources:
            yield _sse({"type": "sources", "sources": state.sources[:8]})

        # ── Aggregate → evaluate → bounded retry ────────────────────────────
        correction = ""
        draft = ""
        previous_issues: set[str] = set()
        while True:
            state.status = TaskStatus.RUNNING
            yield _status("aggregating", state)
            try:
                draft = _aggregate(state, correction)
            except AllProvidersFailed as exc:
                state.status = TaskStatus.FAILED
                state.failures.append(f"aggregation failed ({exc.kind})")
                _persist(state)
                yield _sse({"type": "error", "message": friendly_error(exc.kind)})
                return

            state.status = TaskStatus.EVALUATING
            yield _status("evaluating", state)
            state.evaluation = evaluate(state, draft, memory.working_summary(state))

            if state.evaluation.verdict != "retry" or state.retry_count >= MAX_AGENT_RETRIES:
                break

            # Stop when a round changes nothing. A critic handed the same draft
            # shape will raise the same objections, and spending the rest of the
            # budget to hear them again costs the user tens of seconds for a
            # word-for-word identical outcome. Measured: a 4-step research task
            # burned all three rounds on one unchanging complaint.
            issues = {i.strip().lower() for i in state.evaluation.issues}
            if issues and issues == previous_issues:
                log_event("critic_no_progress", task_id=state.task_id,
                          retries=state.retry_count)
                break
            previous_issues = issues

            state.retry_count += 1
            state.status = TaskStatus.RETRYING
            yield _status("retrying", state, attempt=state.retry_count)
            correction = correction_note(state.evaluation)

        state.final_result = draft
        state.status = TaskStatus.COMPLETED
        _persist(state)

        yield from _emit_text(draft)
        yield _sse({"type": "task", "task": state.to_dict()})
        log_event("task_end", task_id=state.task_id, status=state.status,
                  ms=state.duration_ms, retries=state.retry_count,
                  steps=len(state.plan), failures=len(state.failures),
                  verdict=state.evaluation.verdict if state.evaluation else None)
        yield _sse({"type": "done"})

    except Exception as exc:  # noqa: BLE001 — the endpoint must never 500 mid-stream
        state.status = TaskStatus.FAILED
        state.failures.append(f"unexpected error ({exc.__class__.__name__})")
        _persist(state)
        log_event("task_error", task_id=state.task_id, error=exc.__class__.__name__)
        yield _sse({"type": "error", "message": "Failed to complete that request. Please try again."})
