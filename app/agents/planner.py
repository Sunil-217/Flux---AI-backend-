"""Planner agent — turns a goal into a validated execution plan.

Two stages, and the second is the one that matters.

**Triage** decides whether a goal needs a plan at all. Most messages do not:
"hi", "what is RAG", "write a regex" are one model call, and routing them
through decomposition would add several seconds and several LLM calls to buy
nothing. A cheap keyword pass answers this for the obvious cases and only the
genuinely ambiguous ones cost a model call.

**Validation** treats the model's plan as untrusted input. It is generated from
text that includes the user's message and, indirectly, whatever was retrieved —
so it can name an agent that does not exist, invent a tool, cite a dependency
on a step that was never planned, or describe a cycle. `_validate` resolves
every agent and tool against the registries, drops unknown dependencies,
topologically breaks cycles, and caps the step count. What comes out is
executable by construction; the executor never has to ask whether the plan is
sane.
"""

from __future__ import annotations

import json
import re

from app.agents import registry, tools
from app.agents.memory import short_term_digest
from app.agents.state import SubTask
from app.core.config import MAX_PLAN_STEPS
from app.services.llm_provider import AllProvidersFailed, complete, log_event

# Goals that clearly need several steps. Presence of one of these is not proof
# on its own — it is combined with length below.
_COMPLEX_HINTS = re.compile(
    r"\b(?:compare|comparison|competitors?|research|analys[ei]|analyz[ei]|"
    r"report|summar[iy]|breakdown|step[- ]by[- ]step|plan\b|roadmap|strategy|"
    r"pros and cons|market|survey|benchmark|audit|review of|deep dive|"
    r"investigate|evaluate|assess)\b",
    re.I,
)

# Multi-part phrasing: "X and then Y", "first … then …", enumerations.
_MULTI_PART = re.compile(r"(?:\band then\b|\bafter that\b|\bfirst\b.*\bthen\b|\bfinally\b|;\s*\w)", re.I)

# Things that must NEVER be planned. A greeting decomposed into a research plan
# is absurd; a direct image request turned into an essay about images is the
# specific regression the image-intent routing exists to prevent.
_NEVER_PLAN = re.compile(
    r"^\s*(?:hi|hey|hello|yo|thanks|thank you|thx|ok|okay|cool|nice|bye|"
    r"good morning|good evening|vanakkam|namaste)\b",
    re.I,
)

_PLANNER_SYSTEM = """You are a task planner for a multi-agent AI system.

Break the user's goal into the SMALLEST number of steps that actually completes it.
Never pad a plan: if two steps would use the same agent on the same material, make it one step.

Available agents:
{agents}

Available tools:
{tools}

Rules:
- Output ONLY a JSON object. No prose, no markdown fences.
- Schema: {{"complexity":"simple"|"complex","steps":[{{"id":1,"agent":"<agent name>","task":"<what this step must produce>","depends_on":[<ids>],"tool":"<tool name or null>"}}]}}
- "simple" means one step is enough; still output that one step.
- ids start at 1 and increase.
- depends_on lists ONLY steps whose OUTPUT this step needs. Steps that need nothing
  must have an empty depends_on so they can run in parallel — this is what makes the
  system fast, so do not chain steps that are actually independent.
- Maximum {max_steps} steps.
- Use the "research" agent only for information that must come from the live web.
- Use the "rag" agent only for questions about the user's uploaded documents.
- Use "analyse" for comparing or synthesising what earlier steps produced.
- Steps GATHER and ANALYSE material. Do NOT add a final step that writes up,
  formats or summarises the answer — the system always does that itself once the
  plan finishes, so such a step is a duplicate that only costs the user time.
"""


def _fast_triage(goal: str) -> str | None:
    """Return "simple"/"complex" when it is obvious, else None.

    Deliberately biased towards "simple". A goal wrongly called complex costs
    the user several seconds and several model calls; a goal wrongly called
    simple still gets answered by the normal path, which is what would have
    happened anyway.
    """
    text = (goal or "").strip()
    if not text or _NEVER_PLAN.match(text):
        return "simple"

    words = len(text.split())
    if words <= 8:
        return "simple"

    # Count DISTINCT complexity verbs, not total matches. "Research …, compare
    # …, write a report" is three different asks and is plainly a multi-step
    # goal; "compare these two numbers, then compare them again" is one ask
    # repeated, and counting raw matches would rate it the same.
    hit_words = {m.group(0).lower() for m in _COMPLEX_HINTS.finditer(text)}
    multi = bool(_MULTI_PART.search(text))

    if words >= 12 and (multi or len(hit_words) >= 2 or (hit_words and words >= 15)):
        return "complex"
    if words < 25 and not hit_words and not multi:
        return "simple"
    # Genuinely ambiguous — worth one model call to decide.
    return None


def _extract_json(raw: str) -> dict | None:
    """Pull a JSON object out of a model reply.

    Reasoning models wrap output in <think> blocks and chat models like fences;
    both are stripped before parsing so a correct plan is not thrown away over
    packaging.
    """
    text = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.S)
    text = re.sub(r"```(?:json)?", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start:end + 1])
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, ValueError):
        return None


def _validate(raw_steps: list, ceiling: str) -> list[SubTask]:
    """Turn model-proposed steps into steps that are safe to execute.

    Everything here is a rejection rule, not a repair-by-guessing rule: an
    unresolvable agent drops the step rather than substituting a default,
    because substituting would run something the plan did not ask for.
    """
    steps: list[SubTask] = []
    seen_ids: set[int] = set()
    seen_work: set[tuple[str, str]] = set()

    for item in raw_steps or []:
        if len(steps) >= MAX_PLAN_STEPS:
            break
        if not isinstance(item, dict):
            continue

        agent = registry.resolve(item.get("agent"))
        task = str(item.get("task") or "").strip()
        if agent is None or not task:
            continue

        # The same agent on the same task twice can only produce the same
        # answer at twice the cost — two web searches, two query embeddings,
        # two LLM calls, against an 8,000-token minute. The planner is told not
        # to do this; this makes it impossible rather than discouraged.
        work = (agent.name, " ".join(task.lower().split()))
        if work in seen_work:
            continue
        seen_work.add(work)

        try:
            step_id = int(item.get("id"))
        except (TypeError, ValueError):
            step_id = len(steps) + 1
        if step_id in seen_ids:
            step_id = (max(seen_ids) + 1) if seen_ids else 1
        seen_ids.add(step_id)

        # A tool the plan named must exist AND be permitted at this task's
        # ceiling. An unknown or over-privileged name becomes None — the step
        # still runs, using only its agent's own default capability.
        tool = tools.resolve(item.get("tool"))
        tool_name = tool.name if (tool and tool.allowed_under(ceiling)) else None

        deps_raw = item.get("depends_on")
        deps: list[int] = []
        if isinstance(deps_raw, list):
            for d in deps_raw:
                try:
                    d = int(d)
                except (TypeError, ValueError):
                    continue
                # Only backwards references to steps we actually kept. This
                # single rule makes cycles impossible rather than detected: a
                # dependency can never point at a later or absent step.
                if d in seen_ids and d != step_id:
                    deps.append(d)

        steps.append(SubTask(id=step_id, agent=agent.name, task=task[:600],
                             depends_on=deps, tool=tool_name))

    return steps


def make_plan(goal: str, history: list, ceiling: str, task_id: str,
              has_documents: bool = False, web_enabled: bool = True) -> tuple[str, list[SubTask]]:
    """Return (complexity, steps). "simple" comes back with an empty step list —
    the caller then uses the ordinary chat path, untouched.

    Every failure mode degrades to "simple": no plan, provider down, unparseable
    JSON, or a plan that validated down to nothing. Falling back to the normal
    answer is strictly better than failing the user's message outright.
    """
    verdict = _fast_triage(goal)
    if verdict == "simple":
        return "simple", []

    prompt_context = short_term_digest(history)
    available = (
        f"Documents uploaded to this chat: {'yes' if has_documents else 'no'}. "
        f"Live web search allowed: {'yes' if web_enabled else 'no'}."
    )

    system = _PLANNER_SYSTEM.format(
        agents=registry.describe_for_planner(),
        tools=tools.describe_for_planner(ceiling),
        max_steps=MAX_PLAN_STEPS,
    )
    user = (
        (f"Recent conversation:\n{prompt_context}\n\n" if prompt_context else "")
        + f"{available}\n\nUSER GOAL: {goal[:2000]}"
    )

    try:
        raw = complete(
            "plan",
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.2,
            max_tokens=1200,
            task_id=task_id,
        )
    except AllProvidersFailed as exc:
        log_event("plan_failed", task_id=task_id, kind=exc.kind)
        return "simple", []

    data = _extract_json(raw)
    if not data:
        log_event("plan_unparseable", task_id=task_id, chars=len(raw or ""))
        return "simple", []

    steps = _validate(data.get("steps"), ceiling)
    complexity = "complex" if (data.get("complexity") == "complex" and len(steps) > 1) else "simple"

    if complexity == "simple" or not steps:
        return "simple", []

    log_event("plan_ready", task_id=task_id, steps=len(steps),
              parallel=sum(1 for s in steps if not s.depends_on))
    return "complex", steps
