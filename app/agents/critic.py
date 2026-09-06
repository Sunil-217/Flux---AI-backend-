"""Self-evaluation — does the draft actually answer the goal, from real sources?

The critic is a gate with a budget, not a quality ratchet. Left unbounded, an
evaluator asked "is this good enough?" will nearly always find something to
improve, and the task never finishes. So:

* `MAX_AGENT_RETRIES` bounds the loop, checked by the orchestrator, not here.
* The verdict is three-valued. "retry" is reserved for a draft that is missing
  something a retry could plausibly fix. A draft that is merely improvable
  passes.
* Anything the critic itself cannot do — provider down, unparseable reply —
  passes. A broken critic must not be able to block a finished answer.

Grounding is checked separately from completeness because they fail for
different reasons and only one of them is worth a retry. An ungrounded claim in
a RAG or research task is the failure this whole layer exists to prevent, so it
is reported even when the draft is otherwise complete.
"""

from __future__ import annotations

import json
import re

from app.agents.state import Evaluation, TaskState
from app.services.llm_provider import AllProvidersFailed, complete, log_event

_CRITIC_SYSTEM = """You are a strict evaluator. You do NOT rewrite the answer.

Judge the DRAFT ANSWER against the USER GOAL and the SOURCE MATERIAL.

Output ONLY a JSON object, no prose, no markdown fences:
{"complete": true|false, "grounded": true|false, "issues": ["..."], "verdict": "pass"|"retry"}

- complete: does the draft address every part of the goal that was asked for?
- grounded: is every important factual claim supported by the source material?
  If there is no source material, judge only whether the draft avoids inventing
  specifics (names, numbers, dates, prices) it cannot know.
- issues: at most 3 short, concrete problems. Empty when there are none.
- verdict: "retry" ONLY if a further attempt could realistically fix the issues.
  Style preferences, extra polish and "could be more detailed" are NOT retry
  reasons — those are "pass".

A block marked "[... middle omitted ...]" is an EXCERPT shown to you to fit a
size limit. It is not a defect in the work: do not report it as truncated,
incomplete or cut off, and judge only what you can actually see.
"""


def _extract_json(raw: str) -> dict | None:
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


# Room for a full report plus its sources. Generous on purpose — see _excerpt.
_MAX_DRAFT_CHARS = 12000
_MAX_MATERIAL_CHARS = 8000


def _excerpt(text: str, limit: int) -> str:
    """Fit `text` into `limit` WITHOUT hiding how it ends.

    A plain `text[:limit]` was the single largest source of wasted work in this
    layer. Drafts run past the cap, so the critic was handed a copy that stopped
    mid-sentence and reported — correctly, about the copy — "the draft is cut
    off mid-table". The orchestrator then retried, produced a longer draft, and
    got the same complaint again: three rounds, ~40 seconds, over truncation
    that the evaluation harness itself was introducing.

    Keeping the head and the tail with an explicit marker means the ending is
    always visible, and the marker tells the critic the gap is elision rather
    than a defect in the work.
    """
    text = text or ""
    if len(text) <= limit:
        return text
    head = int(limit * 0.6)
    tail = limit - head
    return (
        text[:head]
        + "\n\n[... middle omitted to fit — this is an excerpt, NOT a truncated answer ...]\n\n"
        + text[-tail:]
    )


def evaluate(state: TaskState, draft: str, source_material: str) -> Evaluation:
    """Judge a draft. Never raises — every internal failure returns a pass."""
    if not (draft or "").strip():
        return Evaluation(complete=False, grounded=True,
                          issues=["The draft answer was empty."], verdict="retry")

    user = (
        f"USER GOAL:\n{state.user_goal[:1500]}\n\n"
        f"SOURCE MATERIAL:\n{_excerpt(source_material, _MAX_MATERIAL_CHARS) or '(none)'}\n\n"
        f"DRAFT ANSWER:\n{_excerpt(draft, _MAX_DRAFT_CHARS)}"
    )

    try:
        raw = complete(
            "critic",
            [{"role": "system", "content": _CRITIC_SYSTEM}, {"role": "user", "content": user}],
            temperature=0.0,
            max_tokens=400,
            task_id=state.task_id,
        )
    except AllProvidersFailed as exc:
        log_event("critic_unavailable", task_id=state.task_id, kind=exc.kind)
        return Evaluation(verdict="pass")

    data = _extract_json(raw)
    if not data:
        log_event("critic_unparseable", task_id=state.task_id)
        return Evaluation(verdict="pass")

    issues = [str(i)[:200] for i in (data.get("issues") or []) if str(i).strip()][:3]
    complete_ = bool(data.get("complete", True))
    grounded = bool(data.get("grounded", True))
    verdict = "retry" if data.get("verdict") == "retry" else "pass"

    # A retry with nothing to act on is a retry that will produce the same
    # draft. Require a stated problem before spending another round.
    if verdict == "retry" and complete_ and grounded and not issues:
        verdict = "pass"

    ev = Evaluation(complete=complete_, grounded=grounded, issues=issues, verdict=verdict)
    log_event("critic", task_id=state.task_id, verdict=ev.verdict,
              complete=complete_, grounded=grounded, issues=len(issues))
    return ev


def correction_note(ev: Evaluation) -> str:
    """The instruction handed to the retry. Names the specific problems so the
    second attempt is a correction rather than a re-roll of the same prompt."""
    if not ev.issues:
        return "The previous attempt was incomplete. Answer the goal fully."
    bullets = "\n".join(f"- {i}" for i in ev.issues)
    return (
        "Your previous draft had these problems:\n"
        f"{bullets}\n"
        "Write a corrected version that fixes them. Do not invent facts that are "
        "not in the material — if something is genuinely missing, say so."
    )
