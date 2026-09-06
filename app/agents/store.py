"""Persistence for TaskState — the TASK memory tier, made durable.

Writes go to the existing database (one new table, no new infrastructure). The
orchestrator calls `save` at each meaningful transition, so a run that dies
mid-flight still leaves a record of what it planned, what each step produced
and where it stopped.

Two rules the rest of the layer depends on:

* **Persistence can never break a task.** Every function here swallows its own
  errors and rolls back. A task that produced a good answer must not fail
  because a write failed — the answer is the product, the record is bookkeeping.

* **Nothing sensitive is written.** Plans and step outputs are built from
  documents, web pages and model text. The same redaction that guards working
  memory runs again on the way to disk, because a value can enter the state
  through paths that did not go through `write_working` (a step's task text, a
  failure string, the user's own goal).
"""

from __future__ import annotations

import json
from datetime import datetime

from app.agents.memory import redact
from app.agents.state import TaskState
from app.db import SessionLocal

# A single step's output can be thousands of words, and a plan can hold eight
# of them. Cap what reaches the row so one runaway task cannot bloat the table.
_MAX_OUTPUT_CHARS = 20000
_MAX_RESULT_CHARS = 60000

# Statuses a task can still leave. Anything else is finished.
NON_TERMINAL = ("PLANNING", "RUNNING", "WAITING", "EVALUATING", "RETRYING")


def _payload(state: TaskState) -> dict:
    return {
        "goal": redact(state.user_goal)[:4000],
        "plan": json.dumps([s.to_dict() for s in state.plan])[:_MAX_OUTPUT_CHARS],
        "status": state.status,
        "current_step": state.current_step,
        "agent_outputs": json.dumps(
            {str(k): redact(v)[:_MAX_OUTPUT_CHARS] for k, v in state.agent_outputs.items()}
        )[:_MAX_RESULT_CHARS],
        "tool_outputs": json.dumps(
            {str(k): redact(str(v))[:4000] for k, v in state.tool_outputs.items()}
        )[:_MAX_OUTPUT_CHARS],
        "failures": json.dumps([redact(f)[:500] for f in state.failures[:20]])[:_MAX_OUTPUT_CHARS],
        "retry_count": state.retry_count,
        "evaluation": json.dumps(state.evaluation.to_dict()) if state.evaluation else None,
        "final_result": redact(state.final_result)[:_MAX_RESULT_CHARS],
        "updated_at": datetime.utcnow(),
    }


def save(state: TaskState, user_id: int | None) -> None:
    """Upsert the task row. No-op when there is no owner to attribute it to —
    an unowned row could not be read back safely anyway."""
    if user_id is None:
        return
    from app.models import AgentTask

    # The session is opened INSIDE the try: when the database is unreachable —
    # which is the case this whole function exists to survive — it is the
    # factory itself that raises, and a failure there must not reach the caller.
    db = None
    try:
        db = SessionLocal()
        row = db.get(AgentTask, state.task_id)
        if row is None:
            row = AgentTask(task_id=state.task_id, user_id=user_id,
                            chat_id=state.chat_id or None)
            db.add(row)
        for key, value in _payload(state).items():
            setattr(row, key, value)
        db.commit()
    except Exception:
        if db is not None:
            db.rollback()
    finally:
        if db is not None:
            db.close()


def _row_to_dict(row) -> dict:
    def _json(raw, default):
        try:
            return json.loads(raw) if raw else default
        except (json.JSONDecodeError, TypeError):
            return default

    return {
        "task_id": row.task_id,
        "chat_id": row.chat_id,
        "goal": row.goal,
        "status": row.status,
        "current_step": row.current_step,
        "plan": _json(row.plan, []),
        "agent_outputs": _json(row.agent_outputs, {}),
        "tool_outputs": _json(row.tool_outputs, {}),
        "failures": _json(row.failures, []),
        "retry_count": row.retry_count,
        "evaluation": _json(row.evaluation, None),
        "final_result": row.final_result,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }


def get(db, user_id: int, task_id: str) -> dict | None:
    """Fetch ONE task belonging to this user.

    Filtered on user_id as well as the id — a task is never addressable by id
    alone, so knowing or guessing one is not enough to read it.
    """
    from app.models import AgentTask

    row = (
        db.query(AgentTask)
        .filter(AgentTask.task_id == task_id, AgentTask.user_id == user_id)
        .one_or_none()
    )
    return _row_to_dict(row) if row else None


def list_for_user(db, user_id: int, limit: int = 20) -> list[dict]:
    from app.models import AgentTask

    rows = (
        db.query(AgentTask)
        .filter(AgentTask.user_id == user_id)
        .order_by(AgentTask.updated_at.desc())
        .limit(max(1, min(limit, 100)))
        .all()
    )
    # Summary shape: the list view does not need every step's full output, and
    # sending them would make a routine listing enormous.
    return [
        {k: v for k, v in _row_to_dict(r).items()
         if k not in ("agent_outputs", "tool_outputs")}
        for r in rows
    ]


def mark_interrupted_tasks() -> int:
    """Called once at startup: close out tasks the previous process was still
    running when it died.

    The client's stream died with the process, so there is nothing to resume.
    Leaving the rows as RUNNING would show a task that never finishes and never
    fails, which is worse than an honest interrupted state — the partial step
    outputs stay readable either way.
    """
    from app.models import AgentTask

    db = None
    try:
        db = SessionLocal()
        rows = db.query(AgentTask).filter(AgentTask.status.in_(NON_TERMINAL)).all()
        for row in rows:
            row.status = "FAILED"
            row.updated_at = datetime.utcnow()
            try:
                failures = json.loads(row.failures or "[]")
            except (json.JSONDecodeError, TypeError):
                failures = []
            failures.append("Interrupted by a server restart.")
            row.failures = json.dumps(failures[:20])
        db.commit()
        return len(rows)
    except Exception:
        if db is not None:
            db.rollback()
        return 0
    finally:
        if db is not None:
            db.close()
