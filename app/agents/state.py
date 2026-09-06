"""Structured task state for the autonomous orchestration layer.

This is deliberately a plain dataclass rather than a framework object. The app
has no orchestration framework today (no LangGraph, no LangChain runtime — only
`langchain-text-splitters`, and even that is bypassed because pulling the
LangChain runtime drags in transformers/torch, which OOM-ed this backend on a
3.75 GB instance). Introducing one to hold six fields would trade a real memory
budget for no capability we don't have here.

The state is process-local and lives for the duration of one request. It is not
a durable job store: a restart loses in-flight tasks, which is the correct
trade for a streaming request whose client is watching it live.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


class TaskStatus:
    PLANNING = "PLANNING"
    RUNNING = "RUNNING"
    WAITING = "WAITING"
    EVALUATING = "EVALUATING"
    RETRYING = "RETRYING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    REQUIRES_APPROVAL = "REQUIRES_APPROVAL"


class StepStatus:
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    REQUIRES_APPROVAL = "REQUIRES_APPROVAL"


@dataclass
class SubTask:
    """One step of a plan. `depends_on` holds the ids of steps whose output this
    step needs; an empty list means it can start immediately, which is what the
    executor uses to decide what may run in parallel."""

    id: int
    agent: str
    task: str
    depends_on: list[int] = field(default_factory=list)
    tool: Optional[str] = None
    status: str = StepStatus.PENDING
    output: str = ""
    error: str = ""
    sources: list = field(default_factory=list)
    attempts: int = 0
    started_at: float = 0.0
    finished_at: float = 0.0

    @property
    def duration_ms(self) -> int:
        if not self.started_at:
            return 0
        end = self.finished_at or time.time()
        return int((end - self.started_at) * 1000)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "agent": self.agent,
            "task": self.task,
            "depends_on": list(self.depends_on),
            "tool": self.tool,
            "status": self.status,
            "attempts": self.attempts,
            "duration_ms": self.duration_ms,
            "error": self.error,
        }


@dataclass
class Evaluation:
    complete: bool = True
    grounded: bool = True
    issues: list[str] = field(default_factory=list)
    verdict: str = "pass"  # "pass" | "retry" | "fail"

    def to_dict(self) -> dict:
        return {
            "complete": self.complete,
            "grounded": self.grounded,
            "verdict": self.verdict,
            "issues": list(self.issues)[:5],
        }


@dataclass
class TaskState:
    user_goal: str
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    chat_id: str = ""
    plan: list[SubTask] = field(default_factory=list)
    current_step: Optional[int] = None
    agent_outputs: dict[int, str] = field(default_factory=dict)
    tool_outputs: dict[str, Any] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    retry_count: int = 0
    evaluation: Optional[Evaluation] = None
    final_result: str = ""
    status: str = TaskStatus.PLANNING
    sources: list = field(default_factory=list)
    started_at: float = field(default_factory=time.time)

    # Grounding switches carried from the request. The orchestrator must obey
    # exactly the same rules as ordinary chat, so they travel with the state
    # rather than being re-derived (and possibly re-derived differently) by
    # each agent.
    web_enabled: bool = True
    active_docs: list = field(default_factory=list)
    has_documents: bool = False

    @property
    def duration_ms(self) -> int:
        return int((time.time() - self.started_at) * 1000)

    def step(self, step_id: int) -> Optional[SubTask]:
        return next((s for s in self.plan if s.id == step_id), None)

    def dependency_context(self, sub: SubTask, limit: int = 2500) -> str:
        """The outputs this step declared a dependency on, as prompt context.

        Only declared dependencies are included. Handing every prior output to
        every step is how a decomposed plan quietly collapses back into one
        giant prompt — and it would let an early step's content steer a later
        step that was never meant to see it.
        """
        blocks = []
        for dep in sub.depends_on:
            out = self.agent_outputs.get(dep)
            if out:
                prior = self.step(dep)
                label = prior.task if prior else f"step {dep}"
                blocks.append(f"[Result of step {dep} — {label}]\n{out[:limit]}")
        return "\n\n".join(blocks)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "user_goal": self.user_goal[:400],
            "status": self.status,
            "current_step": self.current_step,
            "retry_count": self.retry_count,
            "duration_ms": self.duration_ms,
            "plan": [s.to_dict() for s in self.plan],
            "failures": self.failures[:5],
            "evaluation": self.evaluation.to_dict() if self.evaluation else None,
        }
