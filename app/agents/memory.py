"""The four memory tiers, mapped onto storage this app already has.

No new database and no new table. Each tier is a named view over something the
app was already keeping, which is the difference between organising memory and
inventing a second source of truth for the same facts:

  SHORT-TERM  the current conversation — the `history` list the client already
              sends with every turn, capped the same way ordinary chat caps it.
  LONG-TERM   durable user facts — the existing `UserMemory` table, the same
              rows the chat route already folds into custom instructions.
  WORKING     intermediate agent results — `TaskState.agent_outputs`, in
              process, discarded when the task ends.
  TASK        the plan, step statuses, failures and retries — `TaskState`.

Nothing here writes a secret. `redact()` runs over every string admitted to
long-term memory, because working memory is built from tool output and model
text, and a document or a web page can contain a key that would otherwise be
persisted verbatim and read back into a later prompt.
"""

from __future__ import annotations

import json
import re

from app.agents.state import TaskState

# Recognisable credential shapes. This is a safety net over content we did not
# author, not an access-control mechanism — the app never puts its own keys in
# a prompt. Patterns are deliberately broad: a false positive costs a redacted
# string in a remembered fact, a false negative persists someone's key.
_SECRET_PATTERNS = [
    re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9_\-]{16,}"),          # OpenAI-style
    re.compile(r"\bnvapi-[A-Za-z0-9_\-]{16,}"),                  # NVIDIA
    re.compile(r"\bgsk_[A-Za-z0-9_\-]{16,}"),                    # Groq
    re.compile(r"\bjina_[A-Za-z0-9_\-]{16,}"),                   # Jina
    re.compile(r"\btvly-[A-Za-z0-9_\-]{16,}"),                   # Tavily
    re.compile(r"\bghp_[A-Za-z0-9]{20,}"),                       # GitHub
    re.compile(r"\bey[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"),  # JWT
    re.compile(r"(?i)\b(?:api[_-]?key|secret|password|passwd|token)\s*[:=]\s*\S{8,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),                         # AWS access key id
]

_REDACTED = "[redacted]"


def redact(text: str) -> str:
    """Replace anything that looks like a credential. Safe on non-strings."""
    out = str(text or "")
    for pattern in _SECRET_PATTERNS:
        out = pattern.sub(_REDACTED, out)
    return out


# ── Short-term ───────────────────────────────────────────────────────────────

# Matches stream_question's cap: 16 messages ≈ 8 turns. Kept identical on
# purpose — an agent run and a chat turn should see the same conversation, or
# the assistant appears to remember different things depending on which path
# answered.
SHORT_TERM_TURNS = 16


def short_term(history: list) -> list:
    """The recent conversation, capped and normalised to role/content dicts."""
    msgs = [
        {"role": str(m.get("role", "user")), "content": str(m.get("content", ""))}
        for m in (history or [])
        if isinstance(m, dict) and m.get("content")
    ]
    return msgs[-SHORT_TERM_TURNS:]


def short_term_digest(history: list, limit: int = 900) -> str:
    """A compact transcript for prompts that take context as text, not as
    messages (the planner and the critic both do)."""
    lines = [f"{m['role']}: {m['content'][:300]}" for m in short_term(history)[-6:]]
    return "\n".join(lines)[:limit]


# ── Long-term ────────────────────────────────────────────────────────────────

def long_term_facts(db, user_id: int, limit: int = 25) -> list[str]:
    """Durable user facts from the existing UserMemory row. Never raises — a
    memory read failing must not take a task down with it."""
    try:
        from app.models import UserMemory

        rec = db.get(UserMemory, user_id)
        if rec is None:
            return []
        facts = json.loads(rec.facts or "[]")
        if not isinstance(facts, list):
            return []
        return [redact(str(f)).strip()[:200] for f in facts if str(f).strip()][:limit]
    except Exception:
        return []


def long_term_block(db, user_id: int) -> str:
    facts = long_term_facts(db, user_id)
    if not facts:
        return ""
    return (
        "\n\nKNOWN USER FACTS (remembered from past chats — use when relevant, "
        "don't recite):\n" + "\n".join(f"- {f}" for f in facts)
    )


# ── Working ──────────────────────────────────────────────────────────────────

def write_working(state: TaskState, step_id: int, output: str) -> None:
    """Record a step's result. Redacted on the way in, so nothing downstream —
    a later prompt, a log line, the final answer — can carry a credential that
    arrived in a document or a web page."""
    state.agent_outputs[step_id] = redact(output)


def working_summary(state: TaskState, limit: int = 4000) -> str:
    """Every completed step's output, in plan order, for the aggregation step."""
    blocks = []
    for sub in state.plan:
        out = state.agent_outputs.get(sub.id)
        if out:
            blocks.append(f"### Step {sub.id} — {sub.task}\n{out}")
    return "\n\n".join(blocks)[:limit]


# ── Task ─────────────────────────────────────────────────────────────────────

def task_snapshot(state: TaskState) -> dict:
    """Serialisable task memory: plan, statuses, failures, retries."""
    return state.to_dict()
