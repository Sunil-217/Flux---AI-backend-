"""Tool registry with permission boundaries.

The planner is an LLM, and its plan is influenced by text it did not write —
the user's message, retrieved document chunks, web search snippets. So the plan
is treated as a REQUEST, never as an authorisation: a step names a tool by
string, and this registry decides whether that string maps to something the
task is allowed to run.

Two properties make prompt injection unable to escalate:

* **Allowlist, not interpretation.** `resolve()` matches against registered
  names only. A step asking for `shell`, `http_post`, or anything else absent
  from the registry resolves to None and the step is refused. There is no path
  from a string in a document to code that was not registered here.

* **Permission is a property of the tool, not of the plan.** A tool's level is
  fixed at registration. Nothing in a plan — however the plan phrases it, and
  whatever it claims about the user having approved it — can raise it. Levels
  above the task's ceiling stop the task with REQUIRES_APPROVAL and hand the
  decision back to the user.

Every tool here is READ_ONLY today: search, retrieve, translate, generate an
image. The higher levels exist so that the first tool with real-world side
effects is gated by construction rather than by someone remembering to add a
check at the point it is introduced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


class Permission:
    READ_ONLY = "READ_ONLY"              # reads only; no side effects anywhere
    WRITE = "WRITE"                      # mutates our own stored state
    EXTERNAL_ACTION = "EXTERNAL_ACTION"  # acts on a third party (send, pay, post)
    REQUIRES_APPROVAL = "REQUIRES_APPROVAL"  # never runs unattended


_ORDER = {
    Permission.READ_ONLY: 0,
    Permission.WRITE: 1,
    Permission.EXTERNAL_ACTION: 2,
    Permission.REQUIRES_APPROVAL: 3,
}


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    permission: str
    run: Callable[..., str]

    def allowed_under(self, ceiling: str) -> bool:
        return _ORDER[self.permission] <= _ORDER.get(ceiling, 0)


_REGISTRY: dict[str, Tool] = {}


def register(tool: Tool) -> Tool:
    _REGISTRY[tool.name] = tool
    return tool


def resolve(name: Optional[str]) -> Optional[Tool]:
    """Look up a tool by exact registered name. Returns None for anything not
    registered — including a name a plan invented."""
    if not name:
        return None
    return _REGISTRY.get(str(name).strip().lower())


def catalog(ceiling: str = Permission.READ_ONLY) -> list[Tool]:
    """Tools the planner may choose from at this ceiling."""
    return [t for t in _REGISTRY.values() if t.allowed_under(ceiling)]


def describe_for_planner(ceiling: str = Permission.READ_ONLY) -> str:
    """The tool list as it appears in the planner prompt. Only tools at or
    below the ceiling are described — the planner is never shown a capability
    it would not be permitted to use, so it cannot plan around one."""
    return "\n".join(f"- {t.name}: {t.description}" for t in catalog(ceiling))


# ── Built-in tools ───────────────────────────────────────────────────────────
# Each one wraps a service that already exists. The registry adds routing and
# permissions; it does not reimplement any capability.

def _web_search(query: str, **_) -> str:
    from app.services.web_search_service import is_search_available, web_search
    if not is_search_available():
        return ""
    return web_search(query) or ""


def _doc_retrieve(query: str, chat_id: str = "", active_docs: Optional[list] = None, **_) -> str:
    """Vector search over the chat's own documents, honouring the user's
    document selection. Delegates to the existing retrieval so the similarity
    threshold and the selected-document filter are the same ones ordinary RAG
    uses — an agent must not be a second, laxer way into the same corpus."""
    from app.services.chroma_service import get_or_create_collection
    from app.services.rag_service import _retrieve_relevant

    collection = get_or_create_collection(chat_id)
    if collection.count() == 0:
        return ""
    context, _sources = _retrieve_relevant(collection, query, active_docs)
    return context


def _translate(text: str, language: str = "English", **_) -> str:
    from app.services.rag_service import translate_text
    return translate_text(text, language)


def _generate_image(prompt: str, **_) -> str:
    from app.services.generate_service import generate_image_b64
    return generate_image_b64(prompt)


register(Tool(
    "web_search",
    "Search the live web for current or time-sensitive facts. Returns snippets with source URLs.",
    Permission.READ_ONLY,
    _web_search,
))
register(Tool(
    "doc_retrieve",
    "Search the user's uploaded documents for passages relevant to a query.",
    Permission.READ_ONLY,
    _doc_retrieve,
))
register(Tool(
    "translate",
    "Translate text into a target language.",
    Permission.READ_ONLY,
    _translate,
))
register(Tool(
    "generate_image",
    "Generate an image from a text prompt. Returns a data URI.",
    Permission.READ_ONLY,
    _generate_image,
))
