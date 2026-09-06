"""The agent registry.

Each agent is a thin adapter over a capability the app already has. Nothing
here reimplements retrieval, search, code answering, translation or image
generation — an agent chooses a tool, shapes a prompt, and hands back text. If
an agent duplicated a pipeline, the two copies would drift and the orchestrated
path would quietly start behaving differently from the direct one.

The RAG agent matters most in that respect: it calls the same
`_retrieve_relevant` with the same similarity threshold and the same
selected-document filter as ordinary chat, and it returns the same refusal
string when nothing clears the bar. There is deliberately no way for a planner
to reach documents through a laxer route.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from app.agents import tools
from app.agents.state import SubTask, TaskState
from app.services.llm_provider import complete

# (output_text, sources)
AgentResult = tuple[str, list]


@dataclass(frozen=True)
class Agent:
    name: str
    description: str
    run: Callable[[TaskState, SubTask], AgentResult]


_AGENTS: dict[str, Agent] = {}


def register(agent: Agent) -> Agent:
    _AGENTS[agent.name] = agent
    return agent


def resolve(name: str) -> Agent | None:
    """Exact-match lookup. A plan naming an unknown agent gets None and the
    step is refused rather than guessed at."""
    return _AGENTS.get(str(name or "").strip().lower())


def names() -> list[str]:
    return sorted(_AGENTS)


def describe_for_planner() -> str:
    return "\n".join(f"- {a.name}: {a.description}" for a in _AGENTS.values())


def _context_prefix(state: TaskState, sub: SubTask) -> str:
    prior = state.dependency_context(sub)
    return f"Results from earlier steps you depend on:\n{prior}\n\n" if prior else ""


# ── Research ─────────────────────────────────────────────────────────────────

def _research(state: TaskState, sub: SubTask) -> AgentResult:
    """Search the live web, then answer strictly from what came back.

    Returns the refusal path (empty output) rather than model knowledge when
    search is unavailable or finds nothing — a research step that silently
    answers from training data is the exact failure the web-off guard exists to
    prevent, and it would be harder to spot inside a multi-step report."""
    if not state.web_enabled:
        return "", []

    results = tools.resolve("web_search").run(sub.task)
    if not results:
        return "", []

    answer = complete(
        "chat",
        [
            {
                "role": "system",
                "content": (
                    "You are a research assistant. Answer the task using ONLY the web "
                    "results provided. Cite the source URL inline for every factual claim. "
                    "If the results do not cover part of the task, say so plainly instead "
                    "of filling the gap from memory."
                ),
            },
            {"role": "user", "content": f"{_context_prefix(state, sub)}TASK: {sub.task}\n\nWEB RESULTS:\n{results}"},
        ],
        temperature=0.2,
        max_tokens=1200,
        task_id=state.task_id,
    )
    sources = [{"content": line[:400], "metadata": {"filename": "web"}}
               for line in results.splitlines() if line.strip().startswith("-")][:5]
    return answer, sources


# ── Documents (RAG) ──────────────────────────────────────────────────────────

def _rag(state: TaskState, sub: SubTask) -> AgentResult:
    """Answer from the user's selected documents, or refuse.

    Empty retrieval is a real answer here: it means the document cannot support
    this step. Handing the question to the model anyway is how a document Q&A
    starts returning pretrained facts wearing a citation.
    """
    from app.services.rag_service import DOC_NOT_FOUND_MESSAGE, _retrieve_relevant
    from app.services.chroma_service import get_or_create_collection

    collection = get_or_create_collection(state.chat_id)
    if collection.count() == 0:
        return "", []

    context, sources = _retrieve_relevant(collection, sub.task, state.active_docs)
    if not context:
        return DOC_NOT_FOUND_MESSAGE, []

    answer = complete(
        "chat",
        [
            {
                "role": "system",
                "content": (
                    "Answer using ONLY the document context provided. Do not use outside "
                    "knowledge. If the context does not contain the answer, reply exactly: "
                    f"{DOC_NOT_FOUND_MESSAGE}"
                ),
            },
            {"role": "user", "content": f"{_context_prefix(state, sub)}TASK: {sub.task}\n\nDOCUMENT CONTEXT:\n{context}"},
        ],
        temperature=0.2,
        max_tokens=1200,
        task_id=state.task_id,
    )
    return answer, sources


# ── Code ─────────────────────────────────────────────────────────────────────

def _code(state: TaskState, sub: SubTask) -> AgentResult:
    from app.services.rag_service import _strip_code_output

    out = complete(
        "code",
        [
            {
                "role": "system",
                "content": (
                    "You are a senior engineer. Produce correct, runnable code for the task. "
                    "Include a short explanation only where it prevents misuse."
                ),
            },
            {"role": "user", "content": f"{_context_prefix(state, sub)}TASK: {sub.task}"},
        ],
        temperature=0.2,
        max_tokens=2000,
        task_id=state.task_id,
    )
    return _strip_code_output(out), []


# ── Analysis / synthesis ─────────────────────────────────────────────────────

def _analyse(state: TaskState, sub: SubTask) -> AgentResult:
    """Reason over what earlier steps produced. Adds no new facts of its own —
    the prompt forbids it, because an analysis step is where invented
    specifics blend most convincingly into real gathered material."""
    prior = state.dependency_context(sub, limit=3500)
    if not prior:
        prior = "(no prior results — say so if the task cannot be done without them)"
    out = complete(
        "chat",
        [
            {
                "role": "system",
                "content": (
                    "You are an analyst. Work ONLY from the material provided. Compare, "
                    "structure and draw conclusions from it. Never introduce facts that are "
                    "not in the material; if something needed is missing, name the gap."
                ),
            },
            {"role": "user", "content": f"TASK: {sub.task}\n\nMATERIAL:\n{prior}"},
        ],
        temperature=0.3,
        max_tokens=1600,
        task_id=state.task_id,
    )
    return out, []


# ── Translation ──────────────────────────────────────────────────────────────

def _translate(state: TaskState, sub: SubTask) -> AgentResult:
    prior = state.dependency_context(sub) or sub.task
    return tools.resolve("translate").run(prior, language=sub.task), []


# ── Image ────────────────────────────────────────────────────────────────────

def _image(state: TaskState, sub: SubTask) -> AgentResult:
    """Generate an image. The data URI is returned as the step output; the
    orchestrator hands it to the client as an image event rather than pasting a
    megabyte of base64 into an answer."""
    return tools.resolve("generate_image").run(sub.task), []


# ── Plain chat ───────────────────────────────────────────────────────────────

def _chat(state: TaskState, sub: SubTask) -> AgentResult:
    from app.services.rag_service import SYSTEM_NORMAL

    out = complete(
        "chat",
        [
            {"role": "system", "content": SYSTEM_NORMAL},
            {"role": "user", "content": f"{_context_prefix(state, sub)}{sub.task}"},
        ],
        temperature=0.3,
        max_tokens=1600,
        task_id=state.task_id,
    )
    return out, []


register(Agent("research", "Search the live web and report findings with source URLs.", _research))
register(Agent("rag", "Answer from the user's uploaded documents only.", _rag))
register(Agent("code", "Write or modify code.", _code))
register(Agent("analyse", "Compare, structure and draw conclusions from earlier step results.", _analyse))
register(Agent("translate", "Translate earlier results into a target language.", _translate))
register(Agent("image", "Generate an image from a description.", _image))
register(Agent("chat", "Answer directly from general knowledge (no external sources).", _chat))
