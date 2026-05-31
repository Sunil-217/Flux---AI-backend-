import json
from datetime import datetime

from openai import OpenAI

from app.core.config import (
    NVIDIA_API_KEY
)

from app.services.chroma_service import (
    get_or_create_collection
)

from app.services.embedding_service import (
    embedding_model
)

from app.services.web_search_service import (
    web_search,
    is_search_available
)

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)

# Larger model → far more accurate answers. The old 8B model confused
# basic terms (e.g. read "RAG" as the musical "Raga"). 70B fixes that.
MODEL = "meta/llama-3.3-70b-instruct"

# Small + fast model, used only for the cheap yes/no "do we need to search
# the web?" decision so greetings like "hi" stay instant.
ROUTER_MODEL = "meta/llama-3.1-8b-instruct"

ROUTER_SYSTEM = (
    "You are a routing classifier. You do NOT answer questions. "
    "You output ONLY one of two things: the word NO, or a short web search query.\n"
    "Output a search query ONLY when answering correctly needs current, recent, or time-sensitive "
    "information (news, prices, sports rosters/results, weather, events that may have changed after 2023).\n"
    "For greetings, general knowledge, definitions, coding, math, or timeless facts, output exactly: NO\n"
    "Use the conversation so far to resolve references (like 'his', 'that team') into a standalone query.\n"
    "Never answer the question. Output NO or a query, nothing else.\n\n"
    "Q: hi\nA: NO\n"
    "Q: what is RAG in gen ai\nA: NO\n"
    "Q: write a python function to reverse a string\nA: NO\n"
    "Q: what is the capital of France\nA: NO\n"
    "Q: who is the current CSK captain\nA: current Chennai Super Kings captain\n"
    "Q: latest iphone price in india\nA: latest iPhone price India\n"
    "Q: who won the last IPL\nA: most recent IPL winner"
)

SYSTEM_NORMAL = (
    "You are Close AI, a knowledgeable and precise AI assistant. "
    "You are an expert across technology, programming, AI/ML, science, and general knowledge. "
    "When a user mentions a technical term or acronym (such as 'RAG', 'LLM', 'API', 'GAN'), "
    "interpret it in its most common technical meaning unless the context clearly says otherwise "
    "(for example, in an AI/tech context 'RAG' means Retrieval-Augmented Generation, not music). "
    "Give accurate, clear, well-structured answers. If a question is genuinely ambiguous, "
    "briefly state your interpretation and then answer it. Be concise but complete."
)

SYSTEM_RAG = """You are Close AI, a knowledgeable and precise AI assistant with access to an uploaded document.

Guidelines:
- If the user's message is about the document, answer using the context below — accurately and without making things up.
- If the user sends a greeting or a general question unrelated to the document, answer it naturally and helpfully using your own knowledge — do NOT say "no context" or refuse.
- Interpret technical acronyms in their common technical meaning (e.g. 'RAG' = Retrieval-Augmented Generation).
- Be accurate, clear, and concise.

Document Context:
{context}"""


def _needs_web_search(question: str, history: list = []):
    """
    Decide if the question needs live web info.
    Returns an optimized search query string if yes, else None.
    """

    if not is_search_available():
        return None

    try:
        messages = [{"role": "system", "content": ROUTER_SYSTEM}]
        # A little recent history helps resolve references like "his", "that team".
        messages.extend(history[-4:])
        messages.append({"role": "user", "content": question})

        resp = client.chat.completions.create(
            model=ROUTER_MODEL,
            messages=messages,
            temperature=0,
            max_tokens=40
        )

        decision = (
            resp.choices[0].message.content or ""
        ).strip()

        if not decision or decision.upper() == "NO":
            return None

        return decision.strip('"').strip()

    except Exception:
        # On any router failure, fall back to the model's own knowledge.
        return None


def _with_web_context(base_system: str, question: str, history: list) -> str:
    """
    If the question needs current info, run a web search and append the
    results (plus today's date) to the system prompt. Otherwise return it unchanged.
    """

    query = _needs_web_search(question, history)
    if not query:
        return base_system

    results = web_search(query)
    if not results:
        return base_system

    today = datetime.now().strftime("%Y-%m-%d")

    return (
        base_system
        + f"\n\nToday's date is {today}. "
        + "The following are live web search results — prefer them for any current, recent, "
        + "or time-sensitive facts, and don't rely on outdated training knowledge:\n"
        + results
    )


def _normal_chat(question: str, history: list = []) -> dict:
    """No PDF uploaded — behave as a general AI assistant."""

    system_prompt = _with_web_context(SYSTEM_NORMAL, question, history)

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": question})

    completion = client.chat.completions.create(
        model=MODEL,

        messages=messages,

        temperature=0.4,
        max_tokens=900
    )

    return {
        "answer": (
            completion
            .choices[0]
            .message
            .content
        ),
        "sources": []
    }


def _rag_chat(collection, question: str, history: list = []) -> dict:
    """PDF uploaded — answer from document, fallback to general for greetings."""

    # Embed question and retrieve top chunks
    question_embedding = (
        embedding_model
        .encode(question)
        .tolist()
    )

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=3
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    context = "\n\n".join(documents)

    rag_system = SYSTEM_RAG.format(context=context)
    rag_system = _with_web_context(rag_system, question, history)

    messages = [{"role": "system", "content": rag_system}]
    messages.extend(history)
    messages.append({"role": "user", "content": question})

    completion = client.chat.completions.create(
        model=MODEL,

        messages=messages,

        temperature=0.3,
        max_tokens=900
    )

    answer = (
        completion
        .choices[0]
        .message
        .content
    )

    return {
        "answer": answer,

        "sources": [
            {
                "content": documents[i],
                "metadata": metadatas[i]
            }
            for i in range(len(documents))
        ]
    }


def ask_question(
    chat_id: str,
    question: str,
    history: list = []
) -> dict:
    """
    Routes the question based on whether a PDF has been uploaded:
      - Empty collection  →  normal AI chat
      - Has documents     →  RAG document Q&A (with graceful fallback for greetings)
    """

    collection = get_or_create_collection(chat_id)

    if collection.count() == 0:
        return _normal_chat(question, history)

    return _rag_chat(collection, question, history)


# ── Streaming (token-by-token) — makes answers feel instant ─────────────────

def _sse(payload: dict) -> str:
    """Format a Server-Sent Event line."""
    return f"data: {json.dumps(payload)}\n\n"


def _stream_completion(messages: list, temperature: float):
    """Yield SSE 'token' events from a streaming chat completion."""
    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=900,
        stream=True,
    )
    for chunk in stream:
        try:
            delta = chunk.choices[0].delta.content
        except (IndexError, AttributeError):
            delta = None
        if delta:
            yield _sse({"type": "token", "content": delta})


def stream_question(chat_id: str, question: str, history: list = []):
    """
    Generator of SSE events for the /chat endpoint:
      - optional {"type":"sources", ...} (when a PDF is loaded)
      - many       {"type":"token", "content": "..."}
      - final      {"type":"done"}
    Falls back to {"type":"error"} on failure so the UI can react.
    """
    try:
        collection = get_or_create_collection(chat_id)

        if collection.count() == 0:
            system_prompt = _with_web_context(SYSTEM_NORMAL, question, history)
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(history)
            messages.append({"role": "user", "content": question})
            yield from _stream_completion(messages, 0.4)
        else:
            question_embedding = embedding_model.encode(question).tolist()
            results = collection.query(
                query_embeddings=[question_embedding],
                n_results=3,
            )
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            context = "\n\n".join(documents)

            sources = [
                {"content": documents[i], "metadata": metadatas[i]}
                for i in range(len(documents))
            ]
            yield _sse({"type": "sources", "sources": sources})

            rag_system = _with_web_context(
                SYSTEM_RAG.format(context=context), question, history
            )
            messages = [{"role": "system", "content": rag_system}]
            messages.extend(history)
            messages.append({"role": "user", "content": question})
            yield from _stream_completion(messages, 0.3)

        yield _sse({"type": "done"})

    except Exception:
        yield _sse({"type": "error", "message": "Failed to generate a response."})
