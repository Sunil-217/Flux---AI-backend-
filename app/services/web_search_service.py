"""
Live web search via Tavily.

Used to answer questions about current / recent events that fall outside
the language model's training cutoff (e.g. "who is the current CSK captain?").
Gracefully returns an empty string if no API key is configured or the
search fails, so the chat flow never breaks.
"""

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

from tavily import TavilyClient

from app.core.config import (
    TAVILY_API_KEY
)

# Hard ceiling on a single web search — Tavily SDK doesn't accept a timeout
# parameter, so we wrap each call in a future and abandon it if it stalls.
# Without this, a slow Tavily can hang stream_question for minutes.
_SEARCH_TIMEOUT_S = 12
_search_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="tavily")


# Only build a client if a key is present — keeps the app working
# (in offline mode) even when web search isn't configured.
_tavily = (
    TavilyClient(api_key=TAVILY_API_KEY)
    if TAVILY_API_KEY
    else None
)


def is_search_available() -> bool:
    """True when a Tavily API key is configured."""

    return _tavily is not None


def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web and return the results as a formatted context string.

    Returns an empty string if search is unavailable or fails, so callers
    can simply fall back to the model's own knowledge.
    """

    if _tavily is None:
        return ""

    try:
        # Bound the call so a slow Tavily can't wedge the chat stream.
        future = _search_pool.submit(
            _tavily.search,
            query=query,
            max_results=max_results,
            search_depth="basic",
        )
        response = future.result(timeout=_SEARCH_TIMEOUT_S)
    except (FuturesTimeout, Exception):
        return ""

    results = response.get("results", [])

    if not results:
        return ""

    lines = []

    # Tavily can return a short direct answer — include it first if present.
    answer = response.get("answer")
    if answer:
        lines.append(f"Summary: {answer}")

    for r in results:
        title = r.get("title", "")
        content = r.get("content", "")
        url = r.get("url", "")
        lines.append(f"- {title}: {content} (source: {url})")

    return "\n".join(lines)
