"""Deep Research — multi-query web research synthesized into a cited report.

Pipeline:
  1. LLM (Groq) turns the question into 3-4 diverse search queries.
  2. Each query runs through the Tavily web search (web_search_service).
  3. LLM synthesizes a thorough markdown report with [n] citations that map
     to the returned sources list.

Everything degrades gracefully: any hard failure returns an empty report so
the route never 500s on upstream flakiness.
"""

import json
import re

from app.services.rag_service import _groq_or_nvidia, MODEL
from app.services.web_search_service import (
    web_search as run_web_search,
    is_search_available,
)

# web_search() formats each result as: "- {title}: {content} (source: {url})"
_RESULT_LINE = re.compile(r"^- (.+?): (.*) \(source: (https?://\S+)\)$")

_QUERY_SYSTEM = (
    "You are a research planner. Given a research question, output a JSON array of "
    "3 to 4 diverse, specific web search queries that together cover the topic from "
    "different angles (core facts, current state, comparisons/alternatives, "
    "criticism or data). Queries must be in clear English. "
    "Output ONLY the JSON array of strings — no prose, no markdown fences."
)

_REPORT_SYSTEM = (
    "You are a meticulous research analyst. Using ONLY the numbered web sources "
    "provided, write a thorough, well-structured markdown research report that "
    "answers the user's question.\n"
    "- Structure it with ## section headings (start with a short overview, end with "
    "a conclusion / takeaways section).\n"
    "- Cite sources inline with bracketed numbers like [1], [2] that refer to the "
    "numbered sources — cite every important claim.\n"
    "- Be factual: never invent specifics the sources don't support. If the sources "
    "conflict or leave gaps, say so explicitly.\n"
    "- Do NOT append a sources/references list — the app renders it separately."
)


def _generate_queries(question: str) -> list:
    """Ask the LLM for 3-4 diverse search queries. Falls back to the question."""
    try:
        resp = _groq_or_nvidia().chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": _QUERY_SYSTEM},
                {"role": "user", "content": (question or "")[:1000]},
            ],
            temperature=0.3,
            max_tokens=250,
        )
        raw = (resp.choices[0].message.content or "").strip()
        start, end = raw.find("["), raw.rfind("]")
        if start != -1 and end != -1:
            raw = raw[start : end + 1]
        data = json.loads(raw)
        queries = [str(q).strip() for q in data if isinstance(q, (str, int, float)) and str(q).strip()]
        if queries:
            return queries[:4]
    except Exception:
        pass
    return [question]


def _parse_search_results(text: str) -> list:
    """Parse web_search()'s formatted string back into structured results."""
    out = []
    for line in (text or "").splitlines():
        m = _RESULT_LINE.match(line.strip())
        if m:
            title, snippet, url = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
            out.append({"title": title or url, "snippet": snippet, "url": url})
    return out


def deep_research(question: str) -> dict:
    """Run the full deep-research pipeline. Returns {report, sources}."""
    try:
        question = (question or "").strip()
        if not question or not is_search_available():
            return {"report": "", "sources": []}

        # 1) Plan diverse queries.
        queries = _generate_queries(question)

        # 2) Search the web for each query; dedupe sources by URL.
        seen_urls = set()
        sources = []
        for q in queries:
            for r in _parse_search_results(run_web_search(q)):
                if r["url"] in seen_urls:
                    continue
                seen_urls.add(r["url"])
                sources.append(r)
        sources = sources[:12]

        if not sources:
            return {"report": "", "sources": []}

        # 3) Synthesize the cited report.
        source_block = "\n\n".join(
            f"[{i + 1}] {s['title']}\nURL: {s['url']}\n{s['snippet'][:900]}"
            for i, s in enumerate(sources)
        )
        resp = _groq_or_nvidia().chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": _REPORT_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"RESEARCH QUESTION: {question}\n\nNUMBERED SOURCES:\n{source_block}"
                    ),
                },
            ],
            temperature=0.3,
            max_tokens=3000,
        )
        report = (resp.choices[0].message.content or "").strip()
        if not report:
            return {"report": "", "sources": []}

        return {
            "report": report,
            "sources": [{"title": s["title"], "url": s["url"]} for s in sources],
        }
    except Exception:
        return {"report": "", "sources": []}
