import json

import pytest

from app.services import rag_service


def _parse(events):
    return [json.loads(e[len("data: "):].strip()) for e in events]


# ── SSE formatting ──
def test_sse_format():
    out = rag_service._sse({"type": "token", "content": "hi"})
    assert out.startswith("data: ")
    assert out.endswith("\n\n")
    assert json.loads(out[len("data: "):].strip()) == {"type": "token", "content": "hi"}


# ── Web-search router ──
def test_needs_web_search_returns_none_without_key(monkeypatch):
    monkeypatch.setattr(rag_service, "is_search_available", lambda: False)
    assert rag_service._needs_web_search("who is the current CSK captain") is None


def test_needs_web_search_returns_query(monkeypatch, fake_llm):
    monkeypatch.setattr(rag_service, "is_search_available", lambda: True)
    fake_llm["router"] = "current Chennai Super Kings captain"
    result = rag_service._needs_web_search("who is csk captain now")
    assert result == "current Chennai Super Kings captain"


def test_needs_web_search_respects_no(monkeypatch, fake_llm):
    monkeypatch.setattr(rag_service, "is_search_available", lambda: True)
    fake_llm["router"] = "NO"
    assert rag_service._needs_web_search("hi") is None


# ── Routing: normal vs RAG ──
def test_ask_question_routes_to_normal(fake_llm, fake_collection):
    fake_collection.count.return_value = 0
    res = rag_service.ask_question("c1", "hello", [])
    assert res["answer"] == "Hello from the model."
    assert res["sources"] == []


def test_ask_question_routes_to_rag(fake_llm, fake_collection):
    fake_collection.count.return_value = 4
    res = rag_service.ask_question("c1", "what does the doc say", [])
    assert res["answer"] == "Hello from the model."
    assert len(res["sources"]) == 2
    assert res["sources"][0]["content"] == "First chunk."


# ── Streaming ──
def test_stream_question_normal_yields_tokens_then_done(fake_llm, fake_collection):
    fake_collection.count.return_value = 0
    fake_llm["stream_tokens"] = ["Hel", "lo", "!"]
    events = _parse(rag_service.stream_question("c1", "hi", []))
    types = [e["type"] for e in events]
    assert types.count("token") == 3
    assert types[-1] == "done"
    assert "".join(e["content"] for e in events if e["type"] == "token") == "Hello!"


def test_stream_question_falls_back_when_first_provider_cannot_start(
    monkeypatch, fake_llm, fake_collection
):
    fake_collection.count.return_value = 0
    monkeypatch.setattr(rag_service, "groq_client", rag_service.client)
    fake_llm["fail_stream_once"] = True
    fake_llm["stream_tokens"] = ["fallback"]

    events = _parse(rag_service.stream_question("c1", "hi", []))

    assert [e["type"] for e in events if e["type"] == "token"] == ["token"]
    assert events[-1]["type"] == "done"
    stream_calls = [c for c in fake_llm["calls"] if c.get("stream")]
    assert len(stream_calls) == 2
    assert stream_calls[-1]["model"] == rag_service.NVIDIA_CHAT_MODEL


def test_stream_question_rag_emits_sources_first(fake_llm, fake_collection):
    fake_collection.count.return_value = 3
    fake_llm["stream_tokens"] = ["A", "B"]
    events = _parse(rag_service.stream_question("c1", "explain", []))
    assert events[0]["type"] == "sources"
    assert len(events[0]["sources"]) == 2
    assert events[-1]["type"] == "done"
    assert [e["type"] for e in events if e["type"] == "token"] == ["token", "token"]


def test_stream_question_skips_irrelevant_sources(fake_llm, fake_collection):
    """Off-topic question (chunks dissimilar) → no source chips."""
    fake_collection.count.return_value = 3
    fake_collection.query.return_value = {
        "documents": [["unrelated chunk"]],
        "metadatas": [[{"filename": "a.pdf"}]],
        "embeddings": [[[-0.1, -0.2, -0.3]]],  # opposite of the query vector → low similarity
    }
    events = _parse(rag_service.stream_question("c1", "totally off-topic", []))
    assert not any(e["type"] == "sources" for e in events)
    assert events[-1]["type"] == "done"


# ── Fast-path: local time-sensitivity heuristic ──
def test_filename_candidates_include_sanitized_upload_name():
    candidates = rag_service._filename_candidates(["Sunil Gen AI.pdf"])
    assert "Sunil Gen AI.pdf" in candidates
    assert "Sunil_Gen_AI.pdf" in candidates


def test_retrieve_relevant_retries_when_active_doc_filter_is_stale(fake_collection):
    fake_collection.query.side_effect = [
        {"documents": [[]], "metadatas": [[]], "embeddings": [[]]},
        {
            "documents": [["First chunk."]],
            "metadatas": [[{"filename": "Sunil_Gen_AI.pdf"}]],
            "embeddings": [[[0.1, 0.2, 0.3]]],
        },
    ]

    context, sources = rag_service._retrieve_relevant(
        fake_collection, "openings irukka da", ["Sunil Gen AI.pdf"]
    )

    assert context == "First chunk."
    assert len(sources) == 1
    first_where = fake_collection.query.call_args_list[0].kwargs["where"]
    assert "Sunil_Gen_AI.pdf" in first_where["filename"]["$in"]
    assert "where" not in fake_collection.query.call_args_list[1].kwargs


@pytest.mark.parametrize(
    "question,expected",
    [
        ("hi", False),
        ("write a python function to reverse a string", False),
        ("what is RAG in gen ai", False),
        ("what is the capital of France", False),
        ("who is the current CSK captain", True),
        ("latest iPhone price in India", True),
        ("what's the weather today", True),
        ("who won the last IPL", True),
        ("any big news in 2026", True),
    ],
)
def test_might_need_fresh_info(question, expected):
    assert rag_service._might_need_fresh_info(question) is expected


def test_ground_prompt_skips_router_for_normal_question(monkeypatch, fake_llm):
    """Non-time-sensitive questions must NOT trigger the LLM router (keeps it fast)."""
    monkeypatch.setattr(rag_service, "is_search_available", lambda: True)
    out = rag_service._ground_prompt(rag_service.SYSTEM_NORMAL, "hi there", [], "c1")
    # The base system prompt is preserved (today's date is appended for grounding).
    assert out.startswith(rag_service.SYSTEM_NORMAL)
    # The unique marker injected only when live web results are fetched must be absent.
    assert "The following are live web search results" not in out
    assert fake_llm["calls"] == []  # router never called → instant streaming


def test_ground_prompt_invokes_router_for_fresh_question(monkeypatch, fake_llm):
    """Time-sensitive questions still go through the router (correctness preserved)."""
    monkeypatch.setattr(rag_service, "is_search_available", lambda: True)
    fake_llm["router"] = "NO"
    rag_service._ground_prompt(
        rag_service.SYSTEM_NORMAL, "who is the current CSK captain", [], "c1"
    )
    assert any(c.get("model") == rag_service.ROUTER_MODEL for c in fake_llm["calls"])


def test_stream_question_with_image_uses_vision(fake_llm):
    """When an image is attached, the vision model is used."""
    fake_llm["stream_tokens"] = ["A ", "pink ", "square."]
    events = _parse(
        rag_service.stream_question("c1", "what is this", [], image="data:image/png;base64,abc")
    )
    assert any(e["type"] == "token" for e in events)
    assert events[-1]["type"] == "done"
    assert any(c.get("model") == rag_service.VISION_MODEL for c in fake_llm["calls"])


def test_stream_question_history_is_threaded(fake_llm, fake_collection):
    """History messages should be forwarded into the model call."""
    fake_collection.count.return_value = 0
    history = [
        {"role": "user", "content": "my name is Kumar"},
        {"role": "assistant", "content": "Nice to meet you, Kumar."},
    ]
    list(rag_service.stream_question("c1", "what is my name", history))
    # Find the streaming (answer) call and confirm history is present in messages
    stream_calls = [c for c in fake_llm["calls"] if c.get("stream")]
    assert stream_calls, "expected a streaming completion call"
    sent_messages = stream_calls[-1]["messages"]
    contents = [m["content"] for m in sent_messages]
    assert "my name is Kumar" in contents
    assert "what is my name" in contents
