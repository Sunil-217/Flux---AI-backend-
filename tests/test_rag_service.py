import json

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


def test_stream_question_rag_emits_sources_first(fake_llm, fake_collection):
    fake_collection.count.return_value = 3
    fake_llm["stream_tokens"] = ["A", "B"]
    events = _parse(rag_service.stream_question("c1", "explain", []))
    assert events[0]["type"] == "sources"
    assert len(events[0]["sources"]) == 2
    assert events[-1]["type"] == "done"
    assert [e["type"] for e in events if e["type"] == "token"] == ["token", "token"]


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
