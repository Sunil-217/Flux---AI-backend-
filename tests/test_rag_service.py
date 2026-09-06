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
    fake_llm, fake_collection
):
    """A provider that cannot open the stream must not cost the user the answer.

    The retry now crosses PROVIDERS, not just models: the second attempt runs on
    a different provider entirely, so a Groq outage is survivable instead of
    being retried twice against the same dead endpoint.
    """
    fake_collection.count.return_value = 0
    fake_llm["fail_stream_once"] = True
    fake_llm["stream_tokens"] = ["fallback"]

    events = _parse(rag_service.stream_question("c1", "hi", []))

    assert [e["type"] for e in events if e["type"] == "token"] == ["token"]
    assert events[-1]["type"] == "done"
    stream_calls = [c for c in fake_llm["calls"] if c.get("stream")]
    assert len(stream_calls) == 2
    # First attempt is the configured chat model; the retry is a genuinely
    # different (provider, model) pair rather than the same call repeated.
    assert stream_calls[0]["model"] == rag_service.MODEL
    assert stream_calls[-1]["model"] != stream_calls[0]["model"]


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


def test_retrieve_relevant_matches_upload_sanitized_filename(fake_collection):
    """A display name with spaces still finds chunks stored under the
    underscored upload name — via the filter's candidate list, in ONE query."""
    fake_collection.query.return_value = {
        "documents": [["First chunk."]],
        "metadatas": [[{"filename": "Sunil_Gen_AI.pdf"}]],
        "embeddings": [[[0.1, 0.2, 0.3]]],
    }

    context, sources = rag_service._retrieve_relevant(
        fake_collection, "openings irukka da", ["Sunil Gen AI.pdf"]
    )

    assert context == "First chunk."
    assert len(sources) == 1
    assert fake_collection.query.call_count == 1
    where = fake_collection.query.call_args_list[0].kwargs["where"]
    assert "Sunil_Gen_AI.pdf" in where["filename"]["$in"]


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


# ── Strict document grounding ────────────────────────────────────────────────
# A selected document is a scope boundary: when it cannot answer, the assistant
# must say so rather than fall back on what the model happens to know.

def _irrelevant_chunks(fake_collection):
    """Chunks whose embeddings are orthogonal to the query vector, so cosine
    similarity lands below _RAG_MIN_SIMILARITY."""
    fake_collection.count.return_value = 4
    fake_collection.query.return_value = {
        "documents": [["Resume: worked on payment systems.", "Resume: studied at NIT."]],
        "metadatas": [[{"filename": "Sunil_Gen_AI.pdf"}, {"filename": "Sunil_Gen_AI.pdf"}]],
        # fake embed_query returns [0.1, 0.2, 0.3]; these are orthogonal to it.
        "embeddings": [[[0.3, 0.0, -0.1], [-0.2, 0.1, 0.0]]],
    }


# A. Document selected + web OFF + question answerable from the PDF.
def test_doc_mode_answers_from_document(fake_llm, fake_collection):
    fake_collection.count.return_value = 4
    fake_llm["stream_tokens"] = ["Payment ", "systems."]
    events = _parse(
        rag_service.stream_question(
            "c1", "what did they work on", [], web_search=False, active_docs=["a.pdf"]
        )
    )
    assert [e["type"] for e in events if e["type"] == "token"] == ["token", "token"]
    assert any(e["type"] == "sources" for e in events)
    assert events[-1]["type"] == "done"


# B. Document selected + web OFF + question NOT in the PDF → refuse, no LLM call.
def test_doc_mode_refuses_when_nothing_relevant(fake_llm, fake_collection):
    _irrelevant_chunks(fake_collection)
    events = _parse(
        rag_service.stream_question(
            "c1", "who is cm of tamil nadu", [], web_search=False, active_docs=["Sunil_Gen_AI.pdf"]
        )
    )
    tokens = [e["content"] for e in events if e["type"] == "token"]
    assert tokens == [rag_service.DOC_NOT_FOUND_MESSAGE]
    assert events[-1]["type"] == "done"
    # No source chips for a question the document cannot answer.
    assert not any(e["type"] == "sources" for e in events)
    # The guard must run BEFORE the model — no streaming call at all.
    assert not [c for c in fake_llm["calls"] if c.get("stream")]


# C. Document selected + web ON + question not in the PDF → web path still allowed.
def test_doc_mode_with_web_on_still_reaches_the_model(fake_llm, fake_collection, monkeypatch):
    _irrelevant_chunks(fake_collection)
    monkeypatch.setattr(rag_service, "is_search_available", lambda: True)
    monkeypatch.setattr(rag_service, "run_web_search", lambda q: "M.K. Stalin is the CM.")
    fake_llm["router"] = "current chief minister of Tamil Nadu"
    fake_llm["stream_tokens"] = ["M.K. ", "Stalin."]
    events = _parse(
        rag_service.stream_question(
            "c1", "who is cm of tamil nadu", [], web_search=True, active_docs=["Sunil_Gen_AI.pdf"]
        )
    )
    tokens = [e["content"] for e in events if e["type"] == "token"]
    assert "".join(tokens) == "M.K. Stalin."
    assert [c for c in fake_llm["calls"] if c.get("stream")]


# D. No document + web OFF → ordinary chat, not RAG.
def test_no_document_web_off_is_normal_chat(fake_llm, fake_collection):
    fake_collection.count.return_value = 0
    fake_llm["stream_tokens"] = ["Hello ", "there."]
    events = _parse(rag_service.stream_question("c1", "hello", [], web_search=False))
    assert "".join(e["content"] for e in events if e["type"] == "token") == "Hello there."
    stream_calls = [c for c in fake_llm["calls"] if c.get("stream")]
    assert stream_calls
    assert "Document Context" not in stream_calls[-1]["messages"][0]["content"]


# E. Document A selected → document B is never retrieved.
def test_selected_document_filter_is_not_widened(fake_collection):
    """A filter miss stays a miss: no second, unfiltered query."""
    fake_collection.count.return_value = 4
    fake_collection.query.return_value = {"documents": [[]], "metadatas": [[]], "embeddings": [[]]}

    context, sources = rag_service._retrieve_relevant(fake_collection, "anything", ["a.pdf"])

    assert context == ""
    assert sources == []
    assert fake_collection.query.call_count == 1
    assert fake_collection.query.call_args_list[0].kwargs["where"] is not None


def test_query_collection_does_not_retry_unfiltered_on_error(fake_collection):
    fake_collection.query.side_effect = RuntimeError("chroma is unhappy")
    results = rag_service._query_collection(fake_collection, [0.1, 0.2, 0.3], 4, {"filename": {"$in": ["a.pdf"]}})
    assert results["documents"] == [[]]
    assert fake_collection.query.call_count == 1


# F. A new question is answered fresh, never replayed from history.
def test_new_question_is_not_served_from_history(fake_llm, fake_collection):
    _irrelevant_chunks(fake_collection)
    history = [
        {"role": "user", "content": "who is cm of tamil nadu"},
        {"role": "assistant", "content": "M.K. Stalin is the current chief minister."},
    ]
    events = _parse(
        rag_service.stream_question(
            "c1", "who is cm of kerala", history, web_search=False, active_docs=["Sunil_Gen_AI.pdf"]
        )
    )
    tokens = [e["content"] for e in events if e["type"] == "token"]
    assert tokens == [rag_service.DOC_NOT_FOUND_MESSAGE]
    # A prior assistant answer in history must not leak through as the reply.
    assert "Stalin" not in "".join(tokens)


# Web access off must never reach the search provider.
def test_web_search_off_never_calls_the_search_provider(fake_llm, fake_collection, monkeypatch):
    fake_collection.count.return_value = 0
    calls = []
    monkeypatch.setattr(rag_service, "is_search_available", lambda: True)
    monkeypatch.setattr(rag_service, "run_web_search", lambda q: calls.append(q) or "results")
    list(rag_service.stream_question("c1", "what is the bitcoin price today", [], web_search=False))
    assert calls == []


# The prompt itself must forbid pretrained-knowledge answers.
def test_rag_prompt_forbids_general_knowledge():
    assert "ONLY the supplied document context" in rag_service.SYSTEM_RAG
    assert rag_service.DOC_NOT_FOUND_MESSAGE in rag_service.SYSTEM_RAG
    assert "using your own knowledge" not in rag_service.SYSTEM_RAG


# ── Conversational exception inside document mode ────────────────────────────
# Small talk stays conversational; anything that asks for a fact does not.

@pytest.mark.parametrize(
    "message",
    [
        "hi", "hello", "Hey!", "hiya",
        "thanks", "Thanks!", "thank you", "thank you very much", "thx",
        "ok", "ok cool", "got it", "noted",
        "good morning", "good night",
        "bye", "see you later",
        "vanakkam", "nandri da",
    ],
)
def test_is_conversational_accepts_small_talk(message):
    assert rag_service._is_conversational(message) is True


@pytest.mark.parametrize(
    "message",
    [
        # The whole point: a factual question must never take the bypass.
        "who is CM of Tamil Nadu?",
        "what is Python?",
        "what is the capital of France",
        # Greeting glued to a real question must not sneak through.
        "hi, who is the CM?",
        "thanks, now what is python",
        "good morning what is the revenue",
        # Document questions stay in document mode.
        "what does the document say",
        "summarize this",
        "python",
        "",
    ],
)
def test_is_conversational_rejects_questions(message):
    assert rag_service._is_conversational(message) is False


def test_greeting_in_document_mode_gets_a_normal_reply(fake_llm, fake_collection):
    """A document is open, web is off — "hi" must not be refused."""
    _irrelevant_chunks(fake_collection)
    fake_llm["stream_tokens"] = ["Hello! ", "How can I help?"]
    events = _parse(
        rag_service.stream_question(
            "c1", "hi", [], web_search=False, active_docs=["Sunil_Gen_AI.pdf"]
        )
    )
    reply = "".join(e["content"] for e in events if e["type"] == "token")
    assert reply == "Hello! How can I help?"
    assert rag_service.DOC_NOT_FOUND_MESSAGE not in reply
    # Answered as ordinary chat, so no document context was pasted in.
    stream_calls = [c for c in fake_llm["calls"] if c.get("stream")]
    assert "Document Context" not in stream_calls[-1]["messages"][0]["content"]


def test_thanks_in_document_mode_gets_a_normal_reply(fake_llm, fake_collection):
    _irrelevant_chunks(fake_collection)
    fake_llm["stream_tokens"] = ["You're ", "welcome!"]
    events = _parse(
        rag_service.stream_question(
            "c1", "thanks", [], web_search=False, active_docs=["Sunil_Gen_AI.pdf"]
        )
    )
    reply = "".join(e["content"] for e in events if e["type"] == "token")
    assert reply == "You're welcome!"
    assert rag_service.DOC_NOT_FOUND_MESSAGE not in reply


def test_conversational_bypass_does_not_leak_factual_questions(fake_llm, fake_collection):
    """The exception must be narrow: a fact question still gets refused, and
    the model is never asked."""
    _irrelevant_chunks(fake_collection)
    for question in ("who is CM of Tamil Nadu?", "what is Python?", "hi, who is the CM?"):
        fake_llm["calls"].clear()
        events = _parse(
            rag_service.stream_question(
                "c1", question, [], web_search=False, active_docs=["Sunil_Gen_AI.pdf"]
            )
        )
        tokens = [e["content"] for e in events if e["type"] == "token"]
        assert tokens == [rag_service.DOC_NOT_FOUND_MESSAGE], question
        assert not [c for c in fake_llm["calls"] if c.get("stream")], question


# ── Threshold calibration anchors ────────────────────────────────────────────
# Real similarity scores measured against jina-embeddings-v3 and a real PDF.
# These pin the decision so a future threshold change has to confront the data.

def test_threshold_keeps_the_measured_relevant_range():
    """The weakest genuinely-relevant query measured 0.213; it must survive."""
    assert rag_service._RAG_MIN_SIMILARITY < 0.213


def test_threshold_rejects_the_reported_bug_case():
    """"who is cm of tamil nadu" scored 0.085 against the test document."""
    assert rag_service._RAG_MIN_SIMILARITY > 0.085


def test_threshold_rejects_the_measured_unrelated_bulk():
    """12 of 13 unrelated queries scored <= 0.145."""
    assert rag_service._RAG_MIN_SIMILARITY > 0.145


# ── Web off with no grounded source ──────────────────────────────────────────
# Reported from production: web search off, no document selected,
# "who is cm of tamil nadu" answered "M.K. Stalin is the Chief Minister…"
# straight from pretrained knowledge.

def test_web_off_no_document_refuses_the_reported_case(fake_llm, fake_collection):
    """THE regression test for the reported bug, verbatim."""
    fake_collection.count.return_value = 0
    events = _parse(
        rag_service.stream_question("c1", "who is cm of tamil nadu", [], web_search=False)
    )
    tokens = [e["content"] for e in events if e["type"] == "token"]
    assert tokens == [rag_service.NO_GROUNDED_SOURCE_MESSAGE]
    assert events[-1]["type"] == "done"
    # The model must not be consulted at all.
    assert not [c for c in fake_llm["calls"] if c.get("stream")]
    assert "Stalin" not in "".join(tokens)


@pytest.mark.parametrize(
    "question",
    [
        "who is cm of tamil nadu",
        "what is the bitcoin price today",
        "latest news in India",
        "who won the last IPL",
        "what is the weather today",
        "who is the current president",
    ],
)
def test_web_off_no_document_refuses_time_sensitive(question, fake_llm, fake_collection):
    fake_collection.count.return_value = 0
    events = _parse(rag_service.stream_question("c1", question, [], web_search=False))
    tokens = [e["content"] for e in events if e["type"] == "token"]
    assert tokens == [rag_service.NO_GROUNDED_SOURCE_MESSAGE], question
    assert not [c for c in fake_llm["calls"] if c.get("stream")], question


@pytest.mark.parametrize(
    "question",
    [
        # Not time-sensitive: pretrained knowledge here is neither stale nor
        # misleading, and refusing it would gut the assistant with web off.
        "what is Python",
        "explain recursion",
        "write a poem about rain",
        "what is the capital of France",
        "summarize our chat",
        "fix this code",
        # Small talk must never be caught by the guard.
        "hi", "hello", "thanks", "good morning", "bye",
    ],
)
def test_web_off_no_document_still_answers_non_time_sensitive(question, fake_llm, fake_collection):
    fake_collection.count.return_value = 0
    fake_llm["stream_tokens"] = ["Sure", "."]
    events = _parse(rag_service.stream_question("c1", question, [], web_search=False))
    reply = "".join(e["content"] for e in events if e["type"] == "token")
    assert reply == "Sure.", question
    assert rag_service.NO_GROUNDED_SOURCE_MESSAGE not in reply, question


def test_web_on_no_document_still_uses_web_search(monkeypatch, fake_llm, fake_collection):
    """The guard must not fire when web access is available."""
    fake_collection.count.return_value = 0
    calls = []
    monkeypatch.setattr(rag_service, "is_search_available", lambda: True)
    monkeypatch.setattr(rag_service, "run_web_search", lambda q: calls.append(q) or "M.K. Stalin.")
    fake_llm["router"] = "current chief minister of Tamil Nadu"
    fake_llm["stream_tokens"] = ["M.K. ", "Stalin."]
    events = _parse(
        rag_service.stream_question("c1", "who is cm of tamil nadu", [], web_search=True)
    )
    reply = "".join(e["content"] for e in events if e["type"] == "token")
    assert reply == "M.K. Stalin."
    assert calls, "web search should have been called"


def test_document_mode_is_unaffected_by_the_web_off_guard(fake_llm, fake_collection):
    """A time-sensitive question with a document selected still goes through
    document grounding and refuses with the DOCUMENT message, not this one."""
    _irrelevant_chunks(fake_collection)
    events = _parse(
        rag_service.stream_question(
            "c1", "who is cm of tamil nadu", [], web_search=False, active_docs=["a.pdf"]
        )
    )
    tokens = [e["content"] for e in events if e["type"] == "token"]
    assert tokens == [rag_service.DOC_NOT_FOUND_MESSAGE]
    assert not [c for c in fake_llm["calls"] if c.get("stream")]


# ── Per-question system prompt assembly ──────────────────────────────────────
# The full ruleset is ~3,200 tokens against a free-tier budget of 8,000 tokens
# per minute, so sending all of it on every message was most of the reason the
# app felt slow. These pin which blocks are situational and which are not.

def _tok(s):
    return len(s) // 4


def test_situational_blocks_are_absent_from_a_plain_question():
    p = rag_service._system_prompt_for("What is RAG?")
    assert rag_service.DIAGRAM_RULE not in p
    assert rag_service.CODE_RULE not in p
    assert rag_service.MATH_RULE not in p
    assert rag_service.TEMPORAL_RULE not in p
    assert rag_service.REGIONAL_GLOSSARY not in p


def test_core_blocks_are_always_present():
    """Identity, language mirroring, accuracy, instruction following and answer
    depth apply to every reply — trimming must never reach these."""
    for q in ["hi", "What is RAG?", "draw a flowchart", "RAG na enna da"]:
        p = rag_service._system_prompt_for(q)
        assert rag_service.SYSTEM_BASE in p, q
        assert rag_service.LANGUAGE_CORE in p, q
        assert rag_service.ACCURACY_RULE in p, q
        assert rag_service.INSTRUCTION_FOLLOWING_RULE in p, q
        assert rag_service.FORMAT_DEPTH_RULE in p, q


@pytest.mark.parametrize(
    "question,block",
    [
        ("draw a flowchart of the login process", "DIAGRAM_RULE"),
        ("explain closures in javascript", "CODE_RULE"),
        ("what is 2+2", "MATH_RULE"),
        ("who is the current CSK captain", "TEMPORAL_RULE"),
    ],
)
def test_situational_block_is_attached_when_the_question_calls_for_it(question, block):
    assert getattr(rag_service, block) in rag_service._system_prompt_for(question)


@pytest.mark.parametrize(
    "question",
    ["RAG na enna da", "epdi pannradhu", "code venum da", "mujhe kya karna chahiye",
     "enti idi", "Vijay image kaatu"],
)
def test_glossary_is_attached_for_romanised_indic(question):
    assert rag_service.REGIONAL_GLOSSARY in rag_service._system_prompt_for(question)


def test_glossary_follows_the_conversation_not_just_the_last_message():
    """Opening in Tanglish then replying "yes" is still a Tanglish conversation."""
    history = [{"role": "user", "content": "epdi oru API build pannradhu"}]
    p = rag_service._system_prompt_for("yes", history)
    assert rag_service.REGIONAL_GLOSSARY in p


def test_trimming_actually_saves_a_meaningful_share_of_the_budget():
    plain = rag_service._system_prompt_for("What is RAG?")
    assert _tok(plain) < _tok(rag_service.SYSTEM_NORMAL) * 0.6


def test_full_ruleset_still_contains_every_block():
    """SYSTEM_NORMAL remains the complete set for callers with no question."""
    for block in (
        rag_service.SYSTEM_BASE, rag_service.LANGUAGE_CORE, rag_service.REGIONAL_GLOSSARY,
        rag_service.INSTRUCTION_FOLLOWING_RULE, rag_service.ACCURACY_RULE,
        rag_service.TEMPORAL_RULE, rag_service.CODE_RULE, rag_service.MATH_RULE,
        rag_service.FORMAT_DEPTH_RULE, rag_service.DIAGRAM_RULE,
    ):
        assert block in rag_service.SYSTEM_NORMAL
