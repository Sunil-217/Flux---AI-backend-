"""
Shared test fixtures.

The heavy / external pieces are faked so the suite runs fast and fully offline:
  - embedding_service  → no torch / model download
  - pdf_service        → no PyMuPDF
  - chromadb           → no real on-disk vector DB
  - OpenAI client      → no network (configurable fake responses)
"""

import os
import sys
import types
from unittest.mock import MagicMock

import pytest

# ── Dummy key so OpenAI client construction never fails ──
os.environ.setdefault("NVIDIA_API_KEY", "test-nvidia-key")
os.environ.setdefault("GROQ_API_KEY", "")

# ── Use an in-memory DB during tests (never touch the real flux_ai.db file) ──
os.environ.setdefault("DATABASE_URL", "sqlite://")

# ── Fake the embedding service (avoid loading torch / sentence-transformers) ──
_embed = types.ModuleType("app.services.embedding_service")


class _FakeVec(list):
    def tolist(self):
        return list(self)


class _FakeEmbeddingModel:
    def encode(self, data):
        if isinstance(data, (list, tuple)):
            return _FakeVec([[0.1, 0.2, 0.3] for _ in data])
        return _FakeVec([0.1, 0.2, 0.3])


_embed.embedding_model = _FakeEmbeddingModel()
_embed.chunk_text = lambda text: ([text] if text else [])
_embed.create_embeddings = lambda chunks: [[0.1, 0.2, 0.3] for _ in chunks]
# Matches the fake collection's stored embeddings → cosine similarity 1.0 (relevant).
_embed.embed_query = lambda text: [0.1, 0.2, 0.3]
sys.modules["app.services.embedding_service"] = _embed

# ── Fake the PDF service (avoid PyMuPDF) ──
_pdf = types.ModuleType("app.services.pdf_service")
_pdf.extract_text_from_pdf = lambda path: "Sample extracted PDF text for testing."
_pdf.extract_text_from_docx = lambda path: "Sample extracted DOCX text for testing."
# upload.py imports the unified entry point; accept (file_path, ext).
_pdf.extract_text_from_file = lambda path, ext=None: "Sample extracted text for testing."
sys.modules["app.services.pdf_service"] = _pdf

# ── Stop ChromaDB from opening a real on-disk database during tests ──
import chromadb  # noqa: E402

chromadb.PersistentClient = MagicMock(return_value=MagicMock())


# ── Fake OpenAI response objects ───────────────────────────────────────────
class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content, streaming):
        if streaming:
            self.delta = _Msg(content)
        else:
            self.message = _Msg(content)


class _Resp:
    def __init__(self, content, streaming):
        self.choices = [_Choice(content, streaming)]


def make_completion(text):
    return _Resp(text, streaming=False)


def make_stream(tokens):
    return [_Resp(t, streaming=True) for t in tokens]


@pytest.fixture
def fake_llm(monkeypatch):
    """Patch rag_service.client.chat.completions.create with a configurable fake."""
    from app.services import rag_service

    state = {
        "completion_text": "Hello from the model.",
        "stream_tokens": ["Hel", "lo", "."],
        "router": "NO",
        "fail_stream_once": False,
        "calls": [],
    }

    def fake_create(*args, **kwargs):
        state["calls"].append(kwargs)
        # Any streaming call is answer generation.
        if kwargs.get("stream"):
            if state["fail_stream_once"]:
                state["fail_stream_once"] = False
                raise RuntimeError("temporary stream start failure")
            return iter(make_stream(state["stream_tokens"]))
        # The router is identified by its system prompt (ROUTER_SYSTEM), NOT by
        # model name — MODEL and ROUTER_MODEL are the same string, so a
        # model-name check would misclassify normal answer calls as router calls.
        msgs = kwargs.get("messages") or []
        is_router = bool(msgs) and msgs[0].get("content") == rag_service.ROUTER_SYSTEM
        if is_router:
            return make_completion(state["router"])
        return make_completion(state["completion_text"])

    monkeypatch.setattr(rag_service.client.chat.completions, "create", fake_create)
    if rag_service.groq_client is not None:
        monkeypatch.setattr(rag_service.groq_client.chat.completions, "create", fake_create)
    return state


@pytest.fixture
def fake_collection(monkeypatch):
    """Replace get_or_create_collection with a controllable fake collection."""
    from app.services import rag_service

    col = MagicMock()
    col.count.return_value = 0
    # Embeddings identical to the fake query vector → cosine similarity 1.0 → relevant.
    col.query.return_value = {
        "documents": [["First chunk.", "Second chunk."]],
        "metadatas": [[{"filename": "a.pdf"}, {"filename": "a.pdf"}]],
        "embeddings": [[[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]],
    }
    monkeypatch.setattr(rag_service, "get_or_create_collection", lambda chat_id: col)
    return col
