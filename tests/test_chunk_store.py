"""The durable chunk mirror and the rebuild that uses it.

Found on production: ChromaDB persists to a local directory that does not
survive a deploy on the hosting tier. Minutes after a redeploy, a question the
uploaded PDF had answered verbatim came back "I don't have the document", and
re-uploading the identical file was embedded from scratch instead of being
recognised as already indexed. These tests pin the repair: what the upload
route indexes is mirrored to the database, an empty collection is rebuilt from
that mirror with identical ids, documents, embeddings and metadata, and a
rebuilt collection satisfies the reuse check so no embedding quota is spent
twice.
"""

from unittest.mock import MagicMock

import pytest

from app.api.routes import upload as upload_route
from app.db import Base, engine
from app.services import chroma_service, chunk_store


@pytest.fixture(autouse=True, scope="module")
def _tables():
    # In production the table is created at startup by main._ensure_schema.
    # This module does not import main, so create the schema on the test
    # engine here; the in-memory SQLite connection is per-thread and outlives
    # each session, so the tables persist across the module's tests.
    Base.metadata.create_all(bind=engine)


def _empty_collection():
    col = MagicMock()
    col.count.return_value = 0
    return col


def test_hydrate_rebuilds_an_empty_collection_from_the_mirror():
    chunk_store.save_chunks(
        "c-h1", "a.pdf", "h1",
        ["c-h1_x_0", "c-h1_x_1"], ["one", "two"], [[0.1, 0.2], [0.3, 0.4]],
    )
    col = _empty_collection()
    assert chunk_store.hydrate("c-h1", col) == 2
    kwargs = col.add.call_args.kwargs
    assert kwargs["ids"] == ["c-h1_x_0", "c-h1_x_1"]
    assert kwargs["documents"] == ["one", "two"]
    assert kwargs["embeddings"] == [[0.1, 0.2], [0.3, 0.4]]
    assert kwargs["metadatas"][0] == {"filename": "a.pdf", "chat_id": "c-h1", "content_hash": "h1"}


def test_hydrate_leaves_a_live_collection_alone():
    chunk_store.save_chunks("c-h2", "a.pdf", "h", ["c-h2_0"], ["x"], [[0.5]])
    col = MagicMock()
    col.count.return_value = 3
    assert chunk_store.hydrate("c-h2", col) == 0
    col.add.assert_not_called()


def test_hydrate_does_not_requery_a_chat_it_just_found_empty(monkeypatch):
    col = _empty_collection()
    assert chunk_store.hydrate("c-h3", col) == 0

    def boom():
        raise AssertionError("database touched inside the memo window")

    monkeypatch.setattr(chunk_store, "SessionLocal", boom)
    assert chunk_store.hydrate("c-h3", col) == 0


def test_save_replaces_earlier_rows_for_the_same_file_and_clears_the_memo():
    chunk_store.save_chunks("c-h4", "a.pdf", "h1", ["c-h4_a_0"], ["old"], [[0.1]])
    live = MagicMock()
    live.count.return_value = 1
    chunk_store.hydrate("c-h4", live)  # memoised as checked
    chunk_store.save_chunks("c-h4", "a.pdf", "h2", ["c-h4_b_0"], ["new"], [[0.2]])
    col = _empty_collection()
    assert chunk_store.hydrate("c-h4", col) == 1
    assert col.add.call_args.kwargs["documents"] == ["new"]


def test_delete_collection_purges_the_mirror(monkeypatch):
    chunk_store.save_chunks("c-h5", "a.pdf", "h", ["c-h5_0"], ["x"], [[0.1]])
    monkeypatch.setattr(chroma_service, "_get_client", lambda: MagicMock())
    chroma_service.delete_collection("c-h5")
    assert chunk_store.hydrate("c-h5", _empty_collection()) == 0


def test_upload_mirrors_exactly_what_it_indexed(monkeypatch):
    monkeypatch.setattr(upload_route, "create_embeddings", lambda chunks: [[0.1, 0.2] for _ in chunks])
    col = MagicMock()
    col.get.return_value = {"ids": [], "metadatas": []}
    monkeypatch.setattr(upload_route, "get_or_create_collection", lambda cid: col)

    assert upload_route._embed_and_store("c-h6", "c-h6", "doc.pdf", ["alpha", "beta"]) == "indexed"
    added = col.add.call_args.kwargs

    fresh = _empty_collection()
    assert chunk_store.hydrate("c-h6", fresh) == 2
    rebuilt = fresh.add.call_args.kwargs
    assert rebuilt["ids"] == added["ids"]
    assert rebuilt["documents"] == added["documents"]
    assert rebuilt["metadatas"] == added["metadatas"]


def test_after_a_restart_the_same_file_is_reused_not_re_embedded(monkeypatch):
    """The production symptom, end to end: index, lose the store, upload again."""
    embed_calls = []

    def fake_embed(chunks):
        embed_calls.append(len(chunks))
        return [[0.1, 0.2] for _ in chunks]

    monkeypatch.setattr(upload_route, "create_embeddings", fake_embed)

    # Before the restart: a normal first upload into an empty collection.
    before = MagicMock()
    before.get.return_value = {"ids": [], "metadatas": []}
    monkeypatch.setattr(upload_route, "get_or_create_collection", lambda cid: before)
    assert upload_route._embed_and_store("c-h7", "c-h7", "doc.pdf", ["alpha", "beta"]) == "indexed"
    assert embed_calls == [2]

    # After the restart: a brand-new, empty collection. Model what Chroma does
    # once rows are added so the upload route's reuse lookup can see them.
    after = MagicMock()
    after.count.return_value = 0
    stored = {"ids": [], "metadatas": []}

    def add(**kwargs):
        stored["ids"].extend(kwargs["ids"])
        stored["metadatas"].extend(kwargs["metadatas"])
        after.count.return_value = len(stored["ids"])

    after.add.side_effect = add
    after.get.side_effect = lambda **kw: stored

    def get_or_create(cid):
        # What chroma_service.get_or_create_collection does on every access.
        chunk_store.hydrate(cid, after)
        return after

    monkeypatch.setattr(upload_route, "get_or_create_collection", get_or_create)
    assert upload_route._embed_and_store("c-h7", "c-h7", "doc.pdf", ["alpha", "beta"]) == "reused"
    assert embed_calls == [2], "no embedding quota may be spent on a file the mirror already holds"
