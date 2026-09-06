"""Document indexing must not spend embedding quota it does not need to.

Embeddings are the one metered resource in this app that a user can burn by
accident: re-picking the same file re-embedded every chunk of it. That cost a
full round of Jina quota AND left two near-identical copies in the collection,
so retrieval returned the same passage twice and crowded out the rest of the
document.
"""

from unittest.mock import MagicMock

import pytest

from app.api.routes import upload as upload_route

CHUNKS = ["The Pro plan costs 799 rupees.", "The Max plan costs 1499 rupees."]
OTHER = ["Completely different content about the cafeteria."]


class _FakeCollection:
    """Enough of a Chroma collection to exercise the dedup decision."""

    def __init__(self):
        self.rows = {}          # id -> (document, metadata)
        self.add_calls = 0
        self.deleted = []

    def get(self, where=None, include=None):
        want = (where or {}).get("filename")
        ids, metas, docs = [], [], []
        for _id, (doc, meta) in self.rows.items():
            if want is None or meta.get("filename") == want:
                ids.append(_id)
                metas.append(meta)
                docs.append(doc)
        return {"ids": ids, "metadatas": metas, "documents": docs}

    def add(self, documents, embeddings, ids, metadatas):
        self.add_calls += 1
        for _id, doc, meta in zip(ids, documents, metadatas):
            self.rows[_id] = (doc, meta)

    def delete(self, ids):
        self.deleted.extend(ids)
        for _id in ids:
            self.rows.pop(_id, None)


@pytest.fixture
def indexing(monkeypatch):
    col = _FakeCollection()
    embed_calls = []

    def fake_embed(chunks):
        embed_calls.append(list(chunks))
        return [[0.1, 0.2, 0.3] for _ in chunks]

    monkeypatch.setattr(upload_route, "get_or_create_collection", lambda cid: col)
    monkeypatch.setattr(upload_route, "create_embeddings", fake_embed)
    return {"collection": col, "embed_calls": embed_calls}


def store(chunks, filename="pricing.pdf", chat="c1"):
    return upload_route._embed_and_store(chat, chat, filename, chunks)


# ── The three outcomes ───────────────────────────────────────────────────────

def test_a_new_document_is_indexed(indexing):
    assert store(CHUNKS) == "indexed"
    assert len(indexing["embed_calls"]) == 1
    assert len(indexing["collection"].rows) == 2


def test_re_uploading_the_identical_file_embeds_nothing(indexing):
    """The saving that matters: no embedding quota spent at all."""
    store(CHUNKS)
    assert store(CHUNKS) == "reused"
    assert len(indexing["embed_calls"]) == 1          # not 2
    assert indexing["collection"].add_calls == 1


def test_re_uploading_the_identical_file_does_not_duplicate_chunks(indexing):
    """Duplicates are not merely wasteful — near-identical chunks fill the
    retrieval window and push the rest of the document out of the answer."""
    store(CHUNKS)
    store(CHUNKS)
    docs = [doc for doc, _meta in indexing["collection"].rows.values()]
    assert sorted(docs) == sorted(CHUNKS)
    assert len(docs) == len(set(docs))


def test_an_edited_file_replaces_the_old_chunks(indexing):
    """Same name, new content: the old version must GO, not accumulate — an
    answer citing a superseded revision is worse than no answer."""
    store(CHUNKS)
    assert store(["The Pro plan now costs 899 rupees."]) == "replaced"

    docs = [doc for doc, _meta in indexing["collection"].rows.values()]
    assert docs == ["The Pro plan now costs 899 rupees."]
    assert indexing["collection"].deleted
    assert len(indexing["embed_calls"]) == 2


# ── Isolation is preserved ───────────────────────────────────────────────────

def test_dedup_is_scoped_to_one_filename(indexing):
    """Replacing one document must not touch another in the same chat."""
    store(CHUNKS, filename="pricing.pdf")
    store(OTHER, filename="cafeteria.pdf")
    store(["Pricing revised."], filename="pricing.pdf")

    by_file = {}
    for doc, meta in indexing["collection"].rows.values():
        by_file.setdefault(meta["filename"], []).append(doc)

    assert by_file["cafeteria.pdf"] == OTHER
    assert by_file["pricing.pdf"] == ["Pricing revised."]


def test_the_same_filename_in_a_different_chat_is_a_different_document(monkeypatch):
    """Collections are per chat. A file named pricing.pdf in one chat must never
    be treated as already-indexed because another chat has that name."""
    cols = {}
    embed_calls = []
    monkeypatch.setattr(upload_route, "get_or_create_collection",
                        lambda cid: cols.setdefault(cid, _FakeCollection()))
    monkeypatch.setattr(upload_route, "create_embeddings",
                        lambda chunks: (embed_calls.append(1), [[0.1]] * len(chunks))[1])

    assert store(CHUNKS, chat="c1") == "indexed"
    assert store(CHUNKS, chat="c2") == "indexed"
    assert len(embed_calls) == 2
    assert set(cols) == {"c1", "c2"}


def test_metadata_still_carries_filename_and_chat(indexing):
    """The retrieval filter selects on filename — dedup must not disturb it."""
    store(CHUNKS)
    for _doc, meta in indexing["collection"].rows.values():
        assert meta["filename"] == "pricing.pdf"
        assert meta["chat_id"] == "c1"
        assert meta["content_hash"]


# ── Failure modes ────────────────────────────────────────────────────────────

def test_a_broken_lookup_falls_through_to_indexing(monkeypatch, indexing):
    """A dedup lookup failing must never block an upload. Indexing twice is a
    wasted quota round; refusing the upload loses the user's document."""
    col = indexing["collection"]
    col.get = MagicMock(side_effect=RuntimeError("chroma unavailable"))

    assert store(CHUNKS) == "indexed"
    assert len(indexing["embed_calls"]) == 1


def test_a_failed_delete_does_not_abort_the_replacement(monkeypatch, indexing):
    store(CHUNKS)
    indexing["collection"].delete = MagicMock(side_effect=RuntimeError("locked"))

    assert store(["new content"]) == "replaced"
    assert len(indexing["embed_calls"]) == 2


# ── The fingerprint ──────────────────────────────────────────────────────────

def test_the_hash_tracks_content_not_order_of_reading():
    h = upload_route._content_hash
    assert h(CHUNKS) == h(list(CHUNKS))
    assert h(CHUNKS) != h(list(reversed(CHUNKS)))
    assert h(CHUNKS) != h(CHUNKS + ["extra"])
    assert h([]) == h([])


def test_chunk_boundaries_cannot_collide():
    """Concatenating differently-split text must not fingerprint the same."""
    h = upload_route._content_hash
    assert h(["ab", "c"]) != h(["a", "bc"])
