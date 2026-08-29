"""Text chunking + embeddings.

Embeddings come from a cloud API, NOT a local model, so the backend stays
lightweight — critical on a small instance. (Measured 2026-08-29: the lightest
usable local multilingual ONNX model still costs ~500 MB of RSS, which does not
fit alongside the app.) Nothing heavy loads at import.

Provider is chosen by which key is set:

  JINA_API_KEY   → jina-embeddings-v3     (1024-dim, ~89 languages, free tier)
  NVIDIA_API_KEY → nvidia/nv-embedqa-*    (1024-dim, legacy)

Both are 1024-dim, so switching between them does not change the vector width —
but the vectors are NOT interchangeable. Documents indexed with one provider
must be re-indexed before they can be searched with the other.

Chunking is a small dependency-free splitter (NO langchain_text_splitters, which
transitively imports transformers → torch and was OOM-ing the backend on a
3.75 GB machine).
"""

from openai import OpenAI

from app.core.config import JINA_API_KEY, NVIDIA_API_KEY

# Jina's embeddings endpoint is OpenAI-compatible, so the same SDK serves both.
if JINA_API_KEY:
    EMBED_PROVIDER = "jina"
    EMBED_MODEL = "jina-embeddings-v3"
    _client = OpenAI(base_url="https://api.jina.ai/v1", api_key=JINA_API_KEY, timeout=60)
else:
    EMBED_PROVIDER = "nvidia"
    EMBED_MODEL = "nvidia/nv-embedqa-mistral-7b-v2"
    _client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=NVIDIA_API_KEY,
        timeout=60,
    )

EMBED_DIM = 1024
_BATCH = 50  # keep request sizes well within the API's per-call input limit


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150):
    """Split text into overlapping chunks on natural boundaries.

    Dependency-free (no langchain / transformers / torch). Prefers to break at a
    paragraph, then line, sentence, clause, then word boundary near the target
    size, keeping `overlap` characters of context between consecutive chunks.
    """
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            # Look for a clean boundary in the back half of the window.
            window_start = start + chunk_size // 2
            for sep in ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "):
                idx = text.rfind(sep, window_start, end)
                if idx != -1:
                    end = idx + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(end - overlap, start + 1)  # keep overlap, always progress
    return chunks


# Both providers want to know whether a string is a stored document or a live
# question — asymmetric embedding measurably improves retrieval — but they spell
# it differently.
_JINA_TASK = {"passage": "retrieval.passage", "query": "retrieval.query"}


def _extra_body(input_type: str) -> dict:
    if EMBED_PROVIDER == "jina":
        return {"task": _JINA_TASK[input_type], "truncate": True}
    return {"input_type": input_type, "truncate": "END"}


def _embed(inputs, input_type: str):
    """Embed a list of strings in batches. input_type is 'passage' or 'query'."""
    vectors = []
    for i in range(0, len(inputs), _BATCH):
        batch = inputs[i:i + _BATCH]
        resp = _client.embeddings.create(
            model=EMBED_MODEL,
            input=batch,
            extra_body=_extra_body(input_type),
        )
        vectors.extend(item.embedding for item in resp.data)
    return vectors


def create_embeddings(chunks):
    """Embed document chunks (passages). Returns a list of float vectors."""
    if not chunks:
        return []
    return _embed(chunks, "passage")


def embed_query(text: str):
    """Embed a single query string. Returns one float vector."""
    return _embed([text], "query")[0]
