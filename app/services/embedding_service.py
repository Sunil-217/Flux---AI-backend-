"""Text chunking + embeddings.

Embeddings are computed via NVIDIA's cloud embedding API (NOT a local model),
so the backend stays lightweight — critical on low-RAM machines (no PyTorch /
sentence-transformers load). Model: nvidia/nv-embedqa-e5-v5 (1024-dim), using
input_type 'passage' for documents and 'query' for the user's question.

Chunking is a small dependency-free splitter (NO langchain_text_splitters, which
transitively imports transformers → torch and was OOM-ing the backend on a
3.75 GB machine). Everything here is API-based; nothing heavy loads at import.
"""

from openai import OpenAI

from app.core.config import NVIDIA_API_KEY

_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY,
    timeout=60,
)

EMBED_MODEL = "nvidia/nv-embedqa-e5-v5"
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


def _embed(inputs, input_type: str):
    """Embed a list of strings in batches. input_type is 'passage' or 'query'."""
    vectors = []
    for i in range(0, len(inputs), _BATCH):
        batch = inputs[i:i + _BATCH]
        resp = _client.embeddings.create(
            model=EMBED_MODEL,
            input=batch,
            extra_body={"input_type": input_type, "truncate": "END"},
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
