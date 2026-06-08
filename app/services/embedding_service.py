"""Text chunking + embeddings.

Embeddings are computed via NVIDIA's cloud embedding API (NOT a local model),
so the backend stays lightweight — critical on low-RAM machines (no PyTorch /
sentence-transformers load). Model: nvidia/nv-embedqa-e5-v5 (1024-dim), using
input_type 'passage' for documents and 'query' for the user's question.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
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


def chunk_text(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
    )
    return splitter.split_text(text)


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
