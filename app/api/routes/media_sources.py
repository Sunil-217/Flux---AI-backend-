"""Chat with media sources — YouTube videos (transcripts) and GitHub repos.

Both endpoints mirror the /upload-url indexing flow: fetch text →
chunk_text → create_embeddings → collection.add into the chat's Chroma
collection, so the existing RAG pipeline answers questions about them.
"""

import re
import uuid

import requests
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from app.core.security import get_current_user, user_owns_chat
from app.db import get_db
from app.models import User
from app.services.embedding_service import chunk_text, create_embeddings
from app.services.chroma_service import get_or_create_collection, sanitize_chat_id

router = APIRouter()

_UA_HEADERS = {"User-Agent": "Flux-AI-Backend"}


def _ingest(chat_id: str, safe_id: str, source: str, chunks: list, extra_meta: dict = None):
    """Chunk metadata + storage exactly like url.py's _ingest (filename = source)."""
    embeddings = create_embeddings(chunks)
    collection = get_or_create_collection(chat_id)
    uid = uuid.uuid4().hex[:8]  # unique per ingest so IDs never collide
    meta = {"filename": source, "chat_id": chat_id}
    if extra_meta:
        meta.update(extra_meta)
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"{safe_id}_{uid}_{i}" for i in range(len(chunks))],
        metadatas=[dict(meta) for _ in range(len(chunks))],
    )


# ── Feature: YouTube chat ────────────────────────────────────────────────────

_YT_ID_PATTERNS = [
    re.compile(r"youtu\.be/([A-Za-z0-9_-]{11})"),
    re.compile(r"[?&]v=([A-Za-z0-9_-]{11})"),
    re.compile(r"/shorts/([A-Za-z0-9_-]{11})"),
    re.compile(r"/embed/([A-Za-z0-9_-]{11})"),
    re.compile(r"/live/([A-Za-z0-9_-]{11})"),
]


class YoutubeRequest(BaseModel):
    url: str
    chat_id: str


def _extract_video_id(url: str) -> str:
    """Pull the 11-char video id out of any common YouTube URL shape."""
    url = (url or "").strip()
    for pattern in _YT_ID_PATTERNS:
        m = pattern.search(url)
        if m:
            return m.group(1)
    # A bare video id pasted directly also works.
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url):
        return url
    return ""


def _fetch_video_title(video_id: str) -> str:
    """Best-effort video title via YouTube's public oEmbed (no API key)."""
    try:
        resp = requests.get(
            "https://www.youtube.com/oembed",
            params={"url": f"https://www.youtube.com/watch?v={video_id}", "format": "json"},
            headers=_UA_HEADERS,
            timeout=10,
        )
        if resp.ok:
            return str(resp.json().get("title") or "").strip()
    except Exception:
        pass
    return ""


def _fetch_transcript_text(video_id: str) -> str:
    """Fetch the best available transcript: en/ta/hi first, then any
    (including auto-generated). Raises ValueError when none exist."""
    preferred = ["en", "ta", "hi"]
    try:
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
        except ImportError:
            raise HTTPException(
                503,
                "YouTube support isn't installed on the server (pip install youtube-transcript-api).",
            )
        api = YouTubeTranscriptApi()
        try:
            fetched = api.fetch(video_id, languages=preferred)
        except Exception:
            # Fall back to ANY transcript (manual or auto-generated, any language).
            transcript_list = api.list(video_id)
            transcript = next(iter(transcript_list), None)
            if transcript is None:
                raise ValueError("no transcripts")
            fetched = transcript.fetch()
        text = " ".join(
            (getattr(s, "text", None) or (s.get("text") if isinstance(s, dict) else "") or "")
            for s in fetched
        ).strip()
        text = re.sub(r"\s+", " ", text)
        if not text:
            raise ValueError("empty transcript")
        return text
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(str(exc))


@router.post("/upload-youtube")
async def upload_youtube(
    req: YoutubeRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Fetch a YouTube video's transcript, index it, and chat about the video."""
    if not user_owns_chat(db, user, req.chat_id):
        raise HTTPException(status_code=404, detail="Chat not found")

    video_id = _extract_video_id(req.url)
    if not video_id:
        raise HTTPException(
            status_code=400,
            detail="That doesn't look like a YouTube link. Paste a youtube.com or youtu.be video URL.",
        )

    try:
        text = await run_in_threadpool(_fetch_transcript_text, video_id)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=(
                "No transcript/captions are available for this video, so there's nothing "
                "to index. Try a video that has captions enabled."
            ),
        )

    title = await run_in_threadpool(_fetch_video_title, video_id)
    source = title or req.url

    chunks = chunk_text(text)
    try:
        await run_in_threadpool(
            _ingest,
            req.chat_id,
            sanitize_chat_id(req.chat_id),
            source,
            chunks,
            {"source": "youtube", "url": f"https://www.youtube.com/watch?v={video_id}"},
        )
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to index the video. Please try again.")

    return {"source": source, "chars": len(text)}


# ── Feature: GitHub repo chat ────────────────────────────────────────────────

_GITHUB_URL = re.compile(r"github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)")

_CODE_EXTENSIONS = (
    ".py", ".js", ".ts", ".tsx", ".jsx", ".md", ".json", ".yaml", ".yml",
    ".toml", ".java", ".go", ".rs", ".c", ".cpp", ".html", ".css",
)
_SKIP_PATH_PARTS = ("node_modules/", "dist/", "build/")

_MAX_REPO_FILES = 40
_MAX_FILE_BYTES = 60 * 1024     # 60KB per file
_MAX_TOTAL_CHARS = 400 * 1024   # ~400KB of concatenated content


class GithubRequest(BaseModel):
    url: str
    chat_id: str


def _parse_github_repo(url: str):
    m = _GITHUB_URL.search((url or "").strip())
    if not m:
        return None, None
    owner, repo = m.group(1), m.group(2)
    if repo.endswith(".git"):
        repo = repo[:-4]
    return owner, repo


def _github_get(url: str, params: dict = None):
    return requests.get(url, params=params, headers=_UA_HEADERS, timeout=20)


def _is_wanted_file(path: str, size) -> bool:
    lower = path.lower()
    if any(part in lower for part in _SKIP_PATH_PARTS):
        return False
    if not lower.endswith(_CODE_EXTENSIONS):
        return False
    if isinstance(size, int) and size > _MAX_FILE_BYTES:
        return False
    return True


def _fetch_repo_text(owner: str, repo: str):
    """Fetch up to 40 code/doc files from a public repo. Returns (text, n_files).
    Raises ValueError with a user-facing message on any GitHub-side problem."""
    # 1) Repo metadata → default branch.
    resp = _github_get(f"https://api.github.com/repos/{owner}/{repo}")
    if resp.status_code == 404:
        raise ValueError(f"Repository {owner}/{repo} was not found (is it private?).")
    if resp.status_code in (403, 429):
        raise ValueError("GitHub API rate limit reached. Please try again in a few minutes.")
    if not resp.ok:
        raise ValueError("Couldn't reach GitHub for that repository. Try again.")
    branch = (resp.json() or {}).get("default_branch") or "main"

    # 2) Full recursive tree.
    resp = _github_get(
        f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}",
        params={"recursive": "1"},
    )
    if resp.status_code in (403, 429):
        raise ValueError("GitHub API rate limit reached. Please try again in a few minutes.")
    if not resp.ok:
        raise ValueError("Couldn't list the repository's files. Try again.")
    tree = (resp.json() or {}).get("tree") or []

    paths = [
        entry["path"]
        for entry in tree
        if entry.get("type") == "blob"
        and entry.get("path")
        and _is_wanted_file(entry["path"], entry.get("size"))
    ][:_MAX_REPO_FILES]

    if not paths:
        raise ValueError("No readable code or documentation files were found in that repository.")

    # 3) Raw contents, concatenated with file markers (capped at ~400KB total).
    blocks = []
    total = 0
    fetched = 0
    for path in paths:
        if total >= _MAX_TOTAL_CHARS:
            break
        try:
            raw = _github_get(
                f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
            )
            if not raw.ok:
                continue
            content = raw.text
            if not content.strip():
                continue
            block = f"=== {path} ===\n{content[: _MAX_FILE_BYTES]}"
            blocks.append(block[: _MAX_TOTAL_CHARS - total])
            total += len(block)
            fetched += 1
        except Exception:
            continue

    if not blocks:
        raise ValueError("Couldn't download any files from that repository. Try again.")

    return "\n\n".join(blocks)[:_MAX_TOTAL_CHARS], fetched


@router.post("/upload-github")
async def upload_github(
    req: GithubRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Index a public GitHub repository's code + docs into the chat."""
    if not user_owns_chat(db, user, req.chat_id):
        raise HTTPException(status_code=404, detail="Chat not found")

    owner, repo = _parse_github_repo(req.url)
    if not owner or not repo:
        raise HTTPException(
            status_code=400,
            detail="That doesn't look like a GitHub repository URL (expected github.com/owner/repo).",
        )

    try:
        text, files = await run_in_threadpool(_fetch_repo_text, owner, repo)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        raise HTTPException(status_code=400, detail="Couldn't fetch that repository. Try again.")

    source = f"{owner}/{repo}"
    chunks = chunk_text(text)
    try:
        await run_in_threadpool(
            _ingest,
            req.chat_id,
            sanitize_chat_id(req.chat_id),
            source,
            chunks,
            {"source": f"github:{source}", "url": f"https://github.com/{owner}/{repo}"},
        )
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to index the repository. Please try again.")

    return {"source": source, "files": files}
