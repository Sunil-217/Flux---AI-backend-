"""Neural text-to-speech via Microsoft Edge TTS (free, no API key).

edge-tts is natively async, so the route is plain `async def` — no threadpool
needed (unlike the blocking LLM routes in assist.py).
"""

import re

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from app.core.security import get_current_user
from app.models import User

router = APIRouter()

_MAX_TTS_CHARS = 2000
_DEFAULT_VOICE = "en-US-AriaNeural"
_TAMIL_VOICE = "ta-IN-PallaviNeural"

# Tamil Unicode block — used to auto-pick the Tamil voice for Tamil-script text.
_TAMIL_SCRIPT = re.compile(r"[஀-௿]")


class TtsRequest(BaseModel):
    text: str
    voice: str = _DEFAULT_VOICE


@router.post("/tts")
async def tts(req: TtsRequest, user: User = Depends(get_current_user)):
    text = (req.text or "").strip()[:_MAX_TTS_CHARS]
    if not text:
        raise HTTPException(status_code=400, detail="No text to speak.")

    voice = (req.voice or "").strip() or _DEFAULT_VOICE
    # If the caller left the default voice but the text is in Tamil script,
    # fall back to the Tamil neural voice so it actually sounds right.
    if voice == _DEFAULT_VOICE and _TAMIL_SCRIPT.search(text):
        voice = _TAMIL_VOICE

    try:
        import edge_tts
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Text-to-speech isn't installed on the server (pip install edge-tts).",
        )

    audio = bytearray()
    try:
        communicate = edge_tts.Communicate(text, voice)
        async for chunk in communicate.stream():
            if chunk.get("type") == "audio":
                audio.extend(chunk.get("data") or b"")
    except Exception:
        raise HTTPException(status_code=502, detail="Speech synthesis failed. Please try again.")

    if not audio:
        raise HTTPException(status_code=502, detail="Speech synthesis returned no audio.")

    return Response(content=bytes(audio), media_type="audio/mpeg")
