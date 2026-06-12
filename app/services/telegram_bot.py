"""Telegram bridge — chat with Close AI from Telegram.

Gated on the TELEGRAM_BOT_TOKEN env var. When set, `start_telegram_bridge()`
(called from main.py at startup) spawns a daemon thread that long-polls
Telegram's getUpdates API and answers each text message through the same
ask_question() pipeline as the web app (chat_id = "telegram_<chat>", so each
Telegram chat gets its own conversation/collection).

Everything is wrapped in try/except with a short sleep on errors so the
thread never dies and never blocks app startup. No token → one log line.
"""

import threading
import time

import requests

from app.core.config import TELEGRAM_BOT_TOKEN

_POLL_TIMEOUT_S = 50          # Telegram long-poll timeout
_HTTP_TIMEOUT_S = 60          # must exceed the long-poll timeout
_REPLY_CHUNK = 4000           # Telegram message hard limit is 4096 chars
_ERROR_SLEEP_S = 3


def _send_message(base: str, tg_chat_id, text: str) -> None:
    """Send a reply, chunked at 4000 chars (parse_mode omitted for safety)."""
    text = (text or "").strip() or "Sorry, I couldn't generate a reply."
    for i in range(0, len(text), _REPLY_CHUNK):
        try:
            requests.post(
                f"{base}/sendMessage",
                json={"chat_id": tg_chat_id, "text": text[i : i + _REPLY_CHUNK]},
                timeout=30,
            )
        except Exception:
            return  # don't loop on a dead connection


_WELCOME = (
    "👋 Hi! I'm *Close AI* — ask me anything and I'll answer right here.\n\n"
    "I can explain concepts, write code, help in Tamil/Tanglish, and more. "
    "Just send a message to get started!"
)


def _send_action(base: str, tg_chat_id, action: str = "typing") -> None:
    """Show 'typing…' in the chat while the model thinks (best-effort)."""
    try:
        requests.post(
            f"{base}/sendChatAction",
            json={"chat_id": tg_chat_id, "action": action},
            timeout=10,
        )
    except Exception:
        pass


def _handle_update(base: str, update: dict) -> None:
    """Answer one Telegram text message via the RAG chat pipeline."""
    message = update.get("message") or {}
    text = (message.get("text") or "").strip()
    tg_chat_id = (message.get("chat") or {}).get("id")
    if not text or tg_chat_id is None:
        return

    # Bot commands → friendly welcome instead of querying the model.
    if text.split()[0].lower() in ("/start", "/help"):
        _send_message(base, tg_chat_id, _WELCOME)
        return

    _send_action(base, tg_chat_id, "typing")
    try:
        # Imported lazily so a heavy import failure can't break app startup.
        from app.services.rag_service import ask_question

        result = ask_question("telegram_" + str(tg_chat_id), text, [])
        answer = (result or {}).get("answer") or ""
    except Exception:
        answer = "Sorry, something went wrong while answering. Please try again."

    _send_message(base, tg_chat_id, answer)


def _poll_loop(token: str) -> None:
    """Long-poll getUpdates forever; survive every error with a short sleep."""
    base = f"https://api.telegram.org/bot{token}"
    offset = None
    while True:
        try:
            params = {"timeout": _POLL_TIMEOUT_S}
            if offset is not None:
                params["offset"] = offset
            resp = requests.get(f"{base}/getUpdates", params=params, timeout=_HTTP_TIMEOUT_S)
            updates = (resp.json() or {}).get("result") or []
            for update in updates:
                offset = update.get("update_id", 0) + 1
                try:
                    _handle_update(base, update)
                except Exception:
                    pass  # one bad message must not kill the loop
        except Exception:
            time.sleep(_ERROR_SLEEP_S)


def start_telegram_bridge() -> None:
    """Start the polling thread if a bot token is configured (no-op otherwise)."""
    if not TELEGRAM_BOT_TOKEN:
        print("Telegram bridge: disabled (no token)", flush=True)
        return
    try:
        thread = threading.Thread(
            target=_poll_loop,
            args=(TELEGRAM_BOT_TOKEN,),
            daemon=True,
            name="telegram-bridge",
        )
        thread.start()
        print("Telegram bridge: polling started", flush=True)
    except Exception:
        # Never let the bridge block or crash app startup.
        print("Telegram bridge: failed to start (continuing without it)", flush=True)
