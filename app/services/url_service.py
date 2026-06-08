"""Fetch a web page and extract its readable text (for 'chat with a URL').

Includes a basic SSRF guard so the server can't be tricked into fetching
internal/loopback/cloud-metadata addresses.
"""

import socket
import ipaddress
from urllib.parse import urlparse
import urllib.request
from html.parser import HTMLParser

MAX_BYTES = 5_000_000      # don't download more than ~5 MB
MAX_TEXT_CHARS = 60_000    # cap extracted text so we don't embed a whole site


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self.title = ""
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript", "svg", "head"):
            self._skip += 1
        if tag == "title":
            self._in_title = True

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript", "svg", "head") and self._skip:
            self._skip -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data
            return
        if self._skip:
            return
        t = data.strip()
        if t:
            self.parts.append(t)


def _is_safe_url(url: str) -> bool:
    """Reject non-http(s) and any host resolving to a private/loopback range."""
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            return False
        for info in socket.getaddrinfo(parsed.hostname, None):
            ip = ipaddress.ip_address(info[4][0])
            if (ip.is_private or ip.is_loopback or ip.is_link_local
                    or ip.is_reserved or ip.is_multicast or ip.is_unspecified):
                return False
        return True
    except Exception:
        return False


def fetch_url_text(url: str):
    """Return (title, text) for a web page. Raises ValueError on bad/blocked URLs."""
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    if not _is_safe_url(url):
        raise ValueError("That URL isn't allowed (must be a public http/https address).")

    req = urllib.request.Request(
        url,
        headers={
            # A realistic browser UA + Accept headers so sites that gate on the
            # user-agent serve their normal (static) HTML instead of a stub.
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        raw = resp.read(MAX_BYTES)
    html = raw.decode("utf-8", errors="replace")

    parser = _TextExtractor()
    parser.feed(html)
    text = "\n".join(parser.parts)[:MAX_TEXT_CHARS]
    title = (parser.title.strip() or urlparse(url).hostname or url)[:120]
    return title, text
