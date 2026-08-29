"""Image provider selection and failure handling.

Production symptom that motivated these: /generate/image took 103.8s and then
returned HTTP 502 "Image generation failed." — no image, no clue why, and a
minute and a half of the user's time spent on it.
"""

import base64

import pytest

from app.services import generate_service as G

PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


class FakeResponse:
    def __init__(self, status_code=200, content=PNG_BYTES, content_type="image/png", text=""):
        self.status_code = status_code
        self.content = content
        self.headers = {"content-type": content_type}
        self.text = text


@pytest.fixture
def providers(monkeypatch):
    """Both keys present, and every real network call replaced."""
    monkeypatch.setattr(G, "POLLINATIONS_API_KEY", "poll-key")
    monkeypatch.setattr(G, "NVIDIA_API_KEY", "nv-key")
    calls = []
    monkeypatch.setattr(G, "_image_via_pollinations", lambda p, w, h: calls.append("pollinations") or "data:image/jpeg;base64,AAAA")
    monkeypatch.setattr(G, "_image_via_nvidia", lambda p, w, h: calls.append("nvidia") or "data:image/png;base64,BBBB")
    return calls


def test_pollinations_is_tried_first(providers):
    """NVIDIA must not be reached while Pollinations works — the whole point is
    to stop paying its timeout on every single image."""
    uri = G.generate_image_b64("a cat")
    assert uri.startswith("data:image/jpeg")
    assert providers == ["pollinations"]


def test_falls_back_to_nvidia_when_pollinations_fails(monkeypatch, providers):
    monkeypatch.setattr(
        G, "_image_via_pollinations",
        lambda p, w, h: (_ for _ in ()).throw(RuntimeError("pollinations down")),
    )
    uri = G.generate_image_b64("a cat")
    assert uri.startswith("data:image/png")
    assert providers == ["nvidia"]


def test_error_names_every_provider_that_failed(monkeypatch, providers):
    """The old route collapsed all of this into one opaque sentence."""
    monkeypatch.setattr(G, "_image_via_pollinations", lambda p, w, h: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(G, "_image_via_nvidia", lambda p, w, h: (_ for _ in ()).throw(RuntimeError("403")))
    with pytest.raises(RuntimeError) as err:
        G.generate_image_b64("a cat")
    assert "pollinations" in str(err.value)
    assert "nvidia" in str(err.value)


def test_missing_every_key_says_what_to_configure(monkeypatch):
    monkeypatch.setattr(G, "POLLINATIONS_API_KEY", None)
    monkeypatch.setattr(G, "NVIDIA_API_KEY", None)
    with pytest.raises(RuntimeError) as err:
        G.generate_image_b64("a cat")
    assert "POLLINATIONS_API_KEY" in str(err.value)


def test_only_configured_providers_are_attempted(monkeypatch, providers):
    monkeypatch.setattr(G, "NVIDIA_API_KEY", None)
    monkeypatch.setattr(G, "_image_via_pollinations", lambda p, w, h: (_ for _ in ()).throw(RuntimeError("down")))
    with pytest.raises(RuntimeError) as err:
        G.generate_image_b64("a cat")
    assert "nvidia" not in str(err.value)
    assert providers == []


# ── The Pollinations call itself ─────────────────────────────────────────────

def test_pollinations_returns_a_data_uri(monkeypatch):
    monkeypatch.setattr(G, "POLLINATIONS_API_KEY", "poll-key")
    monkeypatch.setattr(G.requests, "get", lambda *a, **k: FakeResponse())
    uri = G._image_via_pollinations("a cat", 1024, 1024)
    assert uri.startswith("data:image/png;base64,")
    assert base64.b64decode(uri.split(",", 1)[1]) == PNG_BYTES


def test_pollinations_bounds_its_own_wait(monkeypatch):
    """A provider is never allowed to hang past the per-attempt budget — that
    is what turned a failure into a 104-second failure."""
    seen = {}
    monkeypatch.setattr(G, "POLLINATIONS_API_KEY", "poll-key")
    monkeypatch.setattr(G.requests, "get", lambda *a, **k: seen.update(k) or FakeResponse())
    G._image_via_pollinations("a cat", 1024, 1024)
    assert seen["timeout"] == G._IMAGE_TIMEOUT
    assert G._IMAGE_TIMEOUT <= 60


def test_pollinations_rejects_a_non_image_response(monkeypatch):
    monkeypatch.setattr(G, "POLLINATIONS_API_KEY", "poll-key")
    monkeypatch.setattr(
        G.requests, "get",
        lambda *a, **k: FakeResponse(content=b"<html>nope</html>", content_type="text/html"),
    )
    with pytest.raises(RuntimeError, match="no image data"):
        G._image_via_pollinations("a cat", 1024, 1024)


def test_pollinations_surfaces_an_http_error(monkeypatch):
    monkeypatch.setattr(G, "POLLINATIONS_API_KEY", "poll-key")
    monkeypatch.setattr(G.requests, "get", lambda *a, **k: FakeResponse(status_code=429, text="rate limited"))
    with pytest.raises(RuntimeError, match="429"):
        G._image_via_pollinations("a cat", 1024, 1024)


def test_nvidia_still_bounds_its_wait(monkeypatch):
    seen = {}
    monkeypatch.setattr(G, "NVIDIA_API_KEY", "nv-key")

    class JsonResponse(FakeResponse):
        def json(self):
            return {"artifacts": [{"base64": "iVBOR" + "A" * 200}]}

    monkeypatch.setattr(G.requests, "post", lambda *a, **k: seen.update(k) or JsonResponse())
    G._image_via_nvidia("a cat", 1024, 1024)
    assert seen["timeout"] == G._IMAGE_TIMEOUT
