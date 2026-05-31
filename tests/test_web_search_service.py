from unittest.mock import MagicMock

from app.services import web_search_service as wss


def test_search_unavailable_without_client(monkeypatch):
    monkeypatch.setattr(wss, "_tavily", None)
    assert wss.is_search_available() is False
    assert wss.web_search("anything") == ""


def test_search_available_with_client(monkeypatch):
    monkeypatch.setattr(wss, "_tavily", MagicMock())
    assert wss.is_search_available() is True


def test_web_search_formats_results(monkeypatch):
    fake = MagicMock()
    fake.search.return_value = {
        "answer": "Quick summary.",
        "results": [
            {"title": "Title A", "content": "Body A", "url": "https://a.com"},
            {"title": "Title B", "content": "Body B", "url": "https://b.com"},
        ],
    }
    monkeypatch.setattr(wss, "_tavily", fake)

    out = wss.web_search("some query")
    assert "Quick summary." in out
    assert "Title A" in out and "https://a.com" in out
    assert "Title B" in out


def test_web_search_empty_results(monkeypatch):
    fake = MagicMock()
    fake.search.return_value = {"results": []}
    monkeypatch.setattr(wss, "_tavily", fake)
    assert wss.web_search("q") == ""


def test_web_search_swallows_errors(monkeypatch):
    fake = MagicMock()
    fake.search.side_effect = RuntimeError("network down")
    monkeypatch.setattr(wss, "_tavily", fake)
    assert wss.web_search("q") == ""
