from app.services.chroma_service import sanitize_chat_id


def test_sanitize_keeps_safe_characters():
    assert sanitize_chat_id("abc-123_XYZ") == "abc-123_XYZ"


def test_sanitize_replaces_special_characters():
    assert sanitize_chat_id("user@id 001!") == "user_id_001_"


def test_sanitize_preserves_uuid():
    uid = "550e8400-e29b-41d4-a716-446655440000"
    assert sanitize_chat_id(uid) == uid


def test_sanitize_handles_slashes_and_dots():
    assert sanitize_chat_id("a/b.c") == "a_b_c"
