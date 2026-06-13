"""Canonical platform feature flags.

Each key maps to a user-facing capability that a platform admin can switch off
for everyone from Admin → Features. The frontend reads the effective map from
GET /features and hides any disabled feature's UI entirely.

This dict is the single source of truth for which flag keys are valid — the
frontend registry (lib/features.ts) must use the SAME keys for its labels.
Defaults are all-on, so a fresh install behaves exactly as before.
"""

DEFAULT_FEATURES: dict[str, bool] = {
    # ── Generation ──
    "image_gen": True,      # /image — image generation
    "pdf_gen": True,        # /pdf — PDF document generation
    "office_gen": True,     # /excel /word /ppt — Office document generation
    # ── Knowledge / RAG ──
    "file_upload": True,    # document & folder upload
    "media_upload": True,   # audio / video upload + transcription
    "url_ingest": True,     # web page / YouTube / GitHub ingestion
    "web_search": True,     # live web grounding
    "research": True,       # /research — deep research
    "quiz": True,           # /quiz — quiz generation
    # ── Voice ──
    "voice_input": True,    # microphone speech-to-text
    "read_aloud": True,     # neural text-to-speech on replies
    # ── Assist ──
    "translation": True,    # translate replies
    "personas": True,       # custom personas
    "memory": True,         # cross-chat memory
    "insights": True,       # usage insights
    "api_keys": True,       # developer API keys
    "code_mode": True,      # Code mode (desktop)
    # ── Chat & data preferences (Settings) ──
    "response_style": True,       # Settings → Chat → response style
    "custom_instructions": True,  # Settings → Chat → custom instructions
    "notifications": True,        # Settings → Chat → notify when a reply finishes
    "data_export": True,          # Settings → Data → export conversations
}
