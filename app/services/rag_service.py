import json
import re
from datetime import datetime

import numpy as np
from openai import OpenAI

from app.db import SessionLocal
from app.models import ChatWebContext

from app.core.config import (
    NVIDIA_API_KEY,
    GROQ_API_KEY,
)

from app.services.chroma_service import (
    get_or_create_collection
)

from app.services.embedding_service import (
    embed_query
)

from app.services.web_search_service import (
    web_search as run_web_search,
    is_search_available
)

# ── NVIDIA client — Code editing + Vision (most powerful free code specialist) ──
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY,
    timeout=60.0,
)

# ── Groq client — Chat / Router / Plan (same llama-3.3-70b, 3-5x faster inference) ──
# Groq benchmarked at ~1.9s vs NVIDIA's ~5-10s for the same model. Chat, routing,
# and JSON planning all benefit from lower latency. Falls back to NVIDIA client
# automatically if GROQ_API_KEY is absent (see _groq_or_nvidia).
groq_client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY or NVIDIA_API_KEY,  # graceful fallback if key missing
    timeout=30.0,
) if GROQ_API_KEY else None

def _groq_or_nvidia():
    """Return the Groq client when available, otherwise the NVIDIA client."""
    return groq_client if groq_client is not None else client

# ── Model routing (verified via benchmark 2026-06-08) ──
# CHAT / RAG answers   → Groq  llama-3.3-70b-versatile  (~1.9s, 3-5x faster)
# Router (web search?) → Groq  llama-3.3-70b-versatile  (~1.9s, fast decisions)
# Agent plan (JSON)    → Groq  llama-3.3-70b-versatile  (~1.9s, reliable JSON)
# Code edit / Q&A      → NVIDIA qwen3-coder-480b         (~15s, strongest free coder)
# Vision               → NVIDIA llama-3.2-11b-vision     (only option with vision)

MODEL        = "llama-3.3-70b-versatile"               # Groq — chat / RAG
ROUTER_MODEL = "llama-3.3-70b-versatile"               # Groq — web-search routing
PLAN_MODEL   = "llama-3.3-70b-versatile"               # Groq — agent JSON planning
NVIDIA_CHAT_MODEL = "meta/llama-3.3-70b-instruct"      # NVIDIA — fallback chat / RAG
CODE_MODEL   = "qwen/qwen3-coder-480b-a35b-instruct"   # NVIDIA — code edit / Q&A
VISION_MODEL = "meta/llama-3.2-11b-vision-instruct"    # NVIDIA — image / screenshot

ROUTER_SYSTEM = (
    "You are a routing classifier. You do NOT answer questions. "
    "You output ONLY one of two things: the word NO, or a short web search query.\n"
    "Output a search query ONLY when answering correctly needs current, recent, or time-sensitive "
    "information (news, prices, sports rosters/results, weather, who currently holds a role/office, "
    "events that may have changed after 2023).\n"
    "For greetings, general knowledge, definitions, coding, math, or timeless facts, output exactly: NO\n"
    "Translate the user's query into clear English when you generate a search query — even if they "
    "asked in Tanglish, Hinglish, Tamil, Hindi, etc. The English version produces better results.\n"
    "When you output a query: keep the user's EXACT topic — never change the subject. Expand common "
    "abbreviations so the search is unambiguous (cm = Chief Minister, PM = Prime Minister, CEO, GDP, etc.).\n"
    "Use the conversation so far to resolve references (like 'his', 'that team') into a standalone query.\n"
    "Never answer the question. Output NO or a query, nothing else.\n\n"
    "Q: hi\nA: NO\n"
    "Q: what is RAG in gen ai\nA: NO\n"
    "Q: write a python function to reverse a string\nA: NO\n"
    "Q: explain pannu da what is fastapi\nA: NO\n"
    "Q: what is the capital of France\nA: NO\n"
    "Q: who is the current CSK captain\nA: current Chennai Super Kings captain\n"
    "Q: current cm in tamil nadu\nA: current Chief Minister of Tamil Nadu\n"
    "Q: who is the pm now\nA: current Prime Minister of India\n"
    "Q: latest iphone price in india\nA: latest iPhone price India\n"
    "Q: who won the last IPL\nA: most recent IPL winner\n"
    "Q: ippo CSK captain yaaru\nA: current Chennai Super Kings captain\n"
    "Q: tamil nadu cm yaaru ippo\nA: current Chief Minister of Tamil Nadu\n"
    "Q: abhi india ka pm kaun hai\nA: current Prime Minister of India\n"
    "Q: bitcoin price now\nA: current Bitcoin price USD"
)

# Reply-language rule shared by both modes.
LANGUAGE_RULE = (
    "LANGUAGE — MATCH THE USER'S SCRIPT EXACTLY. THIS RULE OVERRIDES EVERYTHING ELSE:\n"
    "STEP 1 — look at the ALPHABET (script) of the user's LATEST message, before anything else:\n"
    "  • LATIN / ENGLISH alphabet (a-z), INCLUDING romanized Indian languages — Tanglish, "
    "Hinglish, Tenglish, Manglish, Kanglish: your ENTIRE reply MUST be written in the LATIN "
    "alphabet too. You are STRICTLY FORBIDDEN from outputting Tamil (தமிழ்), Devanagari (हिन्दी), "
    "Telugu, Malayalam, Kannada, Bengali, or ANY other native/Indic script. For example "
    "'tanglish la tha da', 'code venum da', 'RAG nu enna' are LATIN input → reply in ROMANIZED "
    "Tanglish using ONLY English letters (e.g. 'RAG-nu Retrieval-Augmented Generation da. Idhu "
    "oru technique...'), NEVER in தமிழ் script. A Tamil/Hindi answer written in English letters is "
    "what is required — do NOT convert it to native script.\n"
    "  • NATIVE script (the user actually typed தமிழ் / हिन्दी / العربية / 中文 / 日本語 etc.): "
    "reply in that SAME native script.\n"
    "STEP 2 — match the LANGUAGE and tone: romanized-Tamil in → romanized-Tamil out; Hinglish in → "
    "Hinglish out; English in → English out; native-script in → same native script out. Keep the "
    "user's code-switching blend (half English/half Tanglish is fine — mirror it). Do NOT add "
    "parenthetical translations.\n"
    "OVERRIDE: If the user explicitly asks for 'Tanglish' / 'Hinglish' / 'English letters' / "
    "'romanized' / 'tanglish la' — ALWAYS use the LATIN alphabet, no matter what. Only reply in "
    "pure English if the user wrote in pure English or explicitly asked for English.\n\n"
    "UNDERSTANDING CASUAL TANGLISH: Tamil filler/casual particles are informal tone, NOT something "
    "to question. Common ones: 'da'/'machan'/'machi' (casual 'bro'), 'tha'/'dhaan' (emphasis: "
    "just/itself), 'kudu'/'kodu'/'tha'/'venum'/'venaa' (give / I want / need), 'ha'/'aa' (makes it a "
    "question: 'is it?'), 'pannu' (do/make), 'sollu' (tell), 'kaattu'/'kaami' (show), "
    "'eppadi'/'epdi'/'yappdi' (how), 'enna'/'yenna' (what), 'yaaru' (who), 'yethuku'/'edhukku' (why), "
    "'irukku' (is/are there), 'irukkanum'/'irrkanum' (must be / should be), 'aaganum'/'aganum' "
    "(should happen / become), 'aagala'/'aagalla'/'aglada' (didn't happen / not done), "
    "'mudiyum'/'mudium' (can / is it possible), 'podu' (add/put/run), 'venam' (don't want), "
    "'illa' (no/not), 'romba' (very), 'konjam' (a little), 'na' (if / so / 'means'), '-la' (in/at, "
    "or a casual question marker), 'kuda'/'kooda' (also/too), 'kulla'/'ulla' (within/inside), "
    "'innum'/'innu' (more/still/even), 'maari'/'maathiri' (like / similar to), 'ellam'/'yalla' "
    "(all/everything), 'aprm'/'apparam' (then / after that), 'seri' (ok/fine), 'vai'/'vaikka' "
    "(make it / keep it). Examples: 'api code tha da' = 'give me the API code'; 'code venum' = 'I "
    "want the code'; 'epdi pannradhu' = 'how do I do it'; 'explain pannu da' = 'please explain'; "
    "'2 sec kulla aaganum' = 'it should happen within 2 seconds'; 'git la push pannu' = 'push it to "
    "git'; 'innum better ah irukkanum' = 'it should be even better'; 'no na fix pannu' = 'if not, "
    "fix it'; 'testing podu da' = 'run the test'. NEVER reply that these are 'unclear' or ask what "
    "'da'/'tha'/'ha'/'na' means — infer intent from the meaningful words and answer helpfully.\n\n"
    "WRITE TANGLISH NATURALLY: when replying in Tanglish, write like a fluent Tamil-English speaker "
    "actually texts — smooth and conversational, NOT a stiff word-by-word translation. Mirror the "
    "user's exact register and code-switch ratio (keep technical terms in English; connective and "
    "casual words in Tanglish). Stay clear and genuinely helpful, never forced or cringe.\n\n"
    "UNDERSTANDING HINGLISH: Similar romanized particles apply: 'kya' (what), 'kaise' (how), "
    "'kahaan' (where), 'matlab' (means), 'thoda' (a little), 'bahut'/'bohot' (very), 'kar' "
    "(do), 'hai' (is), 'nahi' (no), 'haan' (yes), 'yaar'/'bhai' (bro). Reply in the same "
    "Hinglish style if the user does.\n\n"
    "AMBIGUITY: If the request is genuinely unclear (which API, which language, etc.), make a "
    "sensible assumption from the conversation and give a useful answer with a short example, "
    "rather than only asking for clarification."
)

# Short, forceful recency reminder appended to the VERY END of every system
# prompt (after the long rules + web context). llama models "lose" instructions
# buried mid-prompt; putting the language mirror LAST — right before the user's
# message — makes Tanglish/Hinglish replies dramatically more reliable.
LANGUAGE_REMINDER = (
    "\n\nFINAL REMINDER (highest priority — overrides formatting/verbosity rules): "
    "Reply in the SAME language and script as the user's LATEST message. English in "
    "→ English out. If they wrote romanized Tamil / Tanglish (Tamil words in English "
    "letters — e.g. 'enna da', 'epdi pannrathu', 'code venum da', 'explain pannu', "
    "'idhu enna', 'seri va'), reply in natural, fluent romanized Tanglish using ONLY "
    "English letters — NEVER Tamil script, and NEVER switch to plain English. "
    "Hinglish in → Hinglish out. Mirror the user's casual tone and code-switching, "
    "and treat casual particles (da, machi, tha, dhaan, pannu, venum, kudu, enna, "
    "yaaru, epdi, yappdi, ha, aa, na, la, kuda, kulla, innum, maari, aaganum, "
    "mudiyum, ellam, seri, illa, romba) as normal informal speech — infer intent "
    "and answer helpfully in natural, fluent Tanglish; NEVER say they're unclear "
    "or ask what they mean."
)

# Accuracy + reasoning rule — pushes the model to think carefully on hard
# questions and be honest about uncertainty rather than guess.
ACCURACY_RULE = (
    "ACCURACY & HONESTY:\n"
    "- For factual questions, give the most accurate answer you can. If you are genuinely unsure, "
    "say so plainly ('I'm not sure, but...') and give your best estimate with a clear caveat — "
    "never invent specifics (names, numbers, dates, quotes, URLs, citations) you don't actually "
    "know.\n"
    "- THINK STEP BY STEP for any non-trivial question (math, multi-step reasoning, code, logic, "
    "comparison, planning). Work through the steps internally before you answer, then present the "
    "result clearly. Do not skip steps. For arithmetic, calculate deliberately — never guess a "
    "number you can compute exactly.\n"
    "- Read the user's question CAREFULLY — answer the question they actually asked, not a "
    "related one. If multiple sub-questions are present, address each. If they specify a format "
    "(bullets, numbered list, table, code only), match it precisely.\n"
    "- Prefer specific, concrete answers over vague generalities. If a code example or short "
    "snippet illustrates the point better than prose, include it.\n"
    "- VERIFY before finalizing: did you answer the actual question? Are your facts grounded? Is "
    "your reply in the user's language and format? If anything's off, fix it before sending.\n"
    "- Don't pad. No 'Certainly!' / 'Great question!' / 'I'd be happy to help!' openers — start "
    "with the answer."
)

# Strict instruction-following — small details matter (count, format, language,
# style). The model wins user trust by honoring exact requests, not paraphrasing.
INSTRUCTION_FOLLOWING_RULE = (
    "INSTRUCTION FOLLOWING:\n"
    "- Match the user's exact request: count (\"3 reasons\" → exactly 3), format (bullets vs prose vs "
    "table vs code-only), level (beginner vs expert), and length (short answer vs deep dive).\n"
    "- If they ask in a particular language or romanized style, REPLY in that same style — never "
    "switch back to English unprompted.\n"
    "- If they ask for code only, give code only — no explanation paragraphs. If they ask for an "
    "explanation only, no code unless it directly illustrates the point.\n"
    "- Mirror their tone (casual ↔ formal). Don't over-formalize a casual question or vice versa."
)

# Temporal grounding + anti-fabrication rule. Today's actual date is injected
# per-request (see _ground_prompt) so the model never assumes it is still at its
# training cutoff — this is what stops it from claiming a past event "hasn't
# happened yet", and from inventing winners/scores/prices it can't verify.
TEMPORAL_RULE = (
    "TIME & FACTUAL ACCURACY:\n"
    "- Today's real date is given below. Use it to reason about what has already happened — "
    "NEVER claim a past event 'has not occurred yet' just because it falls after your training "
    "cutoff.\n"
    "- For well-established or historical facts (past results and champions, general knowledge — "
    "anything up to your training cutoff), answer directly and CONFIDENTLY from your own "
    "knowledge. You do NOT need web results to confirm these. Never refuse or say 'the web "
    "results don't mention it' for a fact you actually know — just answer it.\n"
    "- Live web search results (when shown below) are there to SUPPLEMENT your knowledge for "
    "RECENT or fast-changing things (this year's results, current prices, who currently holds a "
    "role, news). Prefer them for those. If they don't cover part of a question but you know that "
    "part from training, still answer that part from your own knowledge.\n"
    "- If web results don't confirm a recent fact but you DO have older knowledge of it, give your "
    "best answer and clearly caveat that it may be out of date (e.g. 'As of my last update it was "
    "X — this may have changed since'). Prefer a caveated best-effort answer over a flat refusal. "
    "Only say you have no information when you genuinely know nothing about it. If web results show "
    "only predictions or schedules for an upcoming event, say the result isn't confirmed yet.\n"
    "- Never present a prediction, rumor, or assumption as a confirmed fact."
)

# Coding-assistant behaviour: generate, debug, and fix code well.
CODE_RULE = (
    "CODING: You are also a strong coding assistant.\n"
    "- When asked to write code, give clean, correct, runnable code in a fenced block tagged with "
    "the language (e.g. ```python). Keep prose short; put it after the code.\n"
    "- When the user PASTES code to debug or fix: first state the bug(s) clearly and briefly, then "
    "return the COMPLETE corrected code in a fenced block (not just a diff or a fragment) so they "
    "can copy-paste it directly. Mention the language/framework if it's obvious.\n"
    "- If details are missing to run it, state your assumption and still provide working code.\n"
    "- Prefer practical, idiomatic solutions; add only the comments that genuinely help."
)

# Math renders via KaTeX on the frontend, which needs $...$ / $$...$$ delimiters.
# (No literal { } here — SYSTEM_RAG runs through str.format for the context.)
MATH_RULE = (
    "MATH: Write mathematical expressions in LaTeX wrapped in dollar signs so they render — "
    "inline math as $...$ (e.g. $x^2 - 5x + 6$) and standalone equations as $$...$$ on their own "
    "line. Use real LaTeX commands (\\frac, \\sqrt, \\pm, \\times, \\cdot, ^, _, and Greek letters) "
    "instead of plain-text symbols like the square-root character."
)

# The core "answer like Claude / ChatGPT" rule — rich, well-structured, depth
# calibrated to the question. This is what makes replies feel premium.
FORMAT_DEPTH_RULE = (
    "ANSWER DEPTH & FORMATTING — respond like a top-tier assistant (Claude / ChatGPT):\n"
    "- CALIBRATE DEPTH to the question. A greeting or a quick factual lookup gets a short, direct "
    "answer (1-3 sentences). But anything conceptual, technical, or open-ended — 'explain', 'what "
    "is', 'how does', 'why', 'difference between', 'how do I', 'best way to', 'compare' — deserves "
    "a THOROUGH, genuinely helpful answer. Default to richer rather than terse for these; never "
    "give a one-line reply to a question that wants understanding.\n"
    "- STRUCTURE longer answers for skimmability: open with a 1-2 sentence direct answer or "
    "definition, then expand in clearly separated sections. Use markdown headings (## and ###), "
    "**bold** for key terms, and bulleted or numbered lists for steps, features, and options.\n"
    "- EXPLAIN, don't just state: give the reasoning and the 'why', cover trade-offs, and include "
    "at least one concrete example, analogy, or mini code snippet for any non-trivial concept. "
    "Show, then tell.\n"
    "- Use a markdown TABLE for comparisons (X vs Y), fenced ```code blocks``` with a language tag "
    "for code, and short blockquotes (>) for important notes, tips, or caveats.\n"
    "- Close a substantial answer with a brief **takeaway**, summary line, or sensible next step "
    "when it helps — but NEVER pad with filler, boilerplate, or repetition. Every sentence must "
    "earn its place. Be complete: anticipate the obvious follow-up and address it.\n"
    "- Aim for the clarity of a knowledgeable expert teaching a sharp beginner: warm, precise, and "
    "well-organized. All of this still obeys the LANGUAGE rule — match the user's language; a "
    "detailed Tanglish answer is detailed AND in Tanglish."
)

# The app renders ```mermaid blocks as real diagrams (see MermaidBlock), so the
# model must emit VALID mermaid for any diagram request — never ASCII art.
DIAGRAM_RULE = (
    "DIAGRAMS & FLOWCHARTS:\n"
    "- When the user asks for a flowchart, flow diagram, process flow, architecture diagram, "
    "sequence diagram, mind map, ER diagram, or any visual diagram — output it as a VALID "
    "Mermaid code block: a line with ```mermaid, then the diagram, then a closing ```. The app "
    "renders it as a real picture. NEVER use ASCII art for diagrams.\n"
    "- Pick the correct Mermaid type: 'flowchart TD' (top-down) or 'flowchart LR' (left-right) for "
    "flows/processes; 'sequenceDiagram' for interactions; 'classDiagram', 'erDiagram', 'mindmap', "
    "or 'gantt' when they fit.\n"
    "- SYNTAX MUST BE VALID (mermaid is strict): one statement per line; node ids are simple "
    "alphanumerics (A, B, step1); arrows are --> (or -->|label|); put display text in brackets "
    "like A[Start]. If a label has spaces or punctuation, WRAP IT IN DOUBLE QUOTES: "
    "A[\"Validate input\"] --> B{\"Valid?\"}. Avoid raw parentheses, slashes, semicolons, or "
    "unquoted special characters inside labels — they break the parser.\n"
    "- Keep diagrams clean (typically 4-12 nodes). Add a one-line explanation before or after if "
    "it helps, but the diagram itself is the answer."
)

# Optional response-style presets the user can pick in Settings. Appended to the
# system prompt at request time so they steer tone WITHOUT overriding accuracy.
STYLE_RULES = {
    "concise": (
        "RESPONSE STYLE: Be concise and direct. Lead with the answer, prefer short "
        "sentences and bullet points, and skip preamble and filler. Only elaborate if asked."
    ),
    "explanatory": (
        "RESPONSE STYLE: Be thorough and educational. Explain the reasoning step by step, "
        "give helpful context, and include concrete examples so the user fully understands."
    ),
    "formal": (
        "RESPONSE STYLE: Use a formal, professional tone. Avoid slang and casual phrasing; "
        "write in clear, polished, well-structured prose."
    ),
}


def _style_suffix(style: str = None, custom_instructions: str = None) -> str:
    """Build the optional style + custom-instruction block appended to the system prompt."""
    parts = []
    rule = STYLE_RULES.get((style or "").strip().lower())
    if rule:
        parts.append(rule)
    ci = (custom_instructions or "").strip()
    if ci:
        parts.append(
            "USER CUSTOM INSTRUCTIONS (honour these in every reply whenever they do not "
            "conflict with accuracy, safety, or the document context):\n" + ci[:1200]
        )
    return ("\n\n" + "\n\n".join(parts)) if parts else ""


SYSTEM_NORMAL = (
    "You are Close AI, a knowledgeable and precise AI assistant. "
    "You are an expert across technology, programming, AI/ML, science, and general knowledge. "
    "When a user mentions a technical term or acronym (such as 'RAG', 'LLM', 'API', 'GAN'), "
    "interpret it in its most common technical meaning unless the context clearly says otherwise "
    "(for example, in an AI/tech context 'RAG' means Retrieval-Augmented Generation, not music). "
    "Give accurate, clear, well-structured answers. If a question is genuinely ambiguous, "
    "briefly state your interpretation and then answer it. Calibrate the length of your reply to "
    "the question (see ANSWER DEPTH & FORMATTING below) — crisp for simple asks, thorough and "
    "well-explained for anything that wants real understanding.\n\n"
    "APP CAPABILITIES — THIS APP CAN GENERATE MEDIA:\n"
    "- IMAGES: this app HAS built-in image generation via the `/image` slash command "
    "(powered by NVIDIA NIM FLUX). When a user asks you to generate / create / draw / "
    "make / render / design an image, picture, photo, drawing, illustration, or art, do "
    "NOT reply 'I can't generate images' — instead tell them to type `/image <prompt>` "
    "(e.g. `/image a fish swimming in the ocean`). Most natural-language image requests "
    "are already auto-routed to /image by the frontend, so usually you won't see them — "
    "but if you do, suggest the command, never refuse.\n"
    "- PDFs: this app HAS built-in PDF document generation via the `/pdf` slash command. "
    "If a user wants a styled PDF document, report, resume, or brief, suggest "
    "`/pdf <topic>` instead of just writing Markdown in chat.\n"
    "- Do NOT mention `/video` — that path requires paid credits and is currently "
    "hidden in this build.\n\n"
    + LANGUAGE_RULE
    + "\n\n"
    + INSTRUCTION_FOLLOWING_RULE
    + "\n\n"
    + ACCURACY_RULE
    + "\n\n"
    + TEMPORAL_RULE
    + "\n\n"
    + CODE_RULE
    + "\n\n"
    + MATH_RULE
    + "\n\n"
    + FORMAT_DEPTH_RULE
    + "\n\n"
    + DIAGRAM_RULE
)

SYSTEM_RAG = (
    "You are Close AI, a knowledgeable and precise AI assistant with access to an uploaded document.\n\n"
    "Guidelines:\n"
    "- If the user's message is about the document, answer using the context below — accurately and without making things up.\n"
    "- If the user sends a greeting or a general question unrelated to the document, answer it naturally and helpfully using your own knowledge — do NOT say \"no context\" or refuse.\n"
    "- Interpret technical acronyms in their common technical meaning (e.g. 'RAG' = Retrieval-Augmented Generation).\n"
    "- Be accurate and clear, with depth calibrated to the question (see ANSWER DEPTH & FORMATTING).\n"
    "- " + LANGUAGE_RULE + "\n\n"
    + INSTRUCTION_FOLLOWING_RULE + "\n\n"
    + ACCURACY_RULE + "\n\n"
    + TEMPORAL_RULE + "\n\n"
    + CODE_RULE + "\n\n"
    + MATH_RULE + "\n\n"
    + FORMAT_DEPTH_RULE + "\n\n"
    + DIAGRAM_RULE + "\n\n"
    "Document Context:\n{context}"
)


# Cheap local pre-filter: only questions matching these time-sensitive signals
# are sent to the (slower) LLM router + web search. Everything else streams
# immediately — so greetings, coding and general questions answer in ~1s.
_FRESH_INFO_PATTERNS = re.compile(
    r"\b("
    r"current(?:ly)?|latest|recent(?:ly)?|nowadays|today|tonight|tomorrow|yesterday|"
    r"news|headlines?|price|prices|cost|stock|market|weather|forecast|temperature|"
    r"score|scores|standings?|fixtures?|champion|winner|trending|live|newest|"
    r"release date|just launched|up to date|as of|"
    # Office / role-holder questions are inherently about the CURRENT holder.
    r"chief minister|prime minister|president|governor|mayor|ceo|captain|"
    # More current-data triggers.
    r"happening|election|results?|update|announcement|launch(?:ed)?|releas(?:ed|e)|"
    r"upcoming|schedule|deadline|version|exchange rate|conversion|"
    # Reviews and 'best' questions (often time-sensitive — phones, laptops, models).
    r"best (?:phone|laptop|model|tool|app|service|game)|review"
    r")\b"
    r"|\bthis (?:year|month|week|quarter)\b"
    r"|\b(?:right )?now\b"
    # Any "who is / who's / who are ..." — likely a person or current role-holder.
    r"|\bwho(?:'s| is| are| s)\b"
    r"|\bwho won\b|\bwho leads\b"
    # Bare abbreviations users type (cm = chief minister, pm = prime minister).
    # The smart router still answers NO for non-current uses (e.g. '5 cm', '3 pm').
    r"|\b(?:cm|pm)\b"
    # Romanized Indian-language time-sensitive triggers.
    r"|\b(?:ippo|ipo|ipoda|innaiku|naliki|naalaiku|ipa|kalyana)\b"  # Tanglish: now/today/etc
    r"|\b(?:abhi|aaj|kal|abhinow|aajkal)\b"                          # Hinglish: now/today
    r"|\b20(?:2[4-9]|3\d)\b",
    re.IGNORECASE,
)


def _might_need_fresh_info(question: str) -> bool:
    """Fast, local check — does the question look time-sensitive at all?"""
    return bool(_FRESH_INFO_PATTERNS.search(question or ""))


def _needs_web_search(question: str, history: list = []):
    """
    Decide if the question needs live web info.
    Returns an optimized search query string if yes, else None.
    """

    if not is_search_available():
        return None

    try:
        messages = [{"role": "system", "content": ROUTER_SYSTEM}]
        # A little recent history helps resolve references like "his", "that team".
        messages.extend(history[-4:])
        messages.append({"role": "user", "content": question})

        resp = _groq_or_nvidia().chat.completions.create(
            model=ROUTER_MODEL,
            messages=messages,
            temperature=0,
            max_tokens=40
        )

        decision = (
            resp.choices[0].message.content or ""
        ).strip()

        if not decision or decision.upper() == "NO":
            return None

        return decision.strip('"').strip()

    except Exception:
        # On any router failure, fall back to the model's own knowledge.
        return None


def generate_title(question: str) -> str:
    """Generate a concise chat title from the first user message."""
    try:
        resp = _groq_or_nvidia().chat.completions.create(
            model=ROUTER_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Create a very short title (3 to 6 words, Title Case) that summarizes the "
                        "user's message. Reply with ONLY the title — no quotes, no trailing "
                        "punctuation, no preamble."
                    ),
                },
                {"role": "user", "content": (question or "")[:500]},
            ],
            temperature=0.3,
            max_tokens=20,
        )
        title = (resp.choices[0].message.content or "").strip().strip('"').strip()
        return title[:60]
    except Exception:
        return ""


def generate_followups(question: str, answer: str) -> list:
    """Suggest 3 short follow-up questions based on the last exchange."""
    try:
        resp = _groq_or_nvidia().chat.completions.create(
            model=ROUTER_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Given a question and answer, suggest exactly 3 short, natural follow-up "
                        "questions the user might ask next. Output ONLY the 3 questions, one per "
                        "line, no numbering or bullets, each under 10 words."
                    ),
                },
                {"role": "user", "content": f"Q: {question[:400]}\nA: {answer[:1500]}"},
            ],
            temperature=0.6,
            max_tokens=90,
        )
        text = resp.choices[0].message.content or ""
        out = []
        for line in text.splitlines():
            q = line.strip().lstrip("-•*0123456789. ").strip()
            if q:
                out.append(q[:100])
        return out[:3]
    except Exception:
        return []


def _chat_complete(messages: list, temperature: float = 0.3, max_tokens: int = 1024) -> str:
    """Non-streaming chat completion with automatic Groq → NVIDIA fallback.

    Groq is faster, but if its key is invalid / rate-limited / down (HTTP 401,
    429, 5xx), retry on the NVIDIA-hosted equivalent so user-facing features
    (translate, summary, …) keep working instead of silently returning empty.
    """
    attempts = []
    if groq_client is not None:
        attempts.append((groq_client, MODEL, "groq"))
    attempts.append((client, NVIDIA_CHAT_MODEL, "nvidia"))
    last_exc = None
    for attempt_client, attempt_model, provider in attempts:
        try:
            resp = attempt_client.chat.completions.create(
                model=attempt_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            out = (resp.choices[0].message.content or "").strip()
            if out:
                return out
        except Exception as exc:  # noqa: BLE001 — try the next provider
            last_exc = exc
            print(f"_chat_complete failed via {provider} ({attempt_model}): {exc}", flush=True)
            continue
    return ""


def translate_text(text: str, language: str) -> str:
    """Translate text into the target language (Groq, NVIDIA fallback)."""
    return _chat_complete(
        [
            {
                "role": "system",
                "content": (
                    f"Translate the user's text into {language}. Output ONLY the translation, "
                    "preserving meaning, tone, and any markdown/code formatting. Add no notes."
                ),
            },
            {"role": "user", "content": text[:4000]},
        ],
        temperature=0.3,
        max_tokens=2000,
    )


def summarize_conversation(history: list) -> str:
    """Produce a concise markdown summary of a conversation (Groq, NVIDIA fallback)."""
    if not history:
        return ""
    transcript = "\n\n".join(
        f"{'User' if m.get('role') == 'user' else 'Assistant'}: {(m.get('content') or '')[:2000]}"
        for m in history
        if m.get("content")
    )[:12000]
    return _chat_complete(
        [
            {
                "role": "system",
                "content": (
                    "Summarize the following conversation concisely in markdown. Start with a "
                    "one-line TL;DR, then 3-6 bullet points of the key topics, decisions, and "
                    "any action items or open questions. Be faithful; add nothing not discussed."
                ),
            },
            {"role": "user", "content": transcript},
        ],
        temperature=0.3,
        max_tokens=900,
    )


def _strip_code_output(out: str) -> str:
    """Clean raw model output into pure file contents — drop reasoning blocks and
    stray markdown code fences that some models add despite instructions."""
    out = (out or "").strip()
    if not out:
        return ""
    # Remove <think>...</think> reasoning blocks (some models emit them).
    out = re.sub(r"<think>.*?</think>", "", out, flags=re.DOTALL).strip()
    # Strip a leading ```lang fence and the matching trailing ``` fence.
    if out.startswith("```"):
        lines = out.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        out = "\n".join(lines)
    return out.strip()


def _code_complete(messages: list, temperature: float, max_tokens: int) -> str:
    """Run a code task on the powerful code model, falling back to MODEL on
    timeout / unavailability. Uses a longer per-call timeout since the 480B MoE
    coder is slower than the chat workhorse. Returns '' if every model fails."""
    models = [CODE_MODEL] + ([MODEL] if MODEL != CODE_MODEL else [])
    for m in models:
        try:
            resp = client.with_options(timeout=120.0).chat.completions.create(
                model=m,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            out = (resp.choices[0].message.content or "").strip()
            if out:
                return out
        except Exception:
            continue
    return ""


def _history_block(history: list, limit: int = 6) -> str:
    """Render recent Code-mode chat turns as a compact context block so the agent
    can resolve references ('it', 'that button', 'the function above') across
    turns. Empty string when there's no usable history."""
    if not history:
        return ""
    turns = []
    for h in history[-limit:]:
        role = "User" if str(h.get("role")) == "user" else "Assistant"
        content = str(h.get("content", "")).strip().replace("\r", "")[:500]
        if content:
            turns.append(f"{role}: {content}")
    if not turns:
        return ""
    return (
        "RECENT CONVERSATION (context only — use it to resolve references like "
        "'it'/'that'; do not repeat past work):\n" + "\n".join(turns) + "\n\n"
    )


def edit_code_file(filename: str, content: str, instruction: str, history: list = None) -> str:
    """Return the COMPLETE updated contents of a code file per the instruction.

    Uses the powerful code model with an automatic fallback to MODEL, a generous
    output budget so large files aren't truncated, recent-conversation memory,
    and output cleaning.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert code editor. You are given a file's full contents and an "
                "instruction. Return ONLY the complete, updated file contents — no explanations, "
                "no commentary, and NO markdown code fences. If the CURRENT FILE section is empty, "
                "CREATE the file's full contents from scratch to satisfy the instruction. Preserve "
                "everything unrelated to the change, keep the existing style and indentation, and "
                "make the smallest correct edit that satisfies the instruction. Output the entire "
                "file — never abbreviate with comments like '... rest unchanged ...'."
            ),
        },
        {
            "role": "user",
            "content": (
                f"{_history_block(history)}File: {filename}\n\nInstruction: {instruction}\n\n"
                f"--- CURRENT FILE ---\n{content[:48000]}"
            ),
        },
    ]
    return _strip_code_output(_code_complete(messages, 0.1, 8192))


def plan_code_changes(tree: list, instruction: str, history: list = None) -> dict:
    """Classify a Code-mode request and pick the minimal files to touch.

    Returns {mode, files, notes}:
      - mode 'answer' → user asked a QUESTION about the code; `files` lists the
        files worth reading to answer it.
      - mode 'edit'   → user wants changes; `files` is the minimal set to
        create/edit, each with an action + one-line reason.

    `history` (recent chat turns) lets follow-ups resolve references.
    """
    import json

    tree_str = "\n".join([str(p) for p in (tree or [])][:800])
    try:
        resp = _groq_or_nvidia().chat.completions.create(
            model=PLAN_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a coding agent working in a user's project. Given the project's "
                        "file list and a request, respond with ONLY JSON (no prose, no markdown fences).\n\n"
                        "First decide the MODE:\n"
                        "- 'answer' if the user is asking a QUESTION about the code (explain, where is, "
                        "how does, what does, why, review, find) and wants NO file changes.\n"
                        "- 'edit' if the user wants to CREATE, ADD, CHANGE, FIX, REFACTOR, or DELETE code.\n\n"
                        "Then choose the MINIMAL set of relevant files (at most 8).\n\n"
                        "JSON shape (exactly):\n"
                        "{\"mode\": \"edit\" or \"answer\", "
                        "\"files\": [{\"path\": \"relative/path\", \"action\": \"edit\" or \"create\", "
                        "\"reason\": \"short why\"}], "
                        "\"notes\": \"one-sentence plan or summary\"}\n\n"
                        "For 'answer' mode, action is always 'edit' (meaning: read this file to answer). "
                        "Prefer existing paths; include a 'create' path only when a new file is clearly "
                        "needed. Keep each reason under 12 words. Output ONLY the JSON object."
                    ),
                },
                {
                    "role": "user",
                    "content": f"{_history_block(history)}REQUEST: {instruction}\n\nPROJECT FILES:\n{tree_str}",
                },
            ],
            temperature=0.1,
            max_tokens=700,
        )
        raw = (resp.choices[0].message.content or "").strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw[:4].lower() == "json":
                raw = raw[4:]
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1:
            raw = raw[start : end + 1]
        data = json.loads(raw)

        mode = "answer" if str(data.get("mode", "")).lower() == "answer" else "edit"
        files = []
        for f in (data.get("files") or [])[:8]:
            if isinstance(f, dict) and f.get("path"):
                files.append(
                    {
                        "path": str(f["path"]).strip(),
                        "action": "create" if str(f.get("action", "")).lower() == "create" else "edit",
                        "reason": str(f.get("reason", "")).strip()[:120],
                    }
                )
            elif isinstance(f, str) and f.strip():
                files.append({"path": f.strip(), "action": "edit", "reason": ""})
        return {"mode": mode, "files": files, "notes": str(data.get("notes", ""))[:240]}
    except Exception:
        return {"mode": "edit", "files": [], "notes": ""}


def answer_code_question(question: str, files: list, history: list = None) -> str:
    """Answer a question about the user's code. `files` = [{path, content}].
    `history` carries recent chat turns so follow-up questions keep context."""
    blocks = []
    budget = 22000
    for f in files or []:
        path = str(f.get("path", "")).strip()
        content = str(f.get("content", ""))
        if not path:
            continue
        block = f"=== {path} ===\n{content[:8000]}"
        if budget - len(block) < 0:
            break
        budget -= len(block)
        blocks.append(block)
    context = "\n\n".join(blocks) if blocks else "(no file contents were provided)"
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert software engineer helping a developer understand their "
                "codebase. Answer the question accurately using the provided file contents. "
                "Be concise and concrete: reference file names and functions, quote short "
                "snippets in fenced code blocks when helpful. If the answer isn't in the "
                "provided files, say what you'd need to see. Use markdown."
            ),
        },
        {"role": "user", "content": f"{_history_block(history)}QUESTION: {question}\n\nFILES:\n{context}"},
    ]
    return _code_complete(messages, 0.3, 2000)


def _today_str() -> str:
    """Today's real date, e.g. 'Monday, June 01, 2026'."""
    return datetime.now().strftime("%A, %B %d, %Y")


# ── Per-chat web-context memory (SQLite-backed) ─────────────────────────────
# When a turn fetches live web results, we persist them per chat. Follow-up
# questions in the same conversation that don't themselves trip the search gate
# (e.g. "are you sure?", "no team has won it") then reuse this fresh context —
# so the assistant stays consistent with the current data instead of
# "forgetting" it and deferring. Persisted (not in-memory) so it survives
# uvicorn --reload and multiple workers. Time-limited so it never serves
# stale data.
_WEB_CACHE_TTL_SECONDS = 1800  # 30 minutes


def _remember_web_context(chat_id: str, results: str) -> None:
    """Persist the latest live web results for a chat."""
    if not chat_id or not results:
        return
    db = SessionLocal()
    try:
        rec = db.get(ChatWebContext, chat_id)
        if rec is None:
            db.add(ChatWebContext(chat_id=chat_id, results=results, updated_at=datetime.utcnow()))
        else:
            rec.results = results
            rec.updated_at = datetime.utcnow()
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()


def _recall_web_context(chat_id: str) -> str:
    """Return recent (non-expired) web results for a chat, or '' if none."""
    if not chat_id:
        return ""
    db = SessionLocal()
    try:
        rec = db.get(ChatWebContext, chat_id)
        if rec is None:
            return ""
        if (datetime.utcnow() - rec.updated_at).total_seconds() > _WEB_CACHE_TTL_SECONDS:
            return ""
        return rec.results
    except Exception:
        return ""
    finally:
        db.close()


def delete_web_context(chat_id: str) -> None:
    """Remove a chat's cached web context (called when the chat is deleted)."""
    if not chat_id:
        return
    db = SessionLocal()
    try:
        rec = db.get(ChatWebContext, chat_id)
        if rec is not None:
            db.delete(rec)
            db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()


def _ground_prompt(base_system: str, question: str, history: list, chat_id: str = None, web_enabled: bool = True) -> str:
    """
    Always grounds the prompt with today's real date (so the model never assumes
    it is still at its training cutoff). When the question looks time-sensitive,
    runs a fresh web search; otherwise reuses recent web results already fetched
    earlier in this same conversation, so follow-ups stay consistent with the
    current data instead of deferring.

    When `web_search` is False the model answers from its own knowledge only
    (date is still injected) — used when the user turns web access off.
    """

    grounded = base_system + f"\n\nToday's real date is {_today_str()}."

    # User turned web access off for this chat — knowledge-only answer.
    if not web_enabled:
        return grounded

    # Fresh search only when the question looks time-sensitive — this is what
    # keeps the common case (greetings, coding, general Q) instant.
    if is_search_available() and _might_need_fresh_info(question):
        query = _needs_web_search(question, history)
        if query:
            results = run_web_search(query)
            if results:
                _remember_web_context(chat_id, results)
                return (
                    grounded
                    + "\n\nThe following are live web search results. Prefer them over outdated "
                    + "training knowledge for any current, recent, or time-sensitive facts:\n"
                    + results
                )

    # No fresh search this turn — reuse current data already fetched earlier in
    # this conversation so the assistant stays consistent and helpful.
    recalled = _recall_web_context(chat_id)
    if recalled:
        return (
            grounded
            + "\n\nThese are web search results fetched moments ago, earlier in this same "
            + "conversation. Treat them as current, verified facts when relevant to the "
            + "question:\n"
            + recalled
        )

    return grounded


# Chunks below this cosine similarity to the question are treated as irrelevant
# (so off-topic questions don't show misleading "sources" from the PDF).
_RAG_MIN_SIMILARITY = 0.3


def _safe_metadata_filename(name: str) -> str:
    """Match upload.py's filename normalization for older saved chat metadata."""
    base = re.split(r"[\\/]+", str(name or ""))[-1]
    base = base.replace("\\", "_")
    base = re.sub(r"[^A-Za-z0-9._-]", "_", base).strip("._")
    return base or "upload.pdf"


def _filename_candidates(active_docs: list = None) -> list:
    """Return both display names and sanitized metadata names for Chroma filters."""
    seen = set()
    candidates = []
    for name in active_docs or []:
        for candidate in (str(name or "").strip(), _safe_metadata_filename(name)):
            if candidate and candidate not in seen:
                seen.add(candidate)
                candidates.append(candidate)
    return candidates


def _empty_query_results() -> dict:
    return {"documents": [[]], "metadatas": [[]], "embeddings": [[]]}


def _query_collection(collection, q_emb: list, n: int, where: dict = None) -> dict:
    """Query Chroma, retrying without a stale filename filter when needed."""
    if n <= 0:
        return _empty_query_results()

    try:
        results = collection.query(
            query_embeddings=[q_emb],
            n_results=n,
            where=where,
            include=["documents", "metadatas", "embeddings"],
        )
    except Exception:
        if not where:
            return _empty_query_results()
        try:
            results = collection.query(
                query_embeddings=[q_emb],
                n_results=n,
                include=["documents", "metadatas", "embeddings"],
            )
        except Exception:
            return _empty_query_results()

    documents = (results.get("documents") or [[]])[0]
    if where and not documents:
        try:
            return collection.query(
                query_embeddings=[q_emb],
                n_results=n,
                include=["documents", "metadatas", "embeddings"],
            )
        except Exception:
            return _empty_query_results()
    return results


def _retrieve_relevant(collection, question: str, active_docs: list = None):
    """
    Query the document collection and keep ONLY chunks that are actually similar
    to the question (cosine similarity). Returns (context_text, sources_list);
    both empty if nothing is relevant — so the answer falls back to general
    knowledge with no misleading source chips.

    When `active_docs` is given, only chunks from those filenames are searched
    (the user's multi-document picker).
    """
    q_emb = embed_query(question)
    # Retrieve plenty of chunks (capped at what the doc has) so specific
    # details — e.g. a project name buried mid-resume — aren't missed. We
    # then filter by cosine similarity below, so over-retrieving here is safe
    # and meaningfully boosts recall on multi-section / longer docs.
    n = min(collection.count() or 16, 16)
    where = {"filename": {"$in": _filename_candidates(active_docs)}} if active_docs else None
    results = _query_collection(collection, q_emb, n, where)
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    embeddings = (results.get("embeddings") or [[]])[0]

    q = np.asarray(q_emb, dtype=float)
    q_norm = float(np.linalg.norm(q)) + 1e-9

    # Always pass the retrieved chunks to the model so it can answer from the PDF.
    context = "\n\n".join(documents)

    # Only the chunks genuinely similar to the question become source chips,
    # so off-topic questions don't show misleading citations.
    sources = []
    for i in range(len(documents)):
        sim = 1.0
        if i < len(embeddings) and embeddings[i] is not None:
            v = np.asarray(embeddings[i], dtype=float)
            sim = float(np.dot(v, q) / ((float(np.linalg.norm(v)) + 1e-9) * q_norm))
        if sim >= _RAG_MIN_SIMILARITY:
            sources.append({"content": documents[i], "metadata": metadatas[i]})

    return context, sources


def _normal_chat(question: str, history: list = [], chat_id: str = None) -> dict:
    """No PDF uploaded — behave as a general AI assistant."""

    system_prompt = _ground_prompt(SYSTEM_NORMAL, question, history, chat_id)

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": question})

    completion = _groq_or_nvidia().chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.4,
        max_tokens=4096
    )

    return {
        "answer": (
            completion
            .choices[0]
            .message
            .content
        ),
        "sources": []
    }


def _rag_chat(collection, question: str, history: list = [], chat_id: str = None) -> dict:
    """PDF uploaded — answer from the relevant document chunks (if any)."""

    context, sources = _retrieve_relevant(collection, question)

    rag_system = SYSTEM_RAG.replace("{context}", context)
    rag_system = _ground_prompt(rag_system, question, history, chat_id)

    messages = [{"role": "system", "content": rag_system}]
    messages.extend(history)
    messages.append({"role": "user", "content": question})

    completion = _groq_or_nvidia().chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=4096,
    )

    answer = completion.choices[0].message.content

    return {"answer": answer, "sources": sources}


def ask_question(
    chat_id: str,
    question: str,
    history: list = []
) -> dict:
    """
    Routes the question based on whether a PDF has been uploaded:
      - Empty collection  →  normal AI chat
      - Has documents     →  RAG document Q&A (with graceful fallback for greetings)
    """

    collection = get_or_create_collection(chat_id)

    if collection.count() == 0:
        return _normal_chat(question, history, chat_id)

    return _rag_chat(collection, question, history, chat_id)


# ── Streaming (token-by-token) — makes answers feel instant ─────────────────

def _sse(payload: dict) -> str:
    """Format a Server-Sent Event line."""
    return f"data: {json.dumps(payload)}\n\n"


def _chat_stream_attempts(model: str):
    """Return provider/model attempts for streaming chat."""
    if model in (VISION_MODEL, CODE_MODEL):
        return [(client, model, "nvidia")]

    attempts = []
    if groq_client is not None:
        attempts.append((groq_client, model, "groq"))
    attempts.append((client, NVIDIA_CHAT_MODEL, "nvidia"))
    return attempts


def _log_stream_start_error(provider: str, model: str, exc: Exception) -> None:
    """Print a useful server-side error without exposing request content or keys."""
    msg = str(exc).replace("\n", " ")[:500]
    print(
        f"Chat stream start failed via {provider} model={model}: "
        f"{exc.__class__.__name__}: {msg}",
        flush=True,
    )


def _stream_completion(messages: list, temperature: float, model: str = MODEL):
    """Yield SSE 'token' events from a streaming chat completion.

    Catches errors at both stream-open and per-chunk so a mid-stream failure
    surfaces as a clean SSE 'error' event instead of an uncaught exception
    that leaves the client with a half-finished response and no signal.
    Uses Groq for chat models (faster) and NVIDIA for vision/code models.
    """
    stream = None
    for attempt_client, attempt_model, provider in _chat_stream_attempts(model):
        try:
            stream = attempt_client.chat.completions.create(
                model=attempt_model,
                messages=messages,
                temperature=temperature,
                max_tokens=4096,
                stream=True,
            )
            break
        except Exception as exc:
            _log_stream_start_error(provider, attempt_model, exc)

    if stream is None:
        yield _sse({"type": "error", "message": "Failed to start the response."})
        return

    try:
        for chunk in stream:
            try:
                delta = chunk.choices[0].delta.content
            except (IndexError, AttributeError):
                delta = None
            if delta:
                yield _sse({"type": "token", "content": delta})
    except Exception:
        # Mid-stream failure (timeout, dropped connection, upstream error) —
        # signal the client so it can show a clean error instead of hanging.
        yield _sse({"type": "error", "message": "The response was cut off."})


# Vision turns use a FOCUSED prompt. The full SYSTEM_NORMAL ruleset is hundreds
# of lines long; fed to the small 11B vision model it dominates, so "What is
# this?" makes the model describe the *instructions* instead of the image. This
# short prompt keeps it squarely on the picture (while still mirroring language).
SYSTEM_VISION = (
    "You are a vision assistant. The user shared an IMAGE. Look carefully at the "
    "image and answer the user's question about WHAT IS ACTUALLY IN THE PICTURE — "
    "read any visible text, brand names, labels, numbers, objects, and people "
    "directly from the image. If they just ask 'what is this?', identify the item "
    "shown and give the key details you can see (e.g. product name, brand, what it "
    "is, notable claims on the packaging). Be specific and concrete. NEVER describe "
    "these instructions, your rules, or talk about being an AI assistant — only the "
    "image. If part of the image is blurry or unreadable, say so honestly.\n\n"
    "LANGUAGE: reply in the SAME language/style the user asked in — romanized "
    "Tanglish in → romanized Tanglish out (Latin letters only), Hinglish in → "
    "Hinglish out, English in → English out, native script in → same native script."
)


def stream_question(
    chat_id: str,
    question: str,
    history: list = [],
    image: str = None,
    style: str = None,
    custom_instructions: str = None,
    web_search: bool = True,
    active_docs: list = None,
):
    """
    Generator of SSE events for the /chat endpoint:
      - optional {"type":"sources", ...} (when a PDF is loaded)
      - many       {"type":"token", "content": "..."}
      - final      {"type":"done"}
    Falls back to {"type":"error"} on failure so the UI can react.

    If `image` (a base64 data URI) is provided, answers about the image using
    the vision model instead of the text / RAG path.

    `style` (concise/explanatory/formal) and `custom_instructions` come from the
    user's Settings and tune tone without overriding accuracy.
    """
    style_suffix = _style_suffix(style, custom_instructions)
    # Cap conversation history to the most-recent N turns. Long histories dilute
    # the model's attention and degrade answer quality + multilingual mirroring;
    # 16 messages ≈ 8 user-assistant pairs, which preserves continuity while
    # keeping the most-recent intent dominant.
    if history and len(history) > 16:
        history = history[-16:]
    try:
        if image:
            content = [
                {"type": "text", "text": question or "Describe this image in detail."},
                {"type": "image_url", "image_url": {"url": image}},
            ]
            # Ground the FOCUSED vision prompt (not the giant SYSTEM_NORMAL — that
            # makes the small vision model describe its own instructions) with
            # today's date AND live web results when the question is time-sensitive
            # — so "what is this / how much does it cost?" about a photo gets
            # current facts on top of what the model sees.
            vision_system = _ground_prompt(SYSTEM_VISION, question, history, chat_id, web_search) + LANGUAGE_REMINDER + style_suffix
            messages = [{"role": "system", "content": vision_system}]
            messages.extend(history)
            messages.append({"role": "user", "content": content})
            yield from _stream_completion(messages, 0.4, model=VISION_MODEL)
            yield _sse({"type": "done"})
            return

        collection = get_or_create_collection(chat_id)

        if collection.count() == 0:
            system_prompt = _ground_prompt(SYSTEM_NORMAL, question, history, chat_id, web_search) + LANGUAGE_REMINDER + style_suffix
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(history)
            messages.append({"role": "user", "content": question})
            # Lower temperature → tighter, more accurate factual + multilingual
            # responses. 0.3 strikes the balance: still feels conversational,
            # but the model is less likely to drift / hallucinate / mix languages.
            yield from _stream_completion(messages, 0.3)
        else:
            context, sources = _retrieve_relevant(collection, question, active_docs)

            # Only show source chips when the document actually had relevant chunks.
            if sources:
                yield _sse({"type": "sources", "sources": sources})

            rag_system = _ground_prompt(
                SYSTEM_RAG.replace("{context}", context), question, history, chat_id, web_search
            ) + LANGUAGE_REMINDER + style_suffix
            messages = [{"role": "system", "content": rag_system}]
            messages.extend(history)
            messages.append({"role": "user", "content": question})
            yield from _stream_completion(messages, 0.3)

        yield _sse({"type": "done"})

    except Exception:
        yield _sse({"type": "error", "message": "Failed to generate a response."})
