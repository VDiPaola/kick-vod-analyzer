"""System prompt, JSON schema, and per-sample user prompt construction."""

from __future__ import annotations

from ..models import ChatWindow, PrimaryCategory

SYSTEM_PROMPT = """You are an expert live stream activity classifier.

You receive one composite screenshot and an optional synchronised chat excerpt.
The screenshot is a 2x2 grid of four frames from the same stream, captured over
a 12 second window and numbered in the top-left corner of each cell:

  cell 1 = T-6s (top left)      cell 2 = T-2s (top right)
  cell 3 = T+2s (bottom left)   cell 4 = T+6s (bottom right)

Read the cells in that order to judge motion, menu transitions, and whether the
scene is static. Identify the streamer's activity, the specific software, game,
or media on screen, the granular sub-state, and whether the streamer is away.

Rules:
- Read on-screen text. Game HUDs, window titles, browser URLs, video titles, and
  overlay text are the strongest evidence available.
- A browser or video player showing someone else's content is
  "Reaction / Media Share", not "Just Chatting / Podcast".
- Loading screens, black frames, and short cutscenes between gameplay remain the
  game they belong to. Do not report "Technical Difficulties / Offline" unless
  the stream itself is clearly broken.
- Set is_afk_or_brb only when the streamer is absent: a BRB or intermission
  card, an empty chair, or a static placeholder scene.
- If evidence is thin, lower confidence_score rather than inventing a title.
- Use an empty string for specific_title_or_context when no specific title is
  identifiable.

Output MUST conform exactly to the provided JSON schema. Return JSON only."""

CATEGORY_VALUES = [category.value for category in PrimaryCategory]

CLASSIFICATION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "primary_category": {"type": "string", "enum": CATEGORY_VALUES},
        "specific_title_or_context": {
            "type": "string",
            "description": (
                "Specific game title, video title or creator, website, or software name. "
                "Empty string when nothing specific is identifiable."
            ),
        },
        "sub_activity": {
            "type": "string",
            "description": (
                "Granular state, for example: In-Game Match, Matchmaking Queue, Main Menu, "
                "Reading Reddit, Watching YouTube Video, Eating Food, Cooking."
            ),
        },
        "is_streamer_on_screen": {"type": "boolean"},
        "is_afk_or_brb": {"type": "boolean"},
        "confidence_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "visual_evidence": {
            "type": "string",
            "description": "One sentence naming the visual markers that drove the decision.",
        },
    },
    "required": [
        "primary_category",
        "specific_title_or_context",
        "sub_activity",
        "is_streamer_on_screen",
        "is_afk_or_brb",
        "confidence_score",
        "visual_evidence",
    ],
    "additionalProperties": False,
}


def openai_response_format() -> dict:
    """Strict structured-output wrapper for the OpenAI chat completions API."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "stream_activity_classification",
            "strict": True,
            "schema": CLASSIFICATION_SCHEMA,
        },
    }


def gemini_response_schema() -> dict:
    """Gemini rejects additionalProperties, so strip it from the schema copy."""
    schema = {k: v for k, v in CLASSIFICATION_SCHEMA.items() if k != "additionalProperties"}
    schema["properties"] = {
        key: {k: v for k, v in value.items() if k != "description"} | (
            {"description": value["description"]} if "description" in value else {}
        )
        for key, value in CLASSIFICATION_SCHEMA["properties"].items()
    }
    return schema


def format_offset(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def build_user_prompt(
    offset_seconds: float,
    chat_window: ChatWindow | None,
    *,
    channel_slug: str = "",
    stream_title: str = "",
) -> str:
    """Assemble the per-sample text block that accompanies the grid image."""
    parts = [f"VOD timestamp: {format_offset(offset_seconds)}"]
    if channel_slug:
        parts.append(f"Channel: {channel_slug}")
    if stream_title:
        parts.append(f"Stream title: {stream_title[:200]}")

    parts.append("")
    if chat_window is None or chat_window.is_empty:
        parts.append(
            "Chat excerpt: unavailable. Classify from the screenshot alone and "
            "reduce confidence_score where the visuals are ambiguous."
        )
    else:
        parts.append("Chat excerpt for the surrounding 90 seconds:")
        parts.append(chat_window.render())

    parts.append("")
    parts.append("Classify this moment and return JSON matching the schema.")
    return "\n".join(parts)
