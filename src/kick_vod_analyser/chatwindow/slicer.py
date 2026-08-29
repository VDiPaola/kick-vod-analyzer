"""Chat window construction: raw messages to a token-capped prompt block."""

from __future__ import annotations

import re
from collections import OrderedDict
from typing import Iterable, Sequence

from ..config import ChatSettings
from ..models import ChatMessage, ChatWindow

BOT_COMMAND_RE = re.compile(r"^\s*[!$#][a-z0-9_]{1,24}\b", re.IGNORECASE)
URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")
NON_ALNUM_RE = re.compile(r"[^a-z0-9 ]+")

BOT_USERNAMES = frozenset(
    {"botrix", "streamelements", "nightbot", "streamlabs", "moobot", "fossabot", "kickbot"}
)

# Phrases that mark a message as directly informative about what is on screen.
SIGNAL_PHRASES = (
    "what game",
    "what is this",
    "what's this",
    "whats this",
    "what song",
    "what video",
    "who is this",
    "what movie",
    "what is he watching",
    "what are we watching",
    "name of the",
    "link",
    "brb",
    "afk",
    "be right back",
    "he left",
    "where did he go",
    "stream is frozen",
    "stream froze",
    "no sound",
    "audio",
    "lagging",
    "loading",
    "queue",
    "lobby",
    "ranked",
    "gameplay",
    "reaction",
    "react",
    "youtube",
    "tiktok",
    "twitter",
    "reddit",
    "slots",
    "bonus",
    "casino",
    "roulette",
)

LOW_SIGNAL_TOKENS = frozenset(
    {
        "lol",
        "lmao",
        "xd",
        "w",
        "l",
        "ez",
        "gg",
        "f",
        "yes",
        "no",
        "true",
        "same",
        "omg",
        "wtf",
        "sheesh",
        "bro",
        "ok",
        "okay",
        "hi",
        "hello",
        "yo",
    }
)


def canonical_form(text: str) -> str:
    """Collapse a message to a key that treats spam variants as identical."""
    lowered = URL_RE.sub(" url ", text.lower())
    lowered = NON_ALNUM_RE.sub(" ", lowered)
    collapsed = WHITESPACE_RE.sub(" ", lowered).strip()
    if not collapsed:
        return ""
    words = collapsed.split(" ")
    deduped = [words[0]]
    for word in words[1:]:
        if word != deduped[-1]:
            deduped.append(word)
    return " ".join(deduped)


def is_noise(message: ChatMessage, settings: ChatSettings) -> bool:
    """Drop bot output, commands, and content-free single tokens."""
    text = message.text.strip()
    if len(text) < settings.min_message_length:
        return True
    if message.username.lower() in BOT_USERNAMES:
        return True
    if settings.drop_bot_commands and BOT_COMMAND_RE.match(text):
        return True
    canonical = canonical_form(text)
    if not canonical:
        return True
    if canonical in LOW_SIGNAL_TOKENS and not message.emotes:
        return True
    return False


SIGNAL_PHRASE_BONUS = 3.0
MAX_REPETITION_BONUS = 2.0


def relevance_score(text: str, repeat_count: int) -> float:
    """Rank messages by how much they say about what is on screen.

    Signal phrases dominate. Repetition contributes on a log scale, capped below
    the signal bonus: a spam wave should surface once as a mood indicator and
    never displace a viewer naming what is on screen, however large the chat.
    """
    from math import log

    lowered = text.lower()
    score = min(MAX_REPETITION_BONUS, log(max(1, repeat_count)) * 0.6)
    for phrase in SIGNAL_PHRASES:
        if phrase in lowered:
            score += SIGNAL_PHRASE_BONUS
            break
    if "?" in text:
        score += 0.8
    word_count = len(lowered.split())
    if 3 <= word_count <= 20:
        score += 1.0
    elif word_count > 40:
        score -= 1.0
    return score


def condense(messages: Sequence[ChatMessage], settings: ChatSettings) -> list[str]:
    """Deduplicate, rank, and cap a message window into prompt lines.

    Repeats collapse into a single line with a multiplier, which is what makes a
    45 second window of a 20k viewer chat fit inside a few hundred tokens.
    """
    kept = [m for m in messages if not is_noise(m, settings)]
    if not kept:
        return []

    groups: OrderedDict[str, dict] = OrderedDict()
    for message in kept:
        key = canonical_form(message.text)
        entry = groups.get(key)
        if entry is None:
            groups[key] = {
                "text": message.text,
                "count": 1,
                "first_offset": message.offset_seconds,
            }
        else:
            entry["count"] += 1
            if len(message.text) > len(entry["text"]):
                entry["text"] = message.text

    scored = [
        (relevance_score(entry["text"], entry["count"]), entry["first_offset"], entry)
        for entry in groups.values()
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    selected = scored[: settings.max_lines]
    selected.sort(key=lambda item: item[1])

    lines: list[str] = []
    for _, offset, entry in selected:
        text = WHITESPACE_RE.sub(" ", entry["text"]).strip()
        if len(text) > 160:
            text = text[:157] + "..."
        suffix = f" (x{entry['count']})" if entry["count"] > 1 else ""
        lines.append(f"[{format_offset(offset)}] {text}{suffix}")
    return lines


def format_offset(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def build_window(
    messages: Iterable[ChatMessage],
    offset_seconds: float,
    settings: ChatSettings,
) -> ChatWindow:
    """Build the condensed chat block for one sample point."""
    window_messages = list(messages)
    return ChatWindow(
        offset_seconds=offset_seconds,
        lines=condense(window_messages, settings),
        message_count=len(window_messages),
        unique_chatters=len({m.username for m in window_messages}),
    )
