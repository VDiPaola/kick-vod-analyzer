"""Chat replay ingestion.

Kick exposes no documented VOD chat history API. The replay the web player
renders is served by an undocumented, Cloudflare-fronted endpoint that can
change without notice, so chat is treated as optional enrichment: every source
degrades to an empty index rather than failing the run.
"""

from __future__ import annotations

import bisect
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..models import ChatMessage, VodInfo
from .http import build_client

log = logging.getLogger(__name__)

MESSAGES_API = "https://kick.com/api/v2/channels/{channel_id}/messages"


class ChatIndex:
    """Offset-ordered chat messages with O(log n) window slicing."""

    def __init__(self, messages: Iterable[ChatMessage] = ()) -> None:
        self._messages: list[ChatMessage] = sorted(messages, key=lambda m: m.offset_seconds)
        self._offsets: list[float] = [m.offset_seconds for m in self._messages]

    def __len__(self) -> int:
        return len(self._messages)

    def __bool__(self) -> bool:
        return bool(self._messages)

    @property
    def messages(self) -> Sequence[ChatMessage]:
        return tuple(self._messages)

    def window(self, center_seconds: float, radius_seconds: float) -> list[ChatMessage]:
        """Return messages within the radius either side of a VOD offset."""
        low = bisect.bisect_left(self._offsets, center_seconds - radius_seconds)
        high = bisect.bisect_right(self._offsets, center_seconds + radius_seconds)
        return self._messages[low:high]

    def to_jsonl(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for message in self._messages:
                handle.write(message.model_dump_json() + "\n")
        return path

    @classmethod
    def from_jsonl(cls, path: Path) -> "ChatIndex":
        messages: list[ChatMessage] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    messages.append(ChatMessage.model_validate_json(stripped))
        return cls(messages)


class ChatSource(ABC):
    """Strategy for obtaining chat replay for a VOD."""

    name = "chat"

    @abstractmethod
    def fetch(self, vod: VodInfo) -> ChatIndex: ...


class NullChatSource(ChatSource):
    """No chat. The classifier runs vision-only."""

    name = "none"

    def fetch(self, vod: VodInfo) -> ChatIndex:
        return ChatIndex()


class JsonlChatSource(ChatSource):
    """Load chat from a local file produced by any external scraper."""

    name = "file"

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def fetch(self, vod: VodInfo) -> ChatIndex:
        if not self.path.exists():
            log.warning("chat file %s not found; continuing without chat", self.path)
            return ChatIndex()
        records = load_records(self.path.read_text(encoding="utf-8"))
        messages = [
            m
            for m in (normalise_record(r, vod.started_at_epoch) for r in records)
            if m is not None
        ]
        log.info("loaded %d chat messages from %s", len(messages), self.path)
        return ChatIndex(messages)


class KickReplayChatSource(ChatSource):
    """Walk the undocumented Kick replay endpoint forward through the VOD.

    The endpoint returns a page of messages at or after start_time. Paging
    advances by the newest message seen, with a fixed step forward when a page
    comes back empty so quiet stretches do not stall the walk.
    """

    name = "kick"

    def __init__(
        self,
        *,
        timeout: float = 30.0,
        page_step_seconds: float = 60.0,
        max_pages: int = 5000,
        max_empty_pages: int = 40,
    ) -> None:
        self.timeout = timeout
        self.page_step_seconds = page_step_seconds
        self.max_pages = max_pages
        self.max_empty_pages = max_empty_pages

    def fetch(self, vod: VodInfo) -> ChatIndex:
        if not vod.channel_id:
            log.warning("no channel id resolved; skipping Kick chat replay")
            return ChatIndex()
        if not vod.started_at_epoch:
            log.warning("no VOD start time resolved; skipping Kick chat replay")
            return ChatIndex()

        url = MESSAGES_API.format(channel_id=vod.channel_id)
        client = build_client(self.timeout)
        seen: set[str] = set()
        messages: list[ChatMessage] = []
        cursor = vod.started_at_epoch
        end = vod.started_at_epoch + vod.duration_seconds
        empty_pages = 0

        try:
            for _ in range(self.max_pages):
                if cursor >= end or empty_pages >= self.max_empty_pages:
                    break
                page = self.request_page(client, url, cursor)
                if page is None:
                    break
                fresh = 0
                newest = cursor
                for record in page:
                    identifier = str(record.get("id") or record.get("uuid") or "")
                    if identifier and identifier in seen:
                        continue
                    if identifier:
                        seen.add(identifier)
                    message = normalise_record(record, vod.started_at_epoch)
                    if message is None or message.offset_seconds > vod.duration_seconds:
                        continue
                    messages.append(message)
                    newest = max(newest, vod.started_at_epoch + message.offset_seconds)
                    fresh += 1
                if fresh == 0:
                    empty_pages += 1
                    cursor += self.page_step_seconds
                else:
                    empty_pages = 0
                    cursor = max(newest + 0.001, cursor + 1.0)
        except Exception as exc:
            log.warning("Kick chat replay aborted after %d messages: %s", len(messages), exc)
        finally:
            client.close()

        log.info("collected %d chat messages for %s", len(messages), vod.vod_id)
        return ChatIndex(messages)

    def request_page(self, client: Any, url: str, cursor: float) -> list[dict[str, Any]] | None:
        response = client.get(url, params={"start_time": int(cursor)})
        if response.status_code != 200:
            log.warning("Kick chat replay returned %s; stopping", response.status_code)
            return None
        payload = response.json()
        if isinstance(payload, dict):
            data = payload.get("data")
            if isinstance(data, dict):
                return list(data.get("messages") or [])
            if isinstance(data, list):
                return data
            return list(payload.get("messages") or [])
        return list(payload or [])


def load_records(raw: str) -> list[dict[str, Any]]:
    """Accept a JSON array, a JSON object wrapper, or JSON Lines."""
    text = raw.strip()
    if not text:
        return []
    if text.startswith("["):
        return list(json.loads(text))
    if text.startswith("{") and len(text.splitlines()) == 1:
        payload = json.loads(text)
        for key in ("messages", "comments", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        return [payload]
    records: list[dict[str, Any]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            records.append(json.loads(stripped))
    return records


def normalise_record(record: dict[str, Any], started_at_epoch: float | None) -> ChatMessage | None:
    """Map a raw chat record onto the internal model.

    Handles the internal Kick payload, chat-downloader output, and this
    package's own JSONL export.
    """
    if not isinstance(record, dict):
        return None

    text = str(record.get("content") or record.get("message") or record.get("text") or "").strip()
    if not text:
        return None

    sender = record.get("sender") if isinstance(record.get("sender"), dict) else {}
    author = record.get("author") if isinstance(record.get("author"), dict) else {}
    username = str(
        record.get("username")
        or sender.get("username")
        or author.get("name")
        or author.get("username")
        or "unknown"
    )

    offset = extract_offset(record, started_at_epoch)
    if offset is None:
        return None

    return ChatMessage(
        offset_seconds=max(0.0, offset),
        username=username,
        text=text,
        emotes=tuple(extract_emotes(record)),
    )


def extract_offset(record: dict[str, Any], started_at_epoch: float | None) -> float | None:
    for key in ("offset_seconds", "time_in_seconds", "video_offset_seconds"):
        if record.get(key) is not None:
            return float(record[key])

    if record.get("timestamp") is not None:
        raw = float(record["timestamp"])
        if raw > 1e14:
            raw /= 1e6
        elif raw > 1e11:
            raw /= 1e3
        if started_at_epoch is None:
            return None
        return raw - started_at_epoch

    from .vod import parse_epoch

    for key in ("created_at", "sent_at", "time"):
        epoch = parse_epoch(record.get(key))
        if epoch is not None and started_at_epoch is not None:
            return epoch - started_at_epoch
    return None


def extract_emotes(record: dict[str, Any]) -> list[str]:
    emotes = record.get("emotes")
    if not isinstance(emotes, list):
        return []
    names: list[str] = []
    for entry in emotes:
        if isinstance(entry, dict):
            name = entry.get("name") or entry.get("id")
            if name:
                names.append(str(name))
        elif entry:
            names.append(str(entry))
    return names


def build_chat_source(
    kind: str, *, chat_file: Path | None = None, timeout: float = 30.0
) -> ChatSource:
    if kind == "none":
        return NullChatSource()
    if kind == "file":
        if chat_file is None:
            raise ValueError("chat source 'file' requires a chat file path")
        return JsonlChatSource(chat_file)
    if kind == "kick":
        return KickReplayChatSource(timeout=timeout)
    raise ValueError(f"unknown chat source: {kind}")
