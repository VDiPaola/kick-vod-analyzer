"""Chat replay ingestion.

Kick exposes no documented VOD chat history API. The replay the web player
renders comes from an undocumented, Cloudflare-fronted endpoint that can change
without notice, so chat is treated as optional enrichment: every source
degrades to an empty index rather than failing the run.
"""

from __future__ import annotations

import bisect
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..classify.retry import call_with_retry
from ..models import ChatMessage, VodInfo
from .http import build_client
from .vod import parse_epoch

log = logging.getLogger(__name__)

HISTORY_API = "https://web.kick.com/api/v1/chat/{chat_id}/history"
INLINE_EMOTE = re.compile(r"\[emote:(\d+):([^\]]+)\]")


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


class KickChatError(Exception):
    """A non-200 reply from the chat history endpoint."""

    def __init__(self, status_code: int, body: str = "") -> None:
        super().__init__(f"Kick chat history returned {status_code}: {body[:120]}")
        self.status_code = status_code


class KickReplayChatSource(ChatSource):
    """Download VOD chat from Kick's chat history endpoint.

    `web.kick.com/api/v1/chat/{chat_id}/history?cursor=<epoch_micros>` returns
    the 25 messages sent before the cursor, newest first, plus the cursor for
    the next older page. No login is needed. The VOD is split into time chunks
    and each chunk is walked backwards from its end on its own thread, so the
    request count scales with message volume and quiet stretches cost nothing.

    The endpoint's `start_time` form is not used: it returns a fixed five second
    bucket truncated to its earliest 25 messages, which drops chat in busy moments.
    """

    name = "kick"

    def __init__(
        self,
        *,
        timeout: float = 30.0,
        chunk_seconds: float = 600.0,
        workers: int = 8,
        max_pages_per_chunk: int = 4000,
        auth_token: str | None = None,
        retry_attempts: int = 5,
        retry_sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.timeout = timeout
        self.chunk_seconds = max(1.0, chunk_seconds)
        self.workers = max(1, workers)
        self.max_pages_per_chunk = max_pages_per_chunk
        self.auth_token = auth_token
        self.retry_attempts = retry_attempts
        self.retry_sleep = retry_sleep

    def fetch(self, vod: VodInfo) -> ChatIndex:
        records = self.download(vod)
        messages = [
            m
            for m in (normalise_record(r, vod.started_at_epoch) for r in records)
            if m is not None and m.offset_seconds <= vod.duration_seconds
        ]
        log.info("collected %d chat messages for %s", len(messages), vod.vod_id)
        return ChatIndex(messages)

    def download(self, vod: VodInfo) -> list[dict[str, Any]]:
        """Return the raw Kick records for the VOD window, oldest first, deduplicated."""
        if not vod.channel_id:
            log.warning("no channel id resolved; skipping Kick chat replay")
            return []
        if not vod.started_at_epoch:
            log.warning("no VOD start time resolved; skipping Kick chat replay")
            return []

        url = HISTORY_API.format(chat_id=vod.channel_id)
        chunks = plan_chunks(
            vod.started_at_epoch, vod.started_at_epoch + vod.duration_seconds, self.chunk_seconds
        )
        client = build_client(self.timeout)
        collected: dict[str, dict[str, Any]] = {}
        try:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                for chunk_records in pool.map(lambda c: self.walk_chunk(client, url, *c), chunks):
                    for record in chunk_records:
                        collected.setdefault(record_id(record), record)
        finally:
            client.close()

        return sorted(collected.values(), key=lambda r: parse_epoch(r.get("created_at")) or 0.0)

    def walk_chunk(
        self, client: Any, url: str, start_epoch: float, end_epoch: float
    ) -> list[dict[str, Any]]:
        """Page backwards from end_epoch until a message older than start_epoch appears."""
        cursor = str(int(end_epoch * 1_000_000))
        records: list[dict[str, Any]] = []
        try:
            for _ in range(self.max_pages_per_chunk):
                page, next_cursor = self.request_page(client, url, cursor)
                if not page:
                    break
                oldest = end_epoch
                for record in page:
                    sent_at = parse_epoch(record.get("created_at"))
                    if sent_at is None:
                        continue
                    oldest = min(oldest, sent_at)
                    if start_epoch <= sent_at < end_epoch:
                        records.append(record)
                if oldest < start_epoch or not next_cursor or next_cursor == cursor:
                    break
                cursor = next_cursor
            else:
                log.warning(
                    "chat chunk starting at %s hit the %d page limit; chat may be incomplete",
                    start_epoch,
                    self.max_pages_per_chunk,
                )
        except Exception as exc:
            log.warning(
                "chat chunk %s..%s aborted after %d messages: %s",
                start_epoch,
                end_epoch,
                len(records),
                exc,
            )
        return records

    def request_page(
        self, client: Any, url: str, cursor: str
    ) -> tuple[list[dict[str, Any]], str | None]:
        headers = {"x-app-platform": "web"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        def once() -> Any:
            response = client.get(url, params={"cursor": cursor}, headers=headers)
            if response.status_code != 200:
                raise KickChatError(response.status_code, response.text)
            return response.json()

        payload = call_with_retry(
            once,
            label="Kick chat history",
            attempts=self.retry_attempts,
            base_delay=1.0,
            max_delay=30.0,
            sleep=self.retry_sleep,
        )
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, dict):
            return [], None
        messages = [m for m in (data.get("messages") or []) if isinstance(m, dict)]
        next_cursor = data.get("cursor")
        return messages, str(next_cursor) if next_cursor else None


def plan_chunks(
    start_epoch: float, end_epoch: float, chunk_seconds: float
) -> list[tuple[float, float]]:
    """Split [start, end) into consecutive windows of at most chunk_seconds."""
    chunks: list[tuple[float, float]] = []
    cursor = start_epoch
    while cursor < end_epoch:
        chunk_end = min(cursor + chunk_seconds, end_epoch)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end
    return chunks


def record_id(record: dict[str, Any]) -> str:
    identifier = record.get("id") or record.get("uuid")
    if identifier:
        return str(identifier)
    return json.dumps(record, sort_keys=True, default=str)


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

    inline = [name for _, name in INLINE_EMOTE.findall(text)]
    if inline:
        text = INLINE_EMOTE.sub(lambda m: m.group(2), text).strip()

    return ChatMessage(
        offset_seconds=max(0.0, offset),
        username=username,
        text=text,
        emotes=tuple(extract_emotes(record) + inline),
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
    kind: str,
    *,
    chat_file: Path | None = None,
    timeout: float = 30.0,
    auth_token: str | None = None,
    workers: int = 8,
) -> ChatSource:
    if kind == "none":
        return NullChatSource()
    if kind == "file":
        if chat_file is None:
            raise ValueError("chat source 'file' requires a chat file path")
        return JsonlChatSource(chat_file)
    if kind == "kick":
        return KickReplayChatSource(timeout=timeout, auth_token=auth_token, workers=workers)
    raise ValueError(f"unknown chat source: {kind}")
