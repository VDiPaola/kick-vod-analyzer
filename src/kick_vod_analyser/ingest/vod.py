"""Kick VOD resolution: URL to metadata plus a playable stream reference."""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from datetime import datetime, timezone
from typing import Any

from ..models import VodInfo
from .http import build_client

log = logging.getLogger(__name__)

_UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE
)
_SLUG_RE = re.compile(r"kick\.com/(?!video/)([A-Za-z0-9_-]+)", re.IGNORECASE)

VIDEO_API = "https://kick.com/api/v1/video/{uuid}"


class VodResolutionError(RuntimeError):
    """Raised when a VOD cannot be resolved to metadata or a stream URL."""


def parse_vod_url(url: str) -> tuple[str | None, str | None]:
    """Extract (video_uuid, channel_slug) from any Kick VOD URL shape."""
    uuid_match = _UUID_RE.search(url)
    slug_match = _SLUG_RE.search(url)
    slug = slug_match.group(1) if slug_match else None
    if slug and slug.lower() in {"video", "videos"}:
        slug = None
    return (uuid_match.group(0) if uuid_match else None), slug


def dig(payload: dict[str, Any], *paths: tuple[str, ...]) -> Any:
    """Return the first non-empty value found at any of the given key paths."""
    for path in paths:
        cursor: Any = payload
        for key in path:
            if not isinstance(cursor, dict) or key not in cursor:
                cursor = None
                break
            cursor = cursor[key]
        if cursor not in (None, "", 0):
            return cursor
    return None


def parse_epoch(value: Any) -> float | None:
    """Parse the several timestamp shapes Kick returns into a unix epoch."""
    if not value:
        return None
    text = str(value).replace("Z", "+00:00")
    candidates = (
        lambda: datetime.fromisoformat(text),
        lambda: datetime.strptime(text, "%Y-%m-%d %H:%M:%S"),
        lambda: datetime.strptime(text, "%Y-%m-%dT%H:%M:%S%z"),
    )
    for build in candidates:
        try:
            parsed = build()
        except ValueError:
            continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    return None


def normalise_duration(raw: Any) -> float:
    """Kick reports duration in milliseconds on some payloads, seconds on others."""
    duration = float(raw or 0.0)
    return duration / 1000.0 if duration > 100_000 else duration


def probe_duration(source: str, *, ffprobe: str = "ffprobe") -> float:
    """Read a duration straight from the stream.

    Kick reports duration 0 while a stream is still live, so the playlist itself
    is the only source of truth for an in-progress VOD.
    """
    from ..ffmpeg import probe

    try:
        return probe(source, ffprobe=ffprobe).duration_seconds
    except Exception as exc:
        log.warning("ffprobe could not read a duration from the stream: %s", exc)
        return 0.0


def resolve_via_api(url: str, *, timeout: float = 30.0) -> VodInfo:
    """Resolve using the internal Kick video endpoint."""
    uuid, slug = parse_vod_url(url)
    if not uuid:
        raise VodResolutionError(f"no video UUID found in URL: {url}")

    client = build_client(timeout)
    try:
        response = client.get(VIDEO_API.format(uuid=uuid))
        if response.status_code != 200:
            raise VodResolutionError(f"Kick video API returned {response.status_code} for {uuid}")
        payload = response.json()
    finally:
        client.close()

    if not isinstance(payload, dict):
        raise VodResolutionError("unexpected payload from Kick video API")

    playback_url = dig(payload, ("source",), ("playback_url",))
    duration = normalise_duration(dig(payload, ("livestream", "duration"), ("duration",)))
    if duration <= 0 and playback_url:
        duration = probe_duration(str(playback_url))
    if duration <= 0:
        raise VodResolutionError("Kick video API did not report a usable duration")

    channel_slug = (
        slug or dig(payload, ("livestream", "channel", "slug"), ("channel", "slug")) or "unknown"
    )
    channel_id = dig(payload, ("livestream", "channel", "id"), ("channel", "id"))

    return VodInfo(
        vod_id=uuid,
        url=url,
        channel_slug=str(channel_slug),
        channel_id=int(channel_id) if channel_id else None,
        title=str(dig(payload, ("livestream", "session_title"), ("session_title",)) or ""),
        duration_seconds=duration,
        started_at_epoch=parse_epoch(
            dig(payload, ("livestream", "start_time"), ("start_time",), ("created_at",))
        ),
        playback_url=playback_url,
    )


def resolve_via_ytdlp(url: str) -> VodInfo:
    """Resolve using yt-dlp, which tracks Kick extractor changes upstream."""
    binary = shutil.which("yt-dlp")
    argv = (
        [binary, "-J", "--no-warnings", url]
        if binary
        else ["python", "-m", "yt_dlp", "-J", "--no-warnings", url]
    )
    proc = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise VodResolutionError(f"yt-dlp failed: {(proc.stderr or '').strip()[-400:]}")

    payload = json.loads(proc.stdout or "{}")
    duration = float(payload.get("duration") or 0.0)
    if duration <= 0:
        raise VodResolutionError("yt-dlp did not report a duration")

    uuid, slug = parse_vod_url(url)
    stream_url = payload.get("url")
    if not stream_url:
        formats = [f for f in payload.get("formats", []) if f.get("url")]
        if formats:
            stream_url = formats[-1]["url"]

    return VodInfo(
        vod_id=str(payload.get("id") or uuid or "vod"),
        url=url,
        channel_slug=str(payload.get("uploader_id") or payload.get("channel") or slug or "unknown"),
        channel_id=None,
        title=str(payload.get("title") or ""),
        duration_seconds=duration,
        started_at_epoch=float(payload["timestamp"]) if payload.get("timestamp") else None,
        playback_url=stream_url,
    )


def resolve_vod(url: str, *, timeout: float = 30.0, prefer: str = "api") -> VodInfo:
    """Resolve a Kick VOD, falling back to the other strategy on failure."""
    order = ["api", "ytdlp"] if prefer == "api" else ["ytdlp", "api"]
    errors: list[str] = []
    for strategy in order:
        try:
            info = (
                resolve_via_api(url, timeout=timeout)
                if strategy == "api"
                else resolve_via_ytdlp(url)
            )
        except Exception as exc:
            log.warning("%s resolution failed: %s", strategy, exc)
            errors.append(f"{strategy}: {exc}")
            continue
        if not info.playback_url:
            errors.append(f"{strategy}: no playback URL")
            continue
        log.info("resolved VOD %s (%.0fs) via %s", info.vod_id, info.duration_seconds, strategy)
        return info
    raise VodResolutionError("could not resolve VOD; " + "; ".join(errors))
