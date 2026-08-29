"""Kick VOD resolution: URL to metadata plus a playable stream reference."""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from typing import Any

from ..models import VodInfo
from .http import build_client

log = logging.getLogger(__name__)

_UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE
)
_SLUG_RE = re.compile(r"kick\.com/(?!video/)([A-Za-z0-9_-]+)", re.IGNORECASE)
_HOST_RE = re.compile(r"^(?:https?://)?(?:([^/@]+)@)?([^/:?#]+)", re.IGNORECASE)

VIDEO_API = "https://kick.com/api/v1/video/{uuid}"
CHANNEL_VIDEOS_API = "https://kick.com/api/v2/channels/{slug}/videos"

KICK_HOSTS = frozenset({"kick.com", "www.kick.com"})

# A URL id and a video id may be minted a few seconds apart, so the match is
# fuzzy. Broadcasts on one channel are hours apart, well outside this window.
UUID7_MATCH_TOLERANCE_SECONDS = 120.0


class VodResolutionError(RuntimeError):
    """Raised when a VOD cannot be resolved to metadata or a stream URL."""


def url_host(url: str) -> str:
    match = _HOST_RE.match(url.strip())
    return match.group(2).lower() if match else ""


def is_kick_url(url: str) -> bool:
    return url_host(url) in KICK_HOSTS


def uuid_version(value: str) -> int | None:
    try:
        return uuid.UUID(value).version
    except (ValueError, AttributeError, TypeError):
        return None


def uuid7_epoch(value: str) -> float | None:
    """Decode the creation time a version 7 UUID carries in its leading 48 bits."""
    if uuid_version(value) != 7:
        return None
    return (uuid.UUID(value).int >> 80) / 1000.0


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


def find_video_uuid_by_time(
    slug: str,
    target_epoch: float,
    *,
    timeout: float = 30.0,
    tolerance: float = UUID7_MATCH_TOLERANCE_SECONDS,
    client: Any = None,
) -> str | None:
    """Find a channel's video whose start time matches an epoch.

    Kick's VOD URLs carry a version 7 id that no read endpoint accepts, and that
    id appears nowhere in the API payloads. The channel listing does expose the
    version 4 `video.uuid` alongside a creation timestamp, so the two id spaces
    are bridged by matching on time.
    """
    owned = client is None
    session = client or build_client(timeout)
    try:
        response = session.get(CHANNEL_VIDEOS_API.format(slug=slug))
        if response.status_code != 200:
            log.warning("channel videos listing returned %s for %s", response.status_code, slug)
            return None
        rows = response.json()
    except Exception as exc:
        log.warning("could not read the channel videos listing for %s: %s", slug, exc)
        return None
    finally:
        if owned:
            session.close()

    if not isinstance(rows, list):
        return None

    best_uuid: str | None = None
    best_delta = float("inf")
    for row in rows:
        if not isinstance(row, dict):
            continue
        created = parse_epoch(row.get("created_at")) or parse_epoch(row.get("start_time"))
        candidate = dig(row, ("video", "uuid"), ("uuid",))
        if created is None or not candidate:
            continue
        delta = abs(created - target_epoch)
        if delta < best_delta:
            best_uuid, best_delta = str(candidate), delta

    if best_uuid is None or best_delta > tolerance:
        log.warning(
            "no video in the %s listing matches the URL timestamp (closest %.0fs away)",
            slug,
            best_delta if best_uuid else -1,
        )
        return None

    log.info("mapped URL id to video uuid %s (%.1fs apart)", best_uuid, best_delta)
    return best_uuid


def resolve_video_uuid(url: str, *, timeout: float = 30.0, client: Any = None) -> str:
    """Turn a VOD URL into the video uuid the read endpoints accept.

    A version 4 id is already canonical and passes straight through. A version 7
    id is the newer URL-only identifier and has to be mapped through the
    channel listing, which needs the slug that the URL also carries.
    """
    if not is_kick_url(url):
        raise VodResolutionError(
            f"not a Kick URL: {url}. Expected a link on kick.com, "
            f"for example https://kick.com/<channel>/videos/<id>"
        )

    url_id, slug = parse_vod_url(url)
    if not url_id:
        raise VodResolutionError(f"no video id found in URL: {url}")

    if uuid_version(url_id) != 7:
        return url_id

    if not slug:
        raise VodResolutionError(
            f"{url_id} is a URL-only video id, which can only be resolved through its "
            f"channel. Use the https://kick.com/<channel>/videos/<id> form of the link."
        )

    target = uuid7_epoch(url_id)
    if target is None:
        raise VodResolutionError(f"could not decode a timestamp from {url_id}")

    mapped = find_video_uuid_by_time(slug, target, timeout=timeout, client=client)
    if not mapped:
        raise VodResolutionError(
            f"could not map URL id {url_id} to a video on channel {slug}. "
            f"The channel listing only covers recent VODs, so an older one may "
            f"no longer appear there."
        )
    return mapped


def resolve_via_api(url: str, *, timeout: float = 30.0) -> VodInfo:
    """Resolve using the internal Kick video endpoint."""
    _, slug = parse_vod_url(url)

    client = build_client(timeout)
    try:
        video_uuid = resolve_video_uuid(url, timeout=timeout, client=client)
        response = client.get(VIDEO_API.format(uuid=video_uuid))
        if response.status_code != 200:
            raise VodResolutionError(
                f"Kick video API returned {response.status_code} for {video_uuid}"
            )
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
        vod_id=video_uuid,
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


def canonical_vod_url(url: str, *, timeout: float = 30.0) -> str:
    """Rewrite a VOD URL so it carries the id the read endpoints accept.

    yt-dlp queries the same endpoint this package does, so it fails on a
    URL-only id for the same reason. Handing it the canonical id lets the
    fallback strategy work instead of duplicating the failure.
    """
    url_id, slug = parse_vod_url(url)
    if not url_id or uuid_version(url_id) != 7:
        return url
    video_uuid = resolve_video_uuid(url, timeout=timeout)
    return f"https://kick.com/{slug}/videos/{video_uuid}" if slug else url


def resolve_vod(url: str, *, timeout: float = 30.0, prefer: str = "api") -> VodInfo:
    """Resolve a Kick VOD, falling back to the other strategy on failure."""
    if not is_kick_url(url):
        raise VodResolutionError(
            f"not a Kick URL: {url}. Expected a link on kick.com, "
            f"for example https://kick.com/<channel>/videos/<id>"
        )

    order = ["api", "ytdlp"] if prefer == "api" else ["ytdlp", "api"]
    errors: list[str] = []
    for strategy in order:
        try:
            info = (
                resolve_via_api(url, timeout=timeout)
                if strategy == "api"
                else resolve_via_ytdlp(canonical_vod_url(url, timeout=timeout))
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
