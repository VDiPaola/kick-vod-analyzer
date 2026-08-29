"""HLS master playlist parsing and rendition selection.

Scene detection reads the entire VOD end to end, so it must run on the smallest
rendition available; frame extraction only fetches the segments around each
sample point and can afford a legible one. Sending both stages at the master
playlist means ffmpeg picks the first variant, which is usually the largest, and
a 12 hour source stream then dominates the runtime.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from urllib.parse import urljoin

log = logging.getLogger(__name__)

STREAM_INF_RE = re.compile(r"^#EXT-X-STREAM-INF:(?P<attrs>.*)$", re.IGNORECASE)
ATTR_RE = re.compile(r'([A-Z0-9-]+)=("[^"]*"|[^,]*)')


@dataclass(frozen=True)
class Rendition:
    """One variant stream from a master playlist."""

    url: str
    bandwidth: int = 0
    width: int = 0
    height: int = 0
    name: str = ""

    @property
    def label(self) -> str:
        if self.height:
            return f"{self.height}p"
        return self.name or f"{self.bandwidth // 1000}kbps"


def parse_master_playlist(text: str, base_url: str = "") -> list[Rendition]:
    """Extract variant streams from a master playlist, ordered as written."""
    renditions: list[Rendition] = []
    lines = [line.strip() for line in text.splitlines()]

    for index, line in enumerate(lines):
        match = STREAM_INF_RE.match(line)
        if not match:
            continue
        target = next((candidate for candidate in lines[index + 1 :] if candidate), "")
        if not target or target.startswith("#"):
            continue

        attributes = {
            key.upper(): value.strip('"') for key, value in ATTR_RE.findall(match.group("attrs"))
        }
        width = height = 0
        resolution = attributes.get("RESOLUTION", "")
        if "x" in resolution:
            raw_width, _, raw_height = resolution.partition("x")
            width, height = _to_int(raw_width), _to_int(raw_height)

        renditions.append(
            Rendition(
                url=urljoin(base_url, target) if base_url else target,
                bandwidth=_to_int(attributes.get("BANDWIDTH")),
                width=width,
                height=height,
                name=attributes.get("NAME", ""),
            )
        )
    return renditions


def _to_int(value: object) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return 0


def _sort_key(rendition: Rendition) -> tuple[int, int]:
    return (rendition.height or 0, rendition.bandwidth)


def smallest(renditions: list[Rendition]) -> Rendition | None:
    """The cheapest variant to decode end to end."""
    return min(renditions, key=_sort_key) if renditions else None


def closest_to_height(renditions: list[Rendition], target_height: int) -> Rendition | None:
    """The variant nearest a target height, preferring one at or above it."""
    if not renditions:
        return None
    at_or_above = [r for r in renditions if r.height >= target_height]
    pool = at_or_above or renditions
    return min(pool, key=lambda r: (abs(r.height - target_height), r.bandwidth))


def fetch_master_playlist(url: str, *, timeout: float = 30.0) -> list[Rendition]:
    """Download and parse a master playlist, returning an empty list on failure."""
    if not url.lower().split("?")[0].endswith(".m3u8"):
        return []
    try:
        import httpx

        response = httpx.get(url, timeout=timeout, follow_redirects=True)
        if response.status_code != 200:
            log.warning("master playlist returned %s", response.status_code)
            return []
        return parse_master_playlist(response.text, base_url=str(response.url))
    except Exception as exc:
        log.warning("could not read the master playlist: %s", exc)
        return []


@dataclass(frozen=True)
class StreamPlan:
    """Which rendition each stage should read."""

    detect_url: str
    extract_url: str
    detect_rendition: Rendition | None = None
    extract_rendition: Rendition | None = None

    @property
    def is_split(self) -> bool:
        return self.detect_url != self.extract_url


def plan_streams(
    playback_url: str, *, extract_height: int = 720, timeout: float = 30.0
) -> StreamPlan:
    """Choose the detection and extraction renditions for a playback URL.

    Falls back to the original URL for both stages whenever the playlist cannot
    be read, so a parsing failure costs speed rather than correctness.
    """
    renditions = fetch_master_playlist(playback_url, timeout=timeout)
    if len(renditions) < 2:
        return StreamPlan(detect_url=playback_url, extract_url=playback_url)

    detect = smallest(renditions)
    extract = closest_to_height(renditions, extract_height)
    log.info(
        "stream plan: detect on %s, extract on %s (%d renditions available)",
        detect.label if detect else "source",
        extract.label if extract else "source",
        len(renditions),
    )
    return StreamPlan(
        detect_url=detect.url if detect else playback_url,
        extract_url=extract.url if extract else playback_url,
        detect_rendition=detect,
        extract_rendition=extract,
    )
