"""Scene-change detection and heartbeat scheduling."""

from __future__ import annotations

import logging
import re
import subprocess
from collections.abc import Iterable
from pathlib import Path

from ..models import SamplePoint

log = logging.getLogger(__name__)

_PTS_RE = re.compile(r"pts_time:\s*(-?\d+(?:\.\d+)?)")
_SCORE_RE = re.compile(r"lavfi\.scene_score=(\d+(?:\.\d+)?)")


def build_detect_argv(
    source: str,
    *,
    threshold: float,
    ffmpeg: str = "ffmpeg",
    detect_width: int = 160,
    keyframes_only: bool = True,
    start_seconds: float = 0.0,
) -> list[str]:
    """Assemble the scene-detection ffmpeg command.

    Decoding is restricted to keyframes by default. HLS renditions carry a
    keyframe every 2-4 seconds, so this cuts decode work by roughly two orders
    of magnitude versus a full-frame pass while keeping every real cut, which
    always lands on or next to a keyframe.
    """
    argv = [ffmpeg, "-hide_banner", "-nostats", "-loglevel", "info"]
    if keyframes_only:
        argv += ["-skip_frame", "nokey"]
    if start_seconds > 0:
        argv += ["-ss", f"{start_seconds:.3f}"]
    argv += [
        "-i", source,
        "-an", "-sn", "-dn",
        "-vf",
        f"scale={detect_width}:-2:flags=fast_bilinear,"
        f"select='gt(scene,{threshold})',metadata=print:file=-",
        "-f", "null",
        "-",
    ]
    return argv


def parse_scene_metadata(lines: Iterable[str], *, offset: float = 0.0) -> list[SamplePoint]:
    """Parse `metadata=print` output into scene sample points.

    ffmpeg emits a `pts_time:` header line followed by a `lavfi.scene_score=`
    line. Either may be interleaved with unrelated log output.
    """
    points: list[SamplePoint] = []
    pending_ts: float | None = None
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        pts_match = _PTS_RE.search(line)
        if pts_match:
            pending_ts = float(pts_match.group(1))
        score_match = _SCORE_RE.search(line)
        if score_match and pending_ts is not None:
            ts = max(0.0, pending_ts + offset)
            points.append(
                SamplePoint(
                    offset_seconds=ts,
                    trigger="scene",
                    scene_score=float(score_match.group(1)),
                )
            )
            pending_ts = None
    return points


def detect_scenes(
    source: str,
    *,
    threshold: float = 0.35,
    ffmpeg: str = "ffmpeg",
    detect_width: int = 160,
    keyframes_only: bool = True,
    start_seconds: float = 0.0,
    timeout: float | None = None,
    log_path: Path | None = None,
) -> list[SamplePoint]:
    """Run the detection pass and return scene-triggered sample points."""
    argv = build_detect_argv(
        source,
        threshold=threshold,
        ffmpeg=ffmpeg,
        detect_width=detect_width,
        keyframes_only=keyframes_only,
        start_seconds=start_seconds,
    )
    log.info("scene detection pass started (threshold=%.2f)", threshold)
    proc = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(combined, encoding="utf-8")
    if proc.returncode != 0 and not combined.strip():
        raise RuntimeError(f"ffmpeg scene detection failed with code {proc.returncode}")
    points = parse_scene_metadata(combined.splitlines(), offset=start_seconds)
    log.info("scene detection found %d candidates", len(points))
    return points


def heartbeat_points(
    duration_seconds: float,
    scene_points: Iterable[SamplePoint],
    *,
    interval_seconds: float = 900.0,
    lead_in_seconds: float = 30.0,
) -> list[SamplePoint]:
    """Emit heartbeats wherever the scene detector has gone quiet too long.

    Walks forward from the last accepted trigger. A static single-game session
    that never trips the scene filter still gets one checkpoint per interval.
    """
    if interval_seconds <= 0 or duration_seconds <= 0:
        return []
    scene_times = sorted(p.offset_seconds for p in scene_points)
    beats: list[SamplePoint] = []
    cursor = min(lead_in_seconds, max(0.0, duration_seconds - 1.0))
    scene_index = 0
    last_trigger = 0.0
    if not scene_times or scene_times[0] > cursor:
        beats.append(SamplePoint(offset_seconds=cursor, trigger="heartbeat"))
        last_trigger = cursor

    while last_trigger + interval_seconds < duration_seconds:
        next_beat = last_trigger + interval_seconds
        advanced = False
        while scene_index < len(scene_times) and scene_times[scene_index] <= next_beat:
            last_trigger = max(last_trigger, scene_times[scene_index])
            scene_index += 1
            advanced = True
        if advanced and last_trigger + interval_seconds > next_beat:
            continue
        beats.append(SamplePoint(offset_seconds=next_beat, trigger="heartbeat"))
        last_trigger = next_beat
    return beats


def _thin_uniformly(points: list[SamplePoint], limit: int) -> list[SamplePoint]:
    """Drop points while preserving temporal spread."""
    if limit <= 0 or len(points) <= limit:
        return points
    step = len(points) / limit
    return [points[min(len(points) - 1, int(i * step))] for i in range(limit)]


def merge_sample_points(
    *groups: Iterable[SamplePoint],
    min_gap_seconds: float = 45.0,
    duration_seconds: float | None = None,
    edge_margin: float = 8.0,
    max_samples: int = 0,
) -> list[SamplePoint]:
    """Combine trigger sources into a deduplicated, ordered sample plan.

    Points closer together than `min_gap_seconds` collapse to the first one,
    keeping the strongest scene score. Points too close to either VOD edge are
    dropped because their burst window would fall outside the media.
    """
    candidates: list[SamplePoint] = []
    for group in groups:
        candidates.extend(group)
    if not candidates:
        return []

    priority = {"boundary": 0, "scene": 1, "heartbeat": 2}
    candidates.sort(key=lambda p: (p.offset_seconds, priority.get(p.trigger, 3)))

    kept: list[SamplePoint] = []
    for point in candidates:
        ts = point.offset_seconds
        if ts < edge_margin:
            continue
        if duration_seconds is not None and ts > duration_seconds - edge_margin:
            continue
        if kept and ts - kept[-1].offset_seconds < min_gap_seconds:
            previous = kept[-1]
            if (point.scene_score or 0.0) > (previous.scene_score or 0.0):
                kept[-1] = previous.model_copy(update={"scene_score": point.scene_score})
            continue
        kept.append(point)

    if max_samples and len(kept) > max_samples:
        kept = _thin_uniformly(kept, max_samples)
    return kept


def plan_samples(
    duration_seconds: float,
    scene_points: Iterable[SamplePoint],
    *,
    heartbeat_interval: float = 900.0,
    min_gap_seconds: float = 45.0,
    max_samples: int = 0,
) -> list[SamplePoint]:
    """Full sampling plan: scene triggers plus heartbeat fallbacks."""
    scenes = list(scene_points)
    beats = heartbeat_points(duration_seconds, scenes, interval_seconds=heartbeat_interval)
    return merge_sample_points(
        scenes,
        beats,
        min_gap_seconds=min_gap_seconds,
        duration_seconds=duration_seconds,
        max_samples=max_samples,
    )
