"""Burst frame extraction: sample point -> composited 2x2 grid."""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ..config import SamplingSettings
from ..ffmpeg import extract_frames
from ..models import SamplePoint
from .grid import compose_grid, frame_signature, is_near_duplicate

log = logging.getLogger(__name__)


@dataclass
class GridArtifact:
    """A composited grid ready for classification."""

    point: SamplePoint
    path: Path
    phash: int
    frame_count: int


def burst_timestamps(
    offset_seconds: float,
    burst_offsets: Sequence[float],
    *,
    duration_seconds: float | None = None,
) -> list[float]:
    """Clamp the burst window into the media bounds, preserving order."""
    upper = None if duration_seconds is None else max(0.0, duration_seconds - 0.5)
    stamps: list[float] = []
    for delta in burst_offsets:
        ts = offset_seconds + delta
        ts = max(0.0, ts)
        if upper is not None:
            ts = min(upper, ts)
        stamps.append(round(ts, 3))
    return stamps


def build_grids(
    source: str,
    points: Sequence[SamplePoint],
    work_dir: Path,
    *,
    settings: SamplingSettings,
    duration_seconds: float | None = None,
    ffmpeg: str = "ffmpeg",
    keep_frames: bool = False,
    dedupe: bool = True,
) -> list[GridArtifact]:
    """Extract burst frames for every point and composite them into grids.

    Grids within `phash_distance` of the previously kept grid are dropped: an
    identical visual state does not need a second classification, and the
    timeline reconstruction interpolates across the gap.
    """
    grids_dir = work_dir / "grids"
    frames_root = work_dir / "frames"
    grids_dir.mkdir(parents=True, exist_ok=True)

    artifacts: list[GridArtifact] = []
    previous_hash: int | None = None

    for point in points:
        stamps = burst_timestamps(
            point.offset_seconds, settings.burst_offsets, duration_seconds=duration_seconds
        )
        frame_dir = frames_root / point.custom_id
        frames = extract_frames(
            source,
            stamps,
            frame_dir,
            ffmpeg=ffmpeg,
            width=settings.grid_width // 2,
            timeout=180.0,
        )
        if not frames:
            log.warning("no frames extracted at %.1fs, skipping", point.offset_seconds)
            if not keep_frames:
                shutil.rmtree(frame_dir, ignore_errors=True)
            continue

        grid_path = grids_dir / f"{point.custom_id}.jpg"
        compose_grid(
            frames,
            grid_path,
            width=settings.grid_width,
            height=settings.grid_height,
            quality=settings.jpeg_quality,
        )
        if not keep_frames:
            shutil.rmtree(frame_dir, ignore_errors=True)

        digest = frame_signature(grid_path)
        if (
            dedupe
            and point.trigger == "scene"
            and previous_hash is not None
            and is_near_duplicate(digest, previous_hash, max_distance=settings.phash_distance)
        ):
            log.debug("dropping near-duplicate grid at %.1fs", point.offset_seconds)
            grid_path.unlink(missing_ok=True)
            continue

        previous_hash = digest
        artifacts.append(
            GridArtifact(point=point, path=grid_path, phash=digest, frame_count=len(frames))
        )

    log.info("built %d grids from %d sample points", len(artifacts), len(points))
    return artifacts
