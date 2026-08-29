"""Thin subprocess wrappers around ffmpeg and ffprobe."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


class FfmpegError(RuntimeError):
    """Raised when an ffmpeg or ffprobe invocation fails."""


@dataclass(frozen=True)
class ProbeResult:
    duration_seconds: float
    width: int
    height: int
    codec: str


def ensure_available(*binaries: str) -> None:
    missing = [b for b in binaries if shutil.which(b) is None]
    if missing:
        raise FfmpegError(f"required binaries not found on PATH: {', '.join(missing)}")


def run(argv: list[str], *, timeout: float | None = None) -> subprocess.CompletedProcess[str]:
    log.debug("exec: %s", " ".join(argv))
    proc = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-12:]
        raise FfmpegError(f"{argv[0]} exited {proc.returncode}:\n" + "\n".join(tail))
    return proc


def probe(source: str, *, ffprobe: str = "ffprobe", timeout: float = 120.0) -> ProbeResult:
    argv = [
        ffprobe,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,codec_name",
        "-show_entries", "format=duration",
        "-of", "json",
        source,
    ]
    payload = json.loads(run(argv, timeout=timeout).stdout or "{}")
    streams = payload.get("streams") or [{}]
    fmt = payload.get("format") or {}
    duration = float(fmt.get("duration") or 0.0)
    if duration <= 0:
        raise FfmpegError(f"could not determine duration for {source}")
    stream = streams[0]
    return ProbeResult(
        duration_seconds=duration,
        width=int(stream.get("width") or 0),
        height=int(stream.get("height") or 0),
        codec=str(stream.get("codec_name") or ""),
    )


def extract_frames(
    source: str,
    timestamps: list[float],
    dest_dir: Path,
    *,
    ffmpeg: str = "ffmpeg",
    width: int = 640,
    prefix: str = "frame",
    timeout: float = 180.0,
) -> list[Path]:
    """Extract one JPEG per timestamp using fast input seeking.

    Each timestamp becomes its own invocation. Input seeking on HLS only fetches
    the segments covering the requested position, so per-frame cost stays flat
    regardless of VOD length.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for index, ts in enumerate(timestamps):
        target = dest_dir / f"{prefix}_{index}.jpg"
        argv = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{max(0.0, ts):.3f}",
            "-i", source,
            "-frames:v", "1",
            "-an", "-sn",
            "-vf", f"scale={width}:-2:flags=fast_bilinear",
            "-q:v", "3",
            str(target),
        ]
        try:
            run(argv, timeout=timeout)
        except (FfmpegError, subprocess.TimeoutExpired) as exc:
            log.warning("frame extraction failed at %.1fs: %s", ts, exc)
            continue
        if target.exists() and target.stat().st_size > 0:
            written.append(target)
    return written
