"""Timeline serialisation: JSON, WebVTT chapters, CSV, and a summary report."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from ..models import PrimaryCategory, Segment, Timeline


def format_timestamp(seconds: float, *, millis: bool = False) -> str:
    """HH:MM:SS, optionally with the milliseconds WebVTT requires."""
    total = max(0.0, seconds)
    hours = int(total // 3600)
    minutes = int((total % 3600) // 60)
    secs = total % 60
    if millis:
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
    return f"{hours:02d}:{minutes:02d}:{int(secs):02d}"


def format_duration(seconds: float) -> str:
    total = int(max(0.0, seconds))
    hours, minutes = total // 3600, (total % 3600) // 60
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {total % 60}s"
    return f"{total}s"


def write_timeline_json(timeline: Timeline, path: Path) -> Path:
    """Machine-readable event log with full per-sample provenance."""
    payload = {
        "vod": timeline.vod.model_dump(),
        "provider": timeline.provider,
        "model": timeline.model,
        "segment_count": len(timeline.segments),
        "sample_count": len(timeline.samples),
        "segments": [
            segment.model_dump() | {
                "start_timestamp": format_timestamp(segment.start_seconds),
                "end_timestamp": format_timestamp(segment.end_seconds),
                "duration_seconds": round(segment.duration_seconds, 3),
                "label": segment.label,
            }
            for segment in timeline.segments
        ],
        "samples": [
            sample.model_dump() | {"timestamp": format_timestamp(sample.offset_seconds)}
            for sample in timeline.samples
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_chapters_vtt(timeline: Timeline, path: Path) -> Path:
    """WebVTT chapter track importable by VLC and web players."""
    lines = ["WEBVTT", ""]
    for index, segment in enumerate(timeline.segments, start=1):
        lines.append(f"chapter-{index}")
        lines.append(
            f"{format_timestamp(segment.start_seconds, millis=True)} --> "
            f"{format_timestamp(segment.end_seconds, millis=True)}"
        )
        lines.append(segment.label)
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_segments_csv(timeline: Timeline, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "start_seconds",
                "end_seconds",
                "duration_seconds",
                "start_timestamp",
                "primary_category",
                "specific_title_or_context",
                "sub_activity",
                "is_afk_or_brb",
                "confidence_score",
                "sample_count",
            ]
        )
        for segment in timeline.segments:
            writer.writerow(
                [
                    round(segment.start_seconds, 3),
                    round(segment.end_seconds, 3),
                    round(segment.duration_seconds, 3),
                    format_timestamp(segment.start_seconds),
                    segment.primary_category.value,
                    segment.specific_title_or_context,
                    segment.sub_activity,
                    int(segment.is_afk_or_brb),
                    round(segment.confidence_score, 4),
                    segment.sample_count,
                ]
            )
    return path


def category_breakdown(segments: list[Segment]) -> list[tuple[str, float]]:
    totals: dict[str, float] = defaultdict(float)
    for segment in segments:
        totals[segment.primary_category.value] += segment.duration_seconds
    return sorted(totals.items(), key=lambda item: -item[1])


def title_breakdown(segments: list[Segment]) -> list[tuple[str, float]]:
    totals: dict[str, float] = defaultdict(float)
    for segment in segments:
        title = segment.specific_title_or_context.strip()
        if title:
            totals[title] += segment.duration_seconds
    return sorted(totals.items(), key=lambda item: -item[1])


def afk_seconds(segments: list[Segment]) -> float:
    return sum(
        s.duration_seconds
        for s in segments
        if s.is_afk_or_brb or s.primary_category is PrimaryCategory.INTERMISSION
    )


def write_summary_report(timeline: Timeline, path: Path) -> Path:
    """Human-readable stream overview with duration shares."""
    vod = timeline.vod
    segments = timeline.segments
    total = vod.duration_seconds or 1.0

    lines = [
        f"# Stream Activity Report: {vod.title or vod.vod_id}",
        "",
        f"- Channel: {vod.channel_slug}",
        f"- VOD: {vod.url}",
        f"- Duration: {format_duration(vod.duration_seconds)}",
        f"- Segments: {len(segments)}",
        f"- Classified samples: {len(timeline.samples)}",
        f"- Model: {timeline.provider}/{timeline.model}",
        "",
        "## Category breakdown",
        "",
        "| Category | Duration | Share |",
        "| --- | ---: | ---: |",
    ]
    for name, seconds in category_breakdown(segments):
        lines.append(f"| {name} | {format_duration(seconds)} | {seconds / total * 100:.1f}% |")

    titles = title_breakdown(segments)
    if titles:
        lines += [
            "",
            "## Games and media",
            "",
            "| Title | Duration | Share |",
            "| --- | ---: | ---: |",
        ]
        for name, seconds in titles:
            lines.append(f"| {name} | {format_duration(seconds)} | {seconds / total * 100:.1f}% |")

    away = afk_seconds(segments)
    lines += [
        "",
        "## Away time",
        "",
        f"- AFK, BRB, or intermission: {format_duration(away)} ({away / total * 100:.1f}%)",
        f"- Active content: {format_duration(total - away)} "
        f"({(total - away) / total * 100:.1f}%)",
        "",
        "## Timeline",
        "",
        "| Start | End | Duration | Activity | Sub-activity | Confidence |",
        "| --- | --- | ---: | --- | --- | ---: |",
    ]
    for segment in segments:
        lines.append(
            f"| {format_timestamp(segment.start_seconds)} "
            f"| {format_timestamp(segment.end_seconds)} "
            f"| {format_duration(segment.duration_seconds)} "
            f"| {segment.label} "
            f"| {segment.sub_activity or '-'} "
            f"| {segment.confidence_score:.2f} |"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_all(timeline: Timeline, out_dir: Path) -> dict[str, Path]:
    """Write every deliverable and return the paths by artefact name."""
    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        "timeline_json": write_timeline_json(timeline, out_dir / "timeline.json"),
        "chapters_vtt": write_chapters_vtt(timeline, out_dir / "chapters.vtt"),
        "segments_csv": write_segments_csv(timeline, out_dir / "segments.csv"),
        "summary_md": write_summary_report(timeline, out_dir / "summary_report.md"),
    }
