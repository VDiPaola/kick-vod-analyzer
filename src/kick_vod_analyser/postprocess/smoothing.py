"""Temporal smoothing: noisy per-sample verdicts into a stable timeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

from ..config import SmoothingSettings
from ..models import PrimaryCategory, SampleResult, Segment

log = logging.getLogger(__name__)

GENERIC_TITLES = frozenset({"", "n/a", "na", "none", "unknown", "unclear", "-"})


def normalise_title(title: str) -> str:
    cleaned = " ".join(title.strip().lower().split())
    return "" if cleaned in GENERIC_TITLES else cleaned


@dataclass
class Run:
    """A maximal stretch of compatible consecutive samples."""

    category: PrimaryCategory
    title: str
    samples: list[SampleResult] = field(default_factory=list)

    @property
    def start(self) -> float:
        return self.samples[0].offset_seconds

    @property
    def last_offset(self) -> float:
        return self.samples[-1].offset_seconds

    @property
    def confidence(self) -> float:
        if not self.samples:
            return 0.0
        return sum(s.classification.confidence_score for s in self.samples) / len(self.samples)

    @property
    def afk_ratio(self) -> float:
        if not self.samples:
            return 0.0
        flagged = sum(1 for s in self.samples if s.classification.is_afk_or_brb)
        return flagged / len(self.samples)

    def best_title(self) -> str:
        """Pick the most specific title the run saw, breaking ties on confidence."""
        candidates = [
            s
            for s in self.samples
            if normalise_title(s.classification.specific_title_or_context)
        ]
        if not candidates:
            return ""
        best = max(candidates, key=lambda s: s.classification.confidence_score)
        return best.classification.specific_title_or_context.strip()

    def dominant_sub_activity(self) -> str:
        counts: dict[str, int] = {}
        for sample in self.samples:
            value = sample.classification.sub_activity.strip()
            if value:
                counts[value] = counts.get(value, 0) + 1
        if not counts:
            return ""
        return max(counts.items(), key=lambda item: item[1])[0]


def compatible(
    left_category: PrimaryCategory, left_title: str, right_category: PrimaryCategory, right_title: str
) -> bool:
    """Two states belong together when the category matches and titles do not conflict.

    An empty title is treated as unknown rather than as a distinct state, so a
    frame where the model could not read the game name does not split a segment.
    """
    if left_category is not right_category:
        return False
    if not left_title or not right_title:
        return True
    return left_title == right_title


def build_runs(samples: Sequence[SampleResult], settings: SmoothingSettings) -> list[Run]:
    """Run-length encode samples into compatible stretches.

    Samples below the confidence floor carry the current state forward instead
    of opening a new one; a hesitant verdict is weaker evidence than continuity.
    """
    runs: list[Run] = []
    for sample in samples:
        category = sample.classification.primary_category
        title = normalise_title(sample.classification.specific_title_or_context)

        if runs and sample.classification.confidence_score < settings.min_confidence:
            runs[-1].samples.append(sample)
            continue

        if runs and compatible(runs[-1].category, runs[-1].title, category, title):
            current = runs[-1]
            current.samples.append(sample)
            if not current.title and title:
                current.title = title
            continue

        runs.append(Run(category=category, title=title, samples=[sample]))
    return runs


def _run_span(runs: list[Run], index: int, end_seconds: float) -> float:
    start = runs[index].start
    end = runs[index + 1].start if index + 1 < len(runs) else end_seconds
    return max(0.0, end - start)


def _absorb(target: Run, victim: Run) -> None:
    target.samples.extend(victim.samples)
    target.samples.sort(key=lambda s: s.offset_seconds)
    if not target.title and victim.title:
        target.title = victim.title


def is_confirmed(run: Run, settings: SmoothingSettings) -> bool:
    """Decide whether a run has earned its own segment.

    Confirmation comes from repeated observation: several samples agreeing, or
    two samples far enough apart to show the state held. The gap to the *next*
    run is deliberately not evidence, since a lone verdict says nothing about
    what happened in the minutes after it was taken.
    """
    if len(run.samples) >= settings.confirm_consecutive:
        return True
    internal_span = run.last_offset - run.start
    return internal_span >= settings.min_segment_seconds


def median_sample_interval(samples: Sequence[SampleResult]) -> float:
    """Median gap between consecutive samples, used to scale the alt-tab window."""
    offsets = sorted(s.offset_seconds for s in samples)
    gaps = [b - a for a, b in zip(offsets, offsets[1:]) if b > a]
    if not gaps:
        return 0.0
    gaps.sort()
    middle = len(gaps) // 2
    if len(gaps) % 2:
        return gaps[middle]
    return (gaps[middle - 1] + gaps[middle]) / 2.0


def alt_tab_ceiling(settings: SmoothingSettings, sample_interval: float) -> float:
    """How much timeline an A-B-A absorption may rewrite.

    The configured window is an absolute floor. When sampling is sparse, the
    forward span of a lone verdict is dominated by the sampling cadence rather
    than by the interruption itself, so anything shorter than half an interval
    is unresolvable and treated as noise. Beyond that the state is kept, because
    absorbing it would rewrite real time the samples never observed.
    """
    return max(settings.alt_tab_window_seconds, sample_interval / 2.0)


def absorb_transients(
    runs: list[Run],
    settings: SmoothingSettings,
    end_seconds: float,
    *,
    sample_interval: float = 0.0,
) -> list[Run]:
    """Fold short-lived runs into their neighbours.

    An unconfirmed run inside an A-B-A sandwich, occupying less than the alt-tab
    ceiling, is a desktop click or a loading screen and rejoins A. An unconfirmed
    run too short to reach the segment floor merges into its stronger neighbour.
    An unconfirmed run that nonetheless spans a long stretch is kept: one sample
    is the only evidence covering that time, and discarding it would erase a
    real state rather than smooth noise.
    """
    if len(runs) < 2:
        return runs

    ceiling = alt_tab_ceiling(settings, sample_interval)
    working = list(runs)
    changed = True
    while changed and len(working) > 1:
        changed = False
        for index in range(len(working)):
            run = working[index]
            if is_confirmed(run, settings):
                continue

            span = _run_span(working, index, end_seconds)
            previous = working[index - 1] if index > 0 else None
            following = working[index + 1] if index + 1 < len(working) else None

            if (
                previous is not None
                and following is not None
                and compatible(
                    previous.category, previous.title, following.category, following.title
                )
                and span < ceiling
            ):
                _absorb(previous, run)
                _absorb(previous, following)
                del working[index : index + 2]
                changed = True
                break

            if span >= settings.min_segment_seconds:
                continue

            host = previous or following
            if host is None:
                continue
            if previous is not None and following is not None:
                host = previous if len(previous.samples) >= len(following.samples) else following
            _absorb(host, run)
            del working[index]
            changed = True
            break

    return working


def merge_compatible(runs: list[Run]) -> list[Run]:
    """Collapse neighbouring runs that became compatible after absorption."""
    merged: list[Run] = []
    for run in runs:
        if merged and compatible(merged[-1].category, merged[-1].title, run.category, run.title):
            _absorb(merged[-1], run)
            continue
        merged.append(run)
    return merged


def runs_to_segments(
    runs: Sequence[Run], end_seconds: float, *, start_seconds: float = 0.0
) -> list[Segment]:
    """Convert runs into wall-clock segments spanning the whole VOD.

    The first segment is extended back to the VOD start and the last forward to
    the end so the timeline has no gaps, which matters for the duration shares
    in the summary report.
    """
    segments: list[Segment] = []
    for index, run in enumerate(runs):
        start = start_seconds if index == 0 else run.start
        end = runs[index + 1].start if index + 1 < len(runs) else end_seconds
        if end <= start:
            continue
        segments.append(
            Segment(
                start_seconds=round(start, 3),
                end_seconds=round(end, 3),
                primary_category=run.category,
                specific_title_or_context=run.best_title(),
                sub_activity=run.dominant_sub_activity(),
                is_afk_or_brb=run.afk_ratio >= 0.5,
                confidence_score=round(run.confidence, 4),
                sample_count=len(run.samples),
            )
        )
    return segments


def enforce_minimum_duration(
    segments: list[Segment], min_seconds: float
) -> list[Segment]:
    """Merge any segment still shorter than the floor into its stronger neighbour."""
    if len(segments) < 2:
        return segments

    working = list(segments)
    changed = True
    while changed and len(working) > 1:
        changed = False
        for index, segment in enumerate(working):
            if segment.duration_seconds >= min_seconds:
                continue
            previous = working[index - 1] if index > 0 else None
            following = working[index + 1] if index + 1 < len(working) else None
            host_index = index - 1 if previous is not None else index + 1
            if previous is not None and following is not None:
                host_index = (
                    index - 1
                    if previous.duration_seconds >= following.duration_seconds
                    else index + 1
                )
            host = working[host_index]
            host.start_seconds = min(host.start_seconds, segment.start_seconds)
            host.end_seconds = max(host.end_seconds, segment.end_seconds)
            host.sample_count += segment.sample_count
            host.absorbed_samples += segment.sample_count
            del working[index]
            changed = True
            break

    return _merge_identical_neighbours(working)


def _merge_identical_neighbours(segments: list[Segment]) -> list[Segment]:
    merged: list[Segment] = []
    for segment in segments:
        if merged and compatible(
            merged[-1].primary_category,
            normalise_title(merged[-1].specific_title_or_context),
            segment.primary_category,
            normalise_title(segment.specific_title_or_context),
        ):
            head = merged[-1]
            head.end_seconds = max(head.end_seconds, segment.end_seconds)
            head.sample_count += segment.sample_count
            if not normalise_title(head.specific_title_or_context):
                head.specific_title_or_context = segment.specific_title_or_context
            continue
        merged.append(segment)
    return merged


def smooth(
    samples: Sequence[SampleResult],
    duration_seconds: float,
    settings: SmoothingSettings,
) -> list[Segment]:
    """Full smoothing pipeline: samples in, stable segments out."""
    ordered = sorted(samples, key=lambda s: s.offset_seconds)
    if not ordered:
        return []

    runs = build_runs(ordered, settings)
    runs = absorb_transients(
        runs, settings, duration_seconds, sample_interval=median_sample_interval(ordered)
    )
    runs = merge_compatible(runs)
    segments = runs_to_segments(runs, duration_seconds)
    segments = enforce_minimum_duration(segments, settings.min_segment_seconds)
    log.info("smoothed %d samples into %d segments", len(ordered), len(segments))
    return segments
