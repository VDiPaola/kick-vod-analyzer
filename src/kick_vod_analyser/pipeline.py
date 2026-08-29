"""End-to-end orchestration from VOD URL to timeline artefacts."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .chatwindow.slicer import build_window
from .classify.base import ClassificationRequest, ClassificationResponse, Classifier
from .classify.factory import build_classifier
from .classify.pricing import estimate_cost
from .config import Settings
from .ffmpeg import ensure_available, probe
from .ingest.chat import ChatIndex, ChatSource, NullChatSource
from .ingest.vod import resolve_vod
from .models import SamplePoint, SampleResult, Timeline, VodInfo
from .postprocess.outputs import write_all
from .postprocess.smoothing import smooth
from .sampling.burst import GridArtifact, build_grids
from .sampling.renditions import StreamPlan, plan_streams
from .sampling.scene import detect_scenes, plan_samples
from .store import Store

log = logging.getLogger(__name__)

ProgressHook = Callable[[str, str], None]


@dataclass
class RunOptions:
    """Everything that varies between runs of the same pipeline."""

    url: str
    provider: str = "gemini"
    model: str | None = None
    mode: str = "sync"
    chat_source_kind: str = "kick"
    chat_file: Path | None = None
    resume: bool = True
    keep_frames: bool = False
    dry_run: bool = False
    wait_for_batch: bool = True
    max_batch_wait_seconds: float = 24 * 3600.0


@dataclass
class RunReport:
    """Outcome of a pipeline run."""

    vod: VodInfo | None = None
    sample_points: list[SamplePoint] = field(default_factory=list)
    grids: int = 0
    results: list[SampleResult] = field(default_factory=list)
    timeline: Timeline | None = None
    outputs: dict[str, Path] = field(default_factory=dict)
    cost: dict[str, float] = field(default_factory=dict)
    batch_job_id: str | None = None
    errors: list[str] = field(default_factory=list)


class Pipeline:
    """Stage-by-stage runner. Each stage caches so a re-run resumes cheaply."""

    def __init__(
        self,
        settings: Settings,
        *,
        chat_source: ChatSource | None = None,
        progress: ProgressHook | None = None,
    ) -> None:
        self.settings = settings
        self.chat_source = chat_source or NullChatSource()
        self.progress = progress or (lambda stage, message: log.info("[%s] %s", stage, message))

    def run(self, options: RunOptions) -> RunReport:
        report = RunReport()
        ensure_available(self.settings.ffmpeg_binary, self.settings.ffprobe_binary)

        vod = self._resolve(options.url)
        report.vod = vod
        work_dir = self.settings.vod_work_dir(vod.vod_id)
        work_dir.mkdir(parents=True, exist_ok=True)

        with Store(self.settings.work_dir / "cache.sqlite") as store:
            store.save_vod(vod)

            streams = self._plan_streams(vod)
            points = self._plan(vod, store, streams, resume=options.resume)
            report.sample_points = points
            if not points:
                report.errors.append("sampling produced no points")
                return report

            if options.dry_run:
                report.cost = estimate_cost(
                    len(points),
                    options.model or self._default_model(options.provider),
                    batch=options.mode == "batch",
                    with_chat=not isinstance(self.chat_source, NullChatSource),
                )
                self.progress("dry-run", f"{len(points)} samples planned")
                return report

            chat = self._fetch_chat(vod, work_dir)
            grids = self._build_grids(
                vod, points, work_dir, streams, keep_frames=options.keep_frames
            )
            report.grids = len(grids)
            if not grids:
                report.errors.append("no grids were produced")
                return report

            classifier = build_classifier(options.provider, self.settings, model=options.model)
            requests = self._build_requests(vod, grids, chat)

            cached = store.load_results(vod.vod_id, classifier.model) if options.resume else []
            cached_ids = {f"t{int(round(r.offset_seconds))}" for r in cached}
            pending = [r for r in requests if r.custom_id not in cached_ids]
            if cached:
                self.progress("classify", f"reusing {len(cached)} cached classifications")

            responses = self._classify(classifier, pending, work_dir, options, report)
            fresh = self._to_results(responses, grids, report)
            store.save_results(vod.vod_id, classifier.model, fresh)

            results = sorted(cached + fresh, key=lambda r: r.offset_seconds)
            report.results = results

            timeline = Timeline(
                vod=vod,
                segments=smooth(results, vod.duration_seconds, self.settings.smoothing),
                samples=results,
                provider=classifier.provider,
                model=classifier.model,
            )
            report.timeline = timeline
            report.outputs = write_all(timeline, self.settings.vod_out_dir(vod.vod_id))
            report.cost = estimate_cost(
                len(pending),
                classifier.model,
                batch=options.mode == "batch",
                with_chat=bool(chat),
                input_tokens=classifier.usage.input_tokens or None,
                output_tokens=classifier.usage.output_tokens or None,
            )
            self.progress("done", f"{len(timeline.segments)} segments written")

        return report

    def _default_model(self, provider: str) -> str:
        if provider == "openai":
            return self.settings.openai_model
        if provider == "mock":
            return "mock-v1"
        return self.settings.gemini_model

    def _resolve(self, url: str) -> VodInfo:
        self.progress("resolve", f"resolving {url}")
        vod = resolve_vod(url, timeout=self.settings.http_timeout)
        if vod.playback_url:
            try:
                probed = probe(vod.playback_url, ffprobe=self.settings.ffprobe_binary)
            except Exception as exc:
                log.warning("ffprobe on the playback URL failed: %s", exc)
            else:
                if abs(probed.duration_seconds - vod.duration_seconds) > 60:
                    log.info(
                        "using ffprobe duration %.0fs over API duration %.0fs",
                        probed.duration_seconds,
                        vod.duration_seconds,
                    )
                    vod = vod.model_copy(update={"duration_seconds": probed.duration_seconds})
        self.progress("resolve", f"{vod.channel_slug} / {vod.duration_seconds / 3600:.2f}h")
        return vod

    def _plan_streams(self, vod: VodInfo) -> StreamPlan:
        source = vod.playback_url or vod.url
        streams = plan_streams(
            source,
            extract_height=self.settings.sampling.grid_height,
            timeout=self.settings.http_timeout,
        )
        if streams.is_split:
            detect = streams.detect_rendition
            extract = streams.extract_rendition
            self.progress(
                "streams",
                f"detecting on {detect.label if detect else 'source'}, "
                f"extracting from {extract.label if extract else 'source'}",
            )
        return streams

    def _plan(
        self, vod: VodInfo, store: Store, streams: StreamPlan, *, resume: bool
    ) -> list[SamplePoint]:
        cached = store.load_scene_points(vod.vod_id) if resume else []
        if cached:
            self.progress("sample", f"reusing {len(cached)} cached scene points")
            scenes = [p for p in cached if p.trigger == "scene"]
        else:
            self.progress("sample", "running ffmpeg scene detection")
            scenes = detect_scenes(
                streams.detect_url,
                threshold=self.settings.sampling.scene_threshold,
                ffmpeg=self.settings.ffmpeg_binary,
                log_path=self.settings.vod_work_dir(vod.vod_id) / "scene_detect.log",
            )
            store.save_scene_points(vod.vod_id, scenes)

        points = plan_samples(
            vod.duration_seconds,
            scenes,
            heartbeat_interval=self.settings.sampling.heartbeat_seconds,
            min_gap_seconds=self.settings.sampling.min_gap_seconds,
            max_samples=self.settings.sampling.max_samples,
        )
        self.progress(
            "sample", f"{len(scenes)} scene triggers -> {len(points)} classification points"
        )
        return points

    def _fetch_chat(self, vod: VodInfo, work_dir: Path) -> ChatIndex:
        self.progress("chat", f"fetching chat via {self.chat_source.name}")
        chat = self.chat_source.fetch(vod)
        if chat:
            chat.to_jsonl(work_dir / "chat.jsonl")
        self.progress("chat", f"{len(chat)} messages indexed")
        return chat

    def _build_grids(
        self,
        vod: VodInfo,
        points: list[SamplePoint],
        work_dir: Path,
        streams: StreamPlan,
        *,
        keep_frames: bool,
    ) -> list[GridArtifact]:
        self.progress("frames", f"extracting bursts for {len(points)} points")
        grids = build_grids(
            streams.extract_url,
            points,
            work_dir,
            settings=self.settings.sampling,
            duration_seconds=vod.duration_seconds,
            ffmpeg=self.settings.ffmpeg_binary,
            keep_frames=keep_frames,
        )
        self.progress("frames", f"{len(grids)} grids composited")
        return grids

    def _build_requests(
        self, vod: VodInfo, grids: list[GridArtifact], chat: ChatIndex
    ) -> list[ClassificationRequest]:
        from .classify.prompts import build_user_prompt

        requests: list[ClassificationRequest] = []
        for grid in grids:
            offset = grid.point.offset_seconds
            window = (
                build_window(
                    chat.window(offset, self.settings.chat.window_seconds),
                    offset,
                    self.settings.chat,
                )
                if chat
                else None
            )
            requests.append(
                ClassificationRequest(
                    custom_id=grid.point.custom_id,
                    offset_seconds=offset,
                    image_path=grid.path,
                    user_prompt=build_user_prompt(
                        offset,
                        window,
                        channel_slug=vod.channel_slug,
                        stream_title=vod.title,
                    ),
                )
            )
        return requests

    def _classify(
        self,
        classifier: Classifier,
        requests: list[ClassificationRequest],
        work_dir: Path,
        options: RunOptions,
        report: RunReport,
    ) -> list[ClassificationResponse]:
        if not requests:
            return []
        if options.mode == "batch" and classifier.supports_batch():
            return self._classify_batch(classifier, requests, work_dir, options, report)
        self.progress("classify", f"classifying {len(requests)} samples synchronously")
        return classifier.classify(requests)

    def _classify_batch(
        self,
        classifier: Classifier,
        requests: list[ClassificationRequest],
        work_dir: Path,
        options: RunOptions,
        report: RunReport,
    ) -> list[ClassificationResponse]:
        job_id = classifier.submit_batch(requests, work_dir)
        report.batch_job_id = job_id
        self.progress("batch", f"submitted job {job_id}")

        if not options.wait_for_batch:
            report.errors.append(
                f"batch job {job_id} submitted but not awaited; "
                f"re-run without --no-wait to collect the results"
            )
            return []

        deadline = time.time() + options.max_batch_wait_seconds
        terminal = {
            "JOB_STATE_SUCCEEDED",
            "JOB_STATE_FAILED",
            "JOB_STATE_CANCELLED",
            "JOB_STATE_EXPIRED",
            "JOB_STATE_PARTIALLY_SUCCEEDED",
            "completed",
            "failed",
            "cancelled",
            "expired",
        }
        state = ""
        while time.time() < deadline:
            state = classifier.poll_batch(job_id)
            self.progress("batch", f"state {state}")
            if state in terminal:
                break
            time.sleep(self.settings.batch_poll_seconds)

        if state not in terminal:
            report.errors.append(f"batch job {job_id} did not finish before the deadline")
            return []
        if state in {"JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED", "failed"}:
            report.errors.append(f"batch job {job_id} ended in state {state}")
            return []

        return classifier.fetch_batch(job_id, work_dir)

    def _to_results(
        self,
        responses: list[ClassificationResponse],
        grids: list[GridArtifact],
        report: RunReport,
    ) -> list[SampleResult]:
        by_id = {grid.point.custom_id: grid for grid in grids}
        results: list[SampleResult] = []
        for response in responses:
            grid = by_id.get(response.custom_id)
            if grid is None:
                report.errors.append(f"response {response.custom_id} has no matching grid")
                continue
            if response.classification is None:
                report.errors.append(
                    f"{response.custom_id}: {response.error or 'no classification'}"
                )
                continue
            results.append(
                SampleResult(
                    offset_seconds=grid.point.offset_seconds,
                    trigger=grid.point.trigger,
                    classification=response.classification,
                    grid_path=str(grid.path),
                )
            )
        return results


def write_run_report(report: RunReport, path: Path) -> Path:
    """Persist the run report for debugging and cost review."""
    payload = {
        "vod_id": report.vod.vod_id if report.vod else None,
        "sample_points": len(report.sample_points),
        "grids": report.grids,
        "results": len(report.results),
        "segments": len(report.timeline.segments) if report.timeline else 0,
        "batch_job_id": report.batch_job_id,
        "cost": report.cost,
        "errors": report.errors,
        "outputs": {name: str(p) for name, p in report.outputs.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
