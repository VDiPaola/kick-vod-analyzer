from __future__ import annotations

import json

import pytest

from kick_vod_analyser.ingest.chat import ChatIndex, ChatSource, JsonlChatSource, NullChatSource
from kick_vod_analyser.models import ChatMessage, VodInfo
from kick_vod_analyser.pipeline import Pipeline, RunOptions, write_run_report
from tests.conftest import requires_ffmpeg


class StubChatSource(ChatSource):
    name = "stub"

    def __init__(self, messages):
        self.messages = messages

    def fetch(self, vod):
        return ChatIndex(self.messages)


@pytest.fixture
def local_vod(synthetic_video):
    return VodInfo(
        vod_id="synthetic",
        url=str(synthetic_video),
        channel_slug="teststreamer",
        channel_id=1,
        title="synthetic test stream",
        duration_seconds=90.0,
        started_at_epoch=1_700_000_000.0,
        playback_url=str(synthetic_video),
    )


@pytest.fixture
def local_pipeline(settings, local_vod, monkeypatch):
    """A pipeline wired to a local file instead of a live Kick VOD."""

    def build(chat_source=None, **setting_overrides):
        for key, value in setting_overrides.items():
            target, _, attribute = key.rpartition(".")
            holder = getattr(settings, target) if target else settings
            setattr(holder, attribute, value)

        pipeline = Pipeline(
            settings, chat_source=chat_source or NullChatSource(), progress=lambda *a: None
        )
        monkeypatch.setattr(pipeline, "_resolve", lambda url: local_vod)
        return pipeline

    return build


@requires_ffmpeg
class TestEndToEnd:
    def test_produces_a_complete_timeline(self, local_pipeline, settings):
        pipeline = local_pipeline(**{"sampling.heartbeat_seconds": 25.0})
        report = pipeline.run(RunOptions(url="local", provider="mock", chat_source_kind="none"))

        assert report.vod is not None
        assert report.sample_points, "sampling produced no points"
        assert report.grids > 0
        assert report.results, "no samples were classified"
        assert report.timeline is not None
        assert report.timeline.segments

    def test_writes_every_deliverable(self, local_pipeline, settings):
        pipeline = local_pipeline(**{"sampling.heartbeat_seconds": 25.0})
        report = pipeline.run(RunOptions(url="local", provider="mock", chat_source_kind="none"))

        assert set(report.outputs) == {
            "timeline_json",
            "chapters_vtt",
            "segments_csv",
            "summary_md",
        }
        for path in report.outputs.values():
            assert path.exists() and path.stat().st_size > 0

        payload = json.loads(report.outputs["timeline_json"].read_text(encoding="utf-8"))
        assert payload["provider"] == "mock"
        assert payload["segments"]

    def test_the_timeline_covers_the_whole_vod(self, local_pipeline):
        pipeline = local_pipeline(**{"sampling.heartbeat_seconds": 25.0})
        report = pipeline.run(RunOptions(url="local", provider="mock", chat_source_kind="none"))

        segments = report.timeline.segments
        assert segments[0].start_seconds == 0.0
        assert segments[-1].end_seconds == pytest.approx(90.0)

    def test_scene_detection_finds_the_synthetic_cuts(self, local_pipeline):
        pipeline = local_pipeline(**{"sampling.heartbeat_seconds": 3600.0})
        report = pipeline.run(RunOptions(url="local", provider="mock", chat_source_kind="none"))

        offsets = [p.offset_seconds for p in report.sample_points if p.trigger == "scene"]
        assert any(28.0 <= t <= 33.0 for t in offsets), offsets

    def test_chat_reaches_the_prompt(self, local_pipeline):
        messages = [
            ChatMessage(offset_seconds=float(t), username=f"v{t}", text="what game is this")
            for t in range(10, 80, 5)
        ]
        pipeline = local_pipeline(
            chat_source=StubChatSource(messages), **{"sampling.heartbeat_seconds": 25.0}
        )
        report = pipeline.run(RunOptions(url="local", provider="mock", chat_source_kind="stub"))

        assert report.timeline is not None
        chat_file = pipeline.settings.vod_work_dir("synthetic") / "chat.jsonl"
        assert chat_file.exists()
        assert "what game is this" in chat_file.read_text(encoding="utf-8")

    def test_a_second_run_reuses_cached_work(self, local_pipeline):
        pipeline = local_pipeline(**{"sampling.heartbeat_seconds": 25.0})
        options = RunOptions(url="local", provider="mock", chat_source_kind="none")

        first = pipeline.run(options)
        second = pipeline.run(options)

        assert len(second.results) == len(first.results)
        assert [r.offset_seconds for r in second.results] == [
            r.offset_seconds for r in first.results
        ]

    def test_no_resume_reclassifies_from_scratch(self, local_pipeline):
        pipeline = local_pipeline(**{"sampling.heartbeat_seconds": 25.0})
        pipeline.run(RunOptions(url="local", provider="mock", chat_source_kind="none"))
        report = pipeline.run(
            RunOptions(url="local", provider="mock", chat_source_kind="none", resume=False)
        )
        assert report.results

    def test_dry_run_makes_no_grids_and_reports_a_cost(self, local_pipeline):
        pipeline = local_pipeline(**{"sampling.heartbeat_seconds": 25.0})
        report = pipeline.run(
            RunOptions(url="local", provider="gemini", chat_source_kind="none", dry_run=True)
        )

        assert report.sample_points
        assert report.grids == 0
        assert report.timeline is None
        assert report.cost["total_cost_usd"] > 0

    def test_dry_run_needs_no_credentials(self, local_pipeline, settings):
        settings.gemini_api_key = None
        pipeline = local_pipeline(**{"sampling.heartbeat_seconds": 25.0})
        report = pipeline.run(
            RunOptions(url="local", provider="gemini", chat_source_kind="none", dry_run=True)
        )
        assert report.errors == []

    def test_the_sample_cap_is_honoured(self, local_pipeline):
        pipeline = local_pipeline(
            **{"sampling.heartbeat_seconds": 10.0, "sampling.max_samples": 2}
        )
        report = pipeline.run(RunOptions(url="local", provider="mock", chat_source_kind="none"))
        assert len(report.sample_points) == 2

    def test_frames_are_cleaned_up_by_default(self, local_pipeline):
        pipeline = local_pipeline(**{"sampling.heartbeat_seconds": 25.0})
        pipeline.run(RunOptions(url="local", provider="mock", chat_source_kind="none"))
        frames_dir = pipeline.settings.vod_work_dir("synthetic") / "frames"
        assert not frames_dir.exists() or not any(frames_dir.iterdir())

    def test_frames_can_be_retained(self, local_pipeline):
        pipeline = local_pipeline(**{"sampling.heartbeat_seconds": 25.0})
        pipeline.run(
            RunOptions(url="local", provider="mock", chat_source_kind="none", keep_frames=True)
        )
        frames_dir = pipeline.settings.vod_work_dir("synthetic") / "frames"
        assert frames_dir.exists() and any(frames_dir.iterdir())

    def test_run_report_is_serialisable(self, local_pipeline, tmp_path):
        pipeline = local_pipeline(**{"sampling.heartbeat_seconds": 25.0})
        report = pipeline.run(RunOptions(url="local", provider="mock", chat_source_kind="none"))

        path = write_run_report(report, tmp_path / "run_report.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["vod_id"] == "synthetic"
        assert payload["segments"] >= 1


@requires_ffmpeg
class TestChatFileIntegration:
    def test_a_chat_export_flows_through_to_the_prompt(self, local_pipeline, tmp_path):
        export = tmp_path / "chat.jsonl"
        export.write_text(
            "\n".join(
                json.dumps({"offset_seconds": t, "username": f"v{t}", "text": "he is watching youtube"})
                for t in range(5, 85, 5)
            ),
            encoding="utf-8",
        )
        pipeline = local_pipeline(
            chat_source=JsonlChatSource(export), **{"sampling.heartbeat_seconds": 25.0}
        )
        report = pipeline.run(RunOptions(url="local", provider="mock", chat_source_kind="file"))

        assert report.timeline is not None
        indexed = pipeline.settings.vod_work_dir("synthetic") / "chat.jsonl"
        assert "he is watching youtube" in indexed.read_text(encoding="utf-8")
