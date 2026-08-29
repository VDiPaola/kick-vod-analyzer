from __future__ import annotations

import pytest

from kick_vod_analyser import pipeline as pipeline_module
from kick_vod_analyser.ffmpeg import FfmpegError, ProbeResult
from kick_vod_analyser.ingest.chat import NullChatSource
from kick_vod_analyser.models import VodInfo
from kick_vod_analyser.pipeline import Pipeline


def make_vod(duration: float, playback: str | None = "https://cdn/x.m3u8") -> VodInfo:
    return VodInfo(
        vod_id="v1",
        url="https://kick.com/teststreamer/videos/v1",
        channel_slug="teststreamer",
        duration_seconds=duration,
        playback_url=playback,
    )


@pytest.fixture
def pipeline(settings):
    return Pipeline(settings, chat_source=NullChatSource(), progress=lambda *a: None)


class TestResolveReconciliation:
    def test_ffprobe_duration_wins_when_it_disagrees_materially(self, pipeline, monkeypatch):
        monkeypatch.setattr(pipeline_module, "resolve_vod", lambda url, timeout: make_vod(1000.0))
        monkeypatch.setattr(
            pipeline_module,
            "probe",
            lambda src, ffprobe: ProbeResult(36000.0, 1920, 1080, "h264"),
        )

        assert pipeline._resolve("url").duration_seconds == pytest.approx(36000.0)

    def test_a_small_disagreement_keeps_the_api_duration(self, pipeline, monkeypatch):
        monkeypatch.setattr(pipeline_module, "resolve_vod", lambda url, timeout: make_vod(36000.0))
        monkeypatch.setattr(
            pipeline_module,
            "probe",
            lambda src, ffprobe: ProbeResult(36010.0, 1920, 1080, "h264"),
        )

        assert pipeline._resolve("url").duration_seconds == pytest.approx(36000.0)

    def test_a_probe_failure_leaves_the_api_duration_intact(self, pipeline, monkeypatch):
        monkeypatch.setattr(pipeline_module, "resolve_vod", lambda url, timeout: make_vod(36000.0))

        def boom(src, ffprobe):
            raise FfmpegError("stream unreachable")

        monkeypatch.setattr(pipeline_module, "probe", boom)

        assert pipeline._resolve("url").duration_seconds == pytest.approx(36000.0)

    def test_no_playback_url_skips_probing_entirely(self, pipeline, monkeypatch):
        calls = []
        monkeypatch.setattr(
            pipeline_module, "resolve_vod", lambda url, timeout: make_vod(100.0, playback=None)
        )
        monkeypatch.setattr(pipeline_module, "probe", lambda *a, **k: calls.append(1))

        pipeline._resolve("url")
        assert calls == []
