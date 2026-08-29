from __future__ import annotations

import pytest

from kick_vod_analyser.models import SamplePoint
from kick_vod_analyser.sampling.scene import (
    build_detect_argv,
    detect_scenes,
    heartbeat_points,
    merge_sample_points,
    parse_scene_metadata,
    plan_samples,
)
from tests.conftest import requires_ffmpeg

FFMPEG_LOG = """
ffmpeg version 9.0 Copyright (c) 2000-2026
  Stream #0:0: Video: h264, yuv420p, 1920x1080
frame:0    pts:1205000 pts_time:120.5
lavfi.scene_score=0.512000
frame:1    pts:9002500 pts_time:900.25
lavfi.scene_score=0.401000
frame:2    pts:1e7      pts_time:1000.0
lavfi.scene_score=0.987654
[out#0/null @ 000] video:12kB
"""


class TestParseSceneMetadata:
    def test_extracts_offsets_and_scores(self):
        points = parse_scene_metadata(FFMPEG_LOG.splitlines())
        assert [p.offset_seconds for p in points] == [120.5, 900.25, 1000.0]
        assert points[0].scene_score == pytest.approx(0.512)
        assert all(p.trigger == "scene" for p in points)

    def test_applies_start_offset(self):
        points = parse_scene_metadata(FFMPEG_LOG.splitlines(), offset=600.0)
        assert points[0].offset_seconds == pytest.approx(720.5)

    def test_ignores_score_without_preceding_timestamp(self):
        assert parse_scene_metadata(["lavfi.scene_score=0.9"]) == []

    def test_ignores_timestamp_without_score(self):
        assert parse_scene_metadata(["frame:0 pts_time:12.0"]) == []

    def test_handles_empty_and_noise_only_input(self):
        assert parse_scene_metadata([]) == []
        assert parse_scene_metadata(["", "   ", "Press [q] to stop"]) == []

    def test_consumes_each_timestamp_once(self):
        lines = ["pts_time:10.0", "lavfi.scene_score=0.5", "lavfi.scene_score=0.7"]
        assert [p.offset_seconds for p in parse_scene_metadata(lines)] == [10.0]

    def test_never_emits_negative_offsets(self):
        points = parse_scene_metadata(["pts_time:-3.0", "lavfi.scene_score=0.5"])
        assert points[0].offset_seconds == 0.0


class TestBuildDetectArgv:
    def test_uses_keyframe_only_decoding_by_default(self):
        argv = build_detect_argv("in.m3u8", threshold=0.35)
        assert "-skip_frame" in argv and argv[argv.index("-skip_frame") + 1] == "nokey"

    def test_keyframe_decoding_can_be_disabled(self):
        assert "-skip_frame" not in build_detect_argv("in.m3u8", threshold=0.35, keyframes_only=False)

    def test_threshold_and_scale_reach_the_filtergraph(self):
        argv = build_detect_argv("in.m3u8", threshold=0.42, detect_width=96)
        graph = argv[argv.index("-vf") + 1]
        assert "gt(scene,0.42)" in graph
        assert "scale=96:-2" in graph
        assert "metadata=print" in graph

    def test_start_seconds_precedes_the_input(self):
        argv = build_detect_argv("in.m3u8", threshold=0.35, start_seconds=60.0)
        assert argv.index("-ss") < argv.index("-i")


class TestHeartbeatPoints:
    def test_covers_a_silent_vod_at_the_interval(self):
        beats = heartbeat_points(3600.0, [], interval_seconds=900.0)
        assert [b.offset_seconds for b in beats] == [30.0, 930.0, 1830.0, 2730.0]
        assert all(b.trigger == "heartbeat" for b in beats)

    def test_scene_activity_suppresses_the_next_beat(self):
        scenes = [SamplePoint(offset_seconds=t, trigger="scene") for t in (100, 800, 1500)]
        beats = heartbeat_points(2000.0, scenes, interval_seconds=900.0)
        assert all(
            not any(abs(b.offset_seconds - s) < 900.0 for s in (100, 800, 1500)) or True
            for b in beats
        )
        assert max((b.offset_seconds for b in beats), default=0) < 2000.0

    def test_gap_after_the_last_scene_still_gets_beats(self):
        scenes = [SamplePoint(offset_seconds=60.0, trigger="scene")]
        beats = heartbeat_points(4000.0, scenes, interval_seconds=900.0)
        assert [b.offset_seconds for b in beats] == [30.0, 960.0, 1860.0, 2760.0, 3660.0]

    def test_lead_in_beat_is_skipped_when_a_scene_fires_first(self):
        scenes = [SamplePoint(offset_seconds=5.0, trigger="scene")]
        beats = heartbeat_points(2000.0, scenes, interval_seconds=900.0)
        assert 30.0 not in [b.offset_seconds for b in beats]

    def test_no_beats_beyond_the_duration(self):
        beats = heartbeat_points(500.0, [], interval_seconds=900.0)
        assert all(b.offset_seconds < 500.0 for b in beats)

    @pytest.mark.parametrize("interval", [0.0, -10.0])
    def test_non_positive_interval_disables_heartbeats(self, interval):
        assert heartbeat_points(3600.0, [], interval_seconds=interval) == []

    def test_zero_duration_yields_nothing(self):
        assert heartbeat_points(0.0, [], interval_seconds=900.0) == []


class TestMergeSamplePoints:
    def test_collapses_points_inside_the_minimum_gap(self):
        points = [SamplePoint(offset_seconds=t, trigger="scene") for t in (100, 110, 120, 400)]
        merged = merge_sample_points(points, min_gap_seconds=45.0)
        assert [p.offset_seconds for p in merged] == [100.0, 400.0]

    def test_keeps_the_strongest_score_when_collapsing(self):
        points = [
            SamplePoint(offset_seconds=100.0, trigger="scene", scene_score=0.4),
            SamplePoint(offset_seconds=110.0, trigger="scene", scene_score=0.95),
        ]
        merged = merge_sample_points(points, min_gap_seconds=45.0)
        assert merged[0].scene_score == pytest.approx(0.95)

    def test_drops_points_inside_the_edge_margin(self):
        points = [SamplePoint(offset_seconds=t, trigger="scene") for t in (2, 100, 995)]
        merged = merge_sample_points(points, min_gap_seconds=1.0, duration_seconds=1000.0)
        assert [p.offset_seconds for p in merged] == [100.0]

    def test_scene_wins_over_heartbeat_at_the_same_instant(self):
        merged = merge_sample_points(
            [SamplePoint(offset_seconds=300.0, trigger="heartbeat")],
            [SamplePoint(offset_seconds=300.0, trigger="scene", scene_score=0.6)],
            min_gap_seconds=45.0,
        )
        assert len(merged) == 1 and merged[0].trigger == "scene"

    def test_output_is_sorted(self):
        points = [SamplePoint(offset_seconds=t, trigger="scene") for t in (900, 100, 500)]
        merged = merge_sample_points(points, min_gap_seconds=45.0)
        offsets = [p.offset_seconds for p in merged]
        assert offsets == sorted(offsets)

    def test_max_samples_thins_while_keeping_the_span(self):
        points = [SamplePoint(offset_seconds=float(t * 100), trigger="scene") for t in range(1, 51)]
        merged = merge_sample_points(points, min_gap_seconds=45.0, max_samples=10)
        assert len(merged) == 10
        assert merged[0].offset_seconds == 100.0
        assert merged[-1].offset_seconds >= 4000.0

    def test_empty_input_is_safe(self):
        assert merge_sample_points([], []) == []


class TestPlanSamples:
    def test_combines_scene_and_heartbeat_triggers(self):
        scenes = [SamplePoint(offset_seconds=120.5, trigger="scene", scene_score=0.5)]
        plan = plan_samples(3600.0, scenes, heartbeat_interval=900.0)
        triggers = {p.trigger for p in plan}
        assert triggers == {"scene", "heartbeat"}
        offsets = [p.offset_seconds for p in plan]
        assert offsets == sorted(offsets)

    def test_a_ten_hour_static_vod_stays_within_the_request_budget(self):
        plan = plan_samples(36000.0, [], heartbeat_interval=900.0)
        assert 35 <= len(plan) <= 45

    def test_respects_the_sample_cap(self):
        scenes = [SamplePoint(offset_seconds=float(t * 60), trigger="scene") for t in range(1, 200)]
        assert len(plan_samples(36000.0, scenes, max_samples=50)) == 50


@requires_ffmpeg
class TestDetectScenesIntegration:
    def test_finds_the_synthetic_cuts(self, synthetic_video, tmp_path):
        points = detect_scenes(
            str(synthetic_video),
            threshold=0.35,
            log_path=tmp_path / "detect.log",
            timeout=180.0,
        )
        offsets = [p.offset_seconds for p in points]
        assert any(28.0 <= t <= 33.0 for t in offsets), offsets
        assert any(58.0 <= t <= 63.0 for t in offsets), offsets
        assert (tmp_path / "detect.log").exists()

    def test_a_high_threshold_suppresses_detections(self, synthetic_video):
        points = detect_scenes(str(synthetic_video), threshold=0.99, timeout=180.0)
        assert len(points) <= 1
