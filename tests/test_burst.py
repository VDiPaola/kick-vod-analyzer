from __future__ import annotations

from pathlib import Path

from PIL import Image

from kick_vod_analyser.config import SamplingSettings
from kick_vod_analyser.models import SamplePoint
from kick_vod_analyser.sampling import burst as burst_module
from kick_vod_analyser.sampling.burst import build_grids, burst_timestamps
from tests.conftest import requires_ffmpeg


class TestBurstTimestamps:
    def test_expands_around_the_trigger(self):
        assert burst_timestamps(100.0, (-6, -2, 2, 6)) == [94.0, 98.0, 102.0, 106.0]

    def test_clamps_below_zero(self):
        assert burst_timestamps(3.0, (-6, -2, 2, 6)) == [0.0, 1.0, 5.0, 9.0]

    def test_clamps_against_the_media_end(self):
        stamps = burst_timestamps(99.0, (-6, -2, 2, 6), duration_seconds=100.0)
        assert max(stamps) <= 99.5
        assert stamps == [93.0, 97.0, 99.5, 99.5]

    def test_order_is_preserved_for_grid_placement(self):
        stamps = burst_timestamps(50.0, (-6, -2, 2, 6))
        assert stamps == sorted(stamps)

    def test_supports_a_custom_burst_shape(self):
        assert burst_timestamps(10.0, (0.0,)) == [10.0]


def fake_extractor(colours_by_call):
    """Replace ffmpeg extraction with deterministic solid-colour frames."""
    calls = {"n": 0}

    def extract(source, timestamps, dest_dir, **kwargs):
        index = calls["n"]
        calls["n"] += 1
        colour = colours_by_call[index % len(colours_by_call)]
        if colour is None:
            return []
        dest_dir.mkdir(parents=True, exist_ok=True)
        written = []
        for i, _ in enumerate(timestamps):
            path = Path(dest_dir) / f"frame_{i}.jpg"
            Image.new("RGB", (160, 90), colour).save(path, quality=90)
            written.append(path)
        return written

    return extract, calls


class TestBuildGrids:
    def test_builds_one_grid_per_point(self, tmp_path, monkeypatch):
        extract, _ = fake_extractor([(200, 10, 10), (10, 200, 10), (10, 10, 200)])
        monkeypatch.setattr(burst_module, "extract_frames", extract)
        points = [SamplePoint(offset_seconds=float(t), trigger="scene") for t in (100, 200, 300)]

        grids = build_grids(
            "src.m3u8", points, tmp_path, settings=SamplingSettings(), duration_seconds=1000.0
        )

        assert len(grids) == 3
        assert all(g.path.exists() for g in grids)
        assert [g.point.offset_seconds for g in grids] == [100.0, 200.0, 300.0]

    def test_skips_points_where_extraction_yields_nothing(self, tmp_path, monkeypatch):
        extract, _ = fake_extractor([(200, 10, 10), None, (10, 10, 200)])
        monkeypatch.setattr(burst_module, "extract_frames", extract)
        points = [SamplePoint(offset_seconds=float(t), trigger="scene") for t in (100, 200, 300)]

        grids = build_grids("src.m3u8", points, tmp_path, settings=SamplingSettings())

        assert [g.point.offset_seconds for g in grids] == [100.0, 300.0]

    def test_drops_near_duplicate_scene_grids(self, tmp_path, monkeypatch):
        extract, _ = fake_extractor([(120, 120, 120)])
        monkeypatch.setattr(burst_module, "extract_frames", extract)
        points = [SamplePoint(offset_seconds=float(t), trigger="scene") for t in (100, 200, 300)]

        grids = build_grids("src.m3u8", points, tmp_path, settings=SamplingSettings())

        assert len(grids) == 1

    def test_heartbeat_grids_survive_deduplication(self, tmp_path, monkeypatch):
        extract, _ = fake_extractor([(120, 120, 120)])
        monkeypatch.setattr(burst_module, "extract_frames", extract)
        points = [
            SamplePoint(offset_seconds=100.0, trigger="scene"),
            SamplePoint(offset_seconds=1000.0, trigger="heartbeat"),
        ]

        grids = build_grids("src.m3u8", points, tmp_path, settings=SamplingSettings())

        assert len(grids) == 2

    def test_dedupe_can_be_disabled(self, tmp_path, monkeypatch):
        extract, _ = fake_extractor([(120, 120, 120)])
        monkeypatch.setattr(burst_module, "extract_frames", extract)
        points = [SamplePoint(offset_seconds=float(t), trigger="scene") for t in (100, 200)]

        grids = build_grids(
            "src.m3u8", points, tmp_path, settings=SamplingSettings(), dedupe=False
        )

        assert len(grids) == 2

    def test_frames_are_removed_unless_retained(self, tmp_path, monkeypatch):
        extract, _ = fake_extractor([(200, 10, 10)])
        monkeypatch.setattr(burst_module, "extract_frames", extract)
        points = [SamplePoint(offset_seconds=100.0, trigger="scene")]

        build_grids("src.m3u8", points, tmp_path, settings=SamplingSettings())
        assert not (tmp_path / "frames" / "t100").exists()

        build_grids(
            "src.m3u8", points, tmp_path, settings=SamplingSettings(), keep_frames=True
        )
        assert (tmp_path / "frames" / "t100").exists()

    def test_grid_files_are_named_by_custom_id(self, tmp_path, monkeypatch):
        extract, _ = fake_extractor([(200, 10, 10)])
        monkeypatch.setattr(burst_module, "extract_frames", extract)
        points = [SamplePoint(offset_seconds=1234.6, trigger="scene")]

        grids = build_grids("src.m3u8", points, tmp_path, settings=SamplingSettings())

        assert grids[0].path.name == "t1235.jpg"

    def test_empty_point_list_is_safe(self, tmp_path):
        assert build_grids("src.m3u8", [], tmp_path, settings=SamplingSettings()) == []


@requires_ffmpeg
class TestBuildGridsIntegration:
    def test_extracts_real_frames_from_a_video(self, synthetic_video, tmp_path):
        points = [SamplePoint(offset_seconds=float(t), trigger="scene") for t in (15, 45, 75)]

        grids = build_grids(
            str(synthetic_video),
            points,
            tmp_path,
            settings=SamplingSettings(),
            duration_seconds=90.0,
        )

        assert len(grids) == 3
        for grid in grids:
            with Image.open(grid.path) as image:
                assert image.size == (1280, 720)
        assert len({g.phash for g in grids}) == 3, "visually distinct acts must hash apart"
