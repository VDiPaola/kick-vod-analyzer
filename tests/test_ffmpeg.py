from __future__ import annotations

import pytest
from PIL import Image

from kick_vod_analyser.ffmpeg import FfmpegError, ensure_available, extract_frames, probe
from tests.conftest import requires_ffmpeg


class TestEnsureAvailable:
    def test_passes_for_present_binaries(self):
        ensure_available("ffmpeg", "ffprobe")

    def test_names_the_missing_binaries(self):
        with pytest.raises(FfmpegError, match="definitely-not-a-real-binary"):
            ensure_available("ffmpeg", "definitely-not-a-real-binary")


@requires_ffmpeg
class TestProbe:
    def test_reads_duration_and_dimensions(self, synthetic_video):
        result = probe(str(synthetic_video))
        assert result.duration_seconds == pytest.approx(90.0, abs=1.0)
        assert (result.width, result.height) == (640, 360)
        assert result.codec == "h264"

    def test_a_missing_file_raises(self, tmp_path):
        with pytest.raises(FfmpegError):
            probe(str(tmp_path / "absent.mp4"))

    def test_a_non_media_file_raises(self, tmp_path):
        path = tmp_path / "notes.txt"
        path.write_text("not a video", encoding="utf-8")
        with pytest.raises(FfmpegError):
            probe(str(path))


@requires_ffmpeg
class TestExtractFrames:
    def test_writes_one_file_per_timestamp(self, synthetic_video, tmp_path):
        frames = extract_frames(str(synthetic_video), [5.0, 35.0, 65.0], tmp_path, width=320)
        assert len(frames) == 3
        assert all(path.exists() and path.stat().st_size > 0 for path in frames)

    def test_honours_the_requested_width(self, synthetic_video, tmp_path):
        frames = extract_frames(str(synthetic_video), [10.0], tmp_path, width=320)
        with Image.open(frames[0]) as image:
            assert image.width == 320

    def test_frames_from_different_acts_differ(self, synthetic_video, tmp_path):
        frames = extract_frames(str(synthetic_video), [10.0, 40.0, 70.0], tmp_path, width=160)
        digests = {path.read_bytes() for path in frames}
        assert len(digests) == 3

    def test_timestamps_past_the_end_are_skipped_not_fatal(self, synthetic_video, tmp_path):
        frames = extract_frames(str(synthetic_video), [10.0, 9999.0], tmp_path, width=160)
        assert len(frames) == 1

    def test_a_negative_timestamp_clamps_to_the_start(self, synthetic_video, tmp_path):
        assert len(extract_frames(str(synthetic_video), [-5.0], tmp_path, width=160)) == 1

    def test_an_unreadable_source_returns_nothing(self, tmp_path):
        assert extract_frames(str(tmp_path / "absent.mp4"), [1.0], tmp_path) == []

    def test_creates_the_destination_directory(self, synthetic_video, tmp_path):
        dest = tmp_path / "deep" / "nested"
        assert extract_frames(str(synthetic_video), [10.0], dest, width=160)
        assert dest.exists()

    def test_an_empty_timestamp_list_is_safe(self, synthetic_video, tmp_path):
        assert extract_frames(str(synthetic_video), [], tmp_path) == []
