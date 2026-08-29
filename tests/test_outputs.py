from __future__ import annotations

import csv
import json

import pytest

from kick_vod_analyser.models import PrimaryCategory, Segment, Timeline
from kick_vod_analyser.postprocess.outputs import (
    afk_seconds,
    category_breakdown,
    format_duration,
    format_timestamp,
    title_breakdown,
    write_all,
    write_chapters_vtt,
    write_segments_csv,
    write_summary_report,
    write_timeline_json,
)
from tests.conftest import make_sample


@pytest.fixture
def timeline(vod):
    segments = [
        Segment(
            start_seconds=0.0,
            end_seconds=1800.0,
            primary_category=PrimaryCategory.GAMING,
            specific_title_or_context="Valorant",
            sub_activity="In-Game Match",
            confidence_score=0.91,
            sample_count=6,
        ),
        Segment(
            start_seconds=1800.0,
            end_seconds=2700.0,
            primary_category=PrimaryCategory.REACTION,
            specific_title_or_context="YouTube",
            sub_activity="Watching YouTube Video",
            confidence_score=0.78,
            sample_count=3,
        ),
        Segment(
            start_seconds=2700.0,
            end_seconds=3600.0,
            primary_category=PrimaryCategory.INTERMISSION,
            specific_title_or_context="",
            sub_activity="BRB screen",
            is_afk_or_brb=True,
            confidence_score=0.95,
            sample_count=3,
        ),
    ]
    return Timeline(
        vod=vod,
        segments=segments,
        samples=[make_sample(100.0), make_sample(2000.0)],
        provider="mock",
        model="mock-v1",
    )


class TestFormatting:
    @pytest.mark.parametrize(
        "seconds,expected",
        [(0, "00:00:00"), (59.9, "00:00:59"), (3600, "01:00:00"), (36061, "10:01:01")],
    )
    def test_timestamp(self, seconds, expected):
        assert format_timestamp(seconds) == expected

    def test_timestamp_with_millis_matches_webvtt(self):
        assert format_timestamp(3661.5, millis=True) == "01:01:01.500"

    def test_negative_input_clamps_to_zero(self):
        assert format_timestamp(-5.0) == "00:00:00"

    @pytest.mark.parametrize(
        "seconds,expected", [(45, "45s"), (90, "1m 30s"), (3600, "1h 0m"), (36000, "10h 0m")]
    )
    def test_duration(self, seconds, expected):
        assert format_duration(seconds) == expected


class TestTimelineJson:
    def test_round_trips_as_valid_json(self, timeline, tmp_path):
        path = write_timeline_json(timeline, tmp_path / "timeline.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["segment_count"] == 3
        assert payload["vod"]["vod_id"] == "test-vod"

    def test_segments_carry_derived_fields(self, timeline, tmp_path):
        path = write_timeline_json(timeline, tmp_path / "timeline.json")
        segment = json.loads(path.read_text(encoding="utf-8"))["segments"][0]
        assert segment["start_timestamp"] == "00:00:00"
        assert segment["duration_seconds"] == pytest.approx(1800.0)
        assert segment["label"] == "Gaming: Valorant"

    def test_samples_are_included_for_provenance(self, timeline, tmp_path):
        path = write_timeline_json(timeline, tmp_path / "timeline.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert len(payload["samples"]) == 2
        assert payload["samples"][0]["timestamp"] == "00:01:40"


class TestChaptersVtt:
    def test_starts_with_the_webvtt_header(self, timeline, tmp_path):
        text = write_chapters_vtt(timeline, tmp_path / "c.vtt").read_text(encoding="utf-8")
        assert text.startswith("WEBVTT\n")

    def test_one_cue_per_segment(self, timeline, tmp_path):
        text = write_chapters_vtt(timeline, tmp_path / "c.vtt").read_text(encoding="utf-8")
        assert text.count("-->") == 3

    def test_cue_times_use_the_millisecond_form(self, timeline, tmp_path):
        text = write_chapters_vtt(timeline, tmp_path / "c.vtt").read_text(encoding="utf-8")
        assert "00:00:00.000 --> 00:30:00.000" in text

    def test_cues_are_contiguous_and_ordered(self, timeline, tmp_path):
        text = write_chapters_vtt(timeline, tmp_path / "c.vtt").read_text(encoding="utf-8")
        cues = [line for line in text.splitlines() if "-->" in line]
        ends = [c.split(" --> ")[1] for c in cues]
        starts = [c.split(" --> ")[0] for c in cues]
        assert ends[:-1] == starts[1:]

    def test_labels_appear_as_cue_text(self, timeline, tmp_path):
        text = write_chapters_vtt(timeline, tmp_path / "c.vtt").read_text(encoding="utf-8")
        assert "Gaming: Valorant" in text
        assert "Intermission / AFK / BRB" in text

    def test_an_empty_timeline_still_writes_a_valid_file(self, vod, tmp_path):
        empty = Timeline(vod=vod)
        text = write_chapters_vtt(empty, tmp_path / "c.vtt").read_text(encoding="utf-8")
        assert text.strip() == "WEBVTT"


class TestSegmentsCsv:
    def test_header_and_row_count(self, timeline, tmp_path):
        path = write_segments_csv(timeline, tmp_path / "s.csv")
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 3
        assert rows[0]["primary_category"] == "Gaming"

    def test_afk_is_written_as_an_integer_flag(self, timeline, tmp_path):
        path = write_segments_csv(timeline, tmp_path / "s.csv")
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert rows[0]["is_afk_or_brb"] == "0"
        assert rows[2]["is_afk_or_brb"] == "1"

    def test_no_blank_rows_are_emitted(self, timeline, tmp_path):
        text = write_segments_csv(timeline, tmp_path / "s.csv").read_text(encoding="utf-8")
        assert "\n\n" not in text


class TestBreakdowns:
    def test_categories_are_ranked_by_duration(self, timeline):
        breakdown = category_breakdown(timeline.segments)
        assert breakdown[0][0] == "Gaming"
        assert breakdown[0][1] == pytest.approx(1800.0)

    def test_categories_are_summed_across_segments(self, vod):
        segments = [
            Segment(start_seconds=0, end_seconds=100, primary_category=PrimaryCategory.GAMING),
            Segment(start_seconds=200, end_seconds=500, primary_category=PrimaryCategory.GAMING),
        ]
        assert category_breakdown(segments) == [("Gaming", 400.0)]

    def test_titles_exclude_blank_entries(self, timeline):
        titles = dict(title_breakdown(timeline.segments))
        assert set(titles) == {"Valorant", "YouTube"}

    def test_afk_counts_intermission_and_flagged_segments(self, timeline):
        assert afk_seconds(timeline.segments) == pytest.approx(900.0)


class TestSummaryReport:
    def test_includes_the_headline_metadata(self, timeline, tmp_path):
        text = write_summary_report(timeline, tmp_path / "r.md").read_text(encoding="utf-8")
        assert "teststreamer" in text
        assert "1h 0m" in text

    def test_shares_sum_to_one_hundred_percent(self, timeline, tmp_path):
        text = write_summary_report(timeline, tmp_path / "r.md").read_text(encoding="utf-8")
        import re

        table = text.split("## Category breakdown")[1].split("##")[0]
        shares = [float(m) for m in re.findall(r"(\d+\.\d)%", table)]
        assert sum(shares) == pytest.approx(100.0, abs=0.3)

    def test_reports_away_and_active_time(self, timeline, tmp_path):
        text = write_summary_report(timeline, tmp_path / "r.md").read_text(encoding="utf-8")
        assert "AFK, BRB, or intermission: 15m 0s" in text
        assert "Active content: 45m 0s" in text

    def test_timeline_table_lists_every_segment(self, timeline, tmp_path):
        text = write_summary_report(timeline, tmp_path / "r.md").read_text(encoding="utf-8")
        section = text.split("## Timeline")[1]
        assert section.count("| 00:") + section.count("| 01:") >= 3

    def test_handles_an_empty_timeline(self, vod, tmp_path):
        text = write_summary_report(Timeline(vod=vod), tmp_path / "r.md").read_text(
            encoding="utf-8"
        )
        assert "## Category breakdown" in text


class TestWriteAll:
    def test_writes_every_deliverable(self, timeline, tmp_path):
        outputs = write_all(timeline, tmp_path / "out")
        assert set(outputs) == {"timeline_json", "chapters_vtt", "segments_csv", "summary_md"}
        assert all(path.exists() and path.stat().st_size > 0 for path in outputs.values())

    def test_creates_the_output_directory(self, timeline, tmp_path):
        outputs = write_all(timeline, tmp_path / "deep" / "nested")
        assert outputs["timeline_json"].parent.exists()
