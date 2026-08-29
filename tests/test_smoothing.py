from __future__ import annotations

import pytest

from kick_vod_analyser.config import SmoothingSettings
from kick_vod_analyser.models import PrimaryCategory
from kick_vod_analyser.postprocess.smoothing import (
    alt_tab_ceiling,
    build_runs,
    compatible,
    median_sample_interval,
    normalise_title,
    smooth,
)
from tests.conftest import make_sample

GAMING = PrimaryCategory.GAMING
CHATTING = PrimaryCategory.JUST_CHATTING
REACTION = PrimaryCategory.REACTION
INTERMISSION = PrimaryCategory.INTERMISSION
OFFLINE = PrimaryCategory.OFFLINE


def labels(segments):
    return [s.label for s in segments]


class TestNormaliseTitle:
    @pytest.mark.parametrize("raw", ["", "  ", "N/A", "none", "Unknown", "unclear", "-"])
    def test_placeholders_normalise_to_empty(self, raw):
        assert normalise_title(raw) == ""

    def test_case_and_spacing_are_folded(self):
        assert normalise_title("  Grand   Theft  Auto V ") == "grand theft auto v"


class TestCompatible:
    def test_different_categories_never_match(self):
        assert not compatible(GAMING, "valorant", CHATTING, "valorant")

    def test_same_category_and_title_matches(self):
        assert compatible(GAMING, "valorant", GAMING, "valorant")

    def test_an_unknown_title_matches_anything_in_the_category(self):
        assert compatible(GAMING, "", GAMING, "valorant")
        assert compatible(GAMING, "valorant", GAMING, "")

    def test_conflicting_titles_do_not_match(self):
        assert not compatible(GAMING, "valorant", GAMING, "gta v")


class TestBuildRuns:
    def test_consecutive_identical_samples_form_one_run(self, smoothing_settings):
        samples = [make_sample(t, GAMING, "Valorant") for t in (100, 200, 300)]
        runs = build_runs(samples, smoothing_settings)
        assert len(runs) == 1 and len(runs[0].samples) == 3

    def test_a_category_change_opens_a_new_run(self, smoothing_settings):
        samples = [make_sample(100, GAMING), make_sample(200, CHATTING, "")]
        assert len(build_runs(samples, smoothing_settings)) == 2

    def test_an_unknown_title_joins_the_current_run_and_inherits_it(self, smoothing_settings):
        samples = [make_sample(100, GAMING, ""), make_sample(200, GAMING, "Valorant")]
        runs = build_runs(samples, smoothing_settings)
        assert len(runs) == 1
        assert runs[0].title == "valorant"

    def test_low_confidence_samples_carry_the_current_state_forward(self, smoothing_settings):
        samples = [
            make_sample(100, GAMING, "Valorant", confidence=0.9),
            make_sample(160, OFFLINE, "", confidence=0.1),
            make_sample(220, GAMING, "Valorant", confidence=0.9),
        ]
        runs = build_runs(samples, smoothing_settings)
        assert len(runs) == 1
        assert runs[0].category is GAMING

    def test_a_leading_low_confidence_sample_still_opens_a_run(self, smoothing_settings):
        runs = build_runs([make_sample(100, GAMING, confidence=0.05)], smoothing_settings)
        assert len(runs) == 1

    def test_best_title_prefers_the_most_confident_specific_reading(self, smoothing_settings):
        samples = [
            make_sample(100, GAMING, "Valorent", confidence=0.5),
            make_sample(200, GAMING, "", confidence=0.9),
        ]
        runs = build_runs(samples, smoothing_settings)
        assert runs[0].best_title() == "Valorent"


class TestAltTabAbsorption:
    def test_a_brief_desktop_click_rejoins_the_game(self, smoothing_settings):
        samples = [
            make_sample(100, GAMING, "Valorant"),
            make_sample(200, GAMING, "Valorant"),
            make_sample(240, CHATTING, ""),
            make_sample(280, GAMING, "Valorant"),
            make_sample(400, GAMING, "Valorant"),
        ]
        segments = smooth(samples, 600.0, smoothing_settings)
        assert labels(segments) == ["Gaming: Valorant"]

    def test_a_loading_screen_does_not_split_the_game(self, smoothing_settings):
        samples = [
            make_sample(100, GAMING, "Valorant"),
            make_sample(170, OFFLINE, ""),
            make_sample(210, GAMING, "Valorant"),
            make_sample(300, GAMING, "Valorant"),
        ]
        segments = smooth(samples, 600.0, smoothing_settings)
        assert labels(segments) == ["Gaming: Valorant"]

    def test_a_genuine_switch_back_and_forth_is_preserved(self, smoothing_settings):
        samples = (
            [make_sample(t, GAMING, "Valorant") for t in (0, 100, 200, 300)]
            + [make_sample(t, REACTION, "YouTube") for t in (400, 600, 800, 1000)]
            + [make_sample(t, GAMING, "Valorant") for t in (1200, 1400, 1600)]
        )
        segments = smooth(samples, 1800.0, smoothing_settings)
        assert labels(segments) == [
            "Gaming: Valorant",
            "Reaction / Media Share: YouTube",
            "Gaming: Valorant",
        ]

    def test_a_long_interruption_survives_the_alt_tab_window(self, smoothing_settings):
        samples = [
            make_sample(0, GAMING, "Valorant"),
            make_sample(100, GAMING, "Valorant"),
            make_sample(200, INTERMISSION, ""),
            make_sample(400, INTERMISSION, ""),
            make_sample(600, GAMING, "Valorant"),
            make_sample(700, GAMING, "Valorant"),
        ]
        segments = smooth(samples, 900.0, smoothing_settings)
        assert len(segments) == 3
        assert segments[1].primary_category is INTERMISSION


class TestConfirmationHysteresis:
    def test_a_single_isolated_verdict_is_absorbed(self, smoothing_settings):
        samples = [
            make_sample(0, GAMING, "Valorant"),
            make_sample(100, GAMING, "Valorant"),
            make_sample(130, CHATTING, ""),
            make_sample(200, GAMING, "Valorant"),
        ]
        segments = smooth(samples, 400.0, smoothing_settings)
        assert len(segments) == 1

    def test_two_consecutive_verdicts_confirm_a_transition(self, smoothing_settings):
        samples = [
            make_sample(0, GAMING, "Valorant"),
            make_sample(100, GAMING, "Valorant"),
            make_sample(200, CHATTING, ""),
            make_sample(280, CHATTING, ""),
            make_sample(360, CHATTING, ""),
        ]
        segments = smooth(samples, 500.0, smoothing_settings)
        assert len(segments) == 2
        assert segments[1].primary_category is CHATTING

    def test_a_lone_verdict_covering_a_long_stretch_is_kept(self):
        """One sample is the only evidence for that stretch; erasing it invents coverage."""
        settings = SmoothingSettings(confirm_consecutive=3, min_segment_seconds=60.0)
        samples = [
            make_sample(0, GAMING, "Valorant"),
            make_sample(100, GAMING, "Valorant"),
            make_sample(200, INTERMISSION, ""),
            make_sample(900, GAMING, "Valorant"),
            make_sample(1000, GAMING, "Valorant"),
        ]
        segments = smooth(samples, 1200.0, settings)
        assert any(s.primary_category is INTERMISSION for s in segments)

    def test_two_samples_spread_past_the_floor_confirm_without_a_third(self):
        settings = SmoothingSettings(confirm_consecutive=3, min_segment_seconds=60.0)
        samples = [
            make_sample(0, GAMING, "Valorant"),
            make_sample(100, GAMING, "Valorant"),
            make_sample(200, REACTION, "YouTube"),
            make_sample(300, REACTION, "YouTube"),
            make_sample(400, GAMING, "Valorant"),
            make_sample(500, GAMING, "Valorant"),
        ]
        segments = smooth(samples, 700.0, settings)
        assert any(s.primary_category is REACTION for s in segments)

    def test_a_stricter_confirmation_count_merges_more(self):
        samples = [
            make_sample(0, GAMING, "Valorant"),
            make_sample(100, GAMING, "Valorant"),
            make_sample(200, CHATTING, ""),
            make_sample(215, CHATTING, ""),
            make_sample(260, GAMING, "Valorant"),
            make_sample(400, GAMING, "Valorant"),
        ]
        lenient = smooth(samples, 600.0, SmoothingSettings(confirm_consecutive=2))
        strict = smooth(samples, 600.0, SmoothingSettings(confirm_consecutive=4))
        assert len(strict) < len(lenient)


class TestSamplingCadence:
    def test_median_interval_of_evenly_spaced_samples(self):
        samples = [make_sample(float(t * 600)) for t in range(10)]
        assert median_sample_interval(samples) == pytest.approx(600.0)

    def test_median_interval_ignores_a_single_outlier_gap(self):
        offsets = [0, 100, 200, 300, 5000, 5100, 5200]
        samples = [make_sample(float(t)) for t in offsets]
        assert median_sample_interval(samples) == pytest.approx(100.0)

    @pytest.mark.parametrize("samples", [[], [make_sample(10.0)]])
    def test_median_interval_of_a_degenerate_series_is_zero(self, samples):
        assert median_sample_interval(samples) == 0.0

    def test_ceiling_never_drops_below_the_configured_window(self, smoothing_settings):
        assert alt_tab_ceiling(smoothing_settings, 0.0) == 90.0
        assert alt_tab_ceiling(smoothing_settings, 60.0) == 90.0

    def test_ceiling_widens_with_sparse_sampling(self, smoothing_settings):
        assert alt_tab_ceiling(smoothing_settings, 900.0) == pytest.approx(450.0)

    def test_sparse_sampling_absorbs_an_unresolvable_lone_verdict(self, smoothing_settings):
        """At a 900s cadence a lone divergent verdict cannot be distinguished from noise."""
        samples = [make_sample(float(t), GAMING, "Valorant") for t in range(60, 9000, 900)]
        samples.append(make_sample(9500.0, CHATTING, ""))
        samples += [make_sample(float(t), GAMING, "Valorant") for t in range(9660, 18000, 900)]
        segments = smooth(sorted(samples, key=lambda s: s.offset_seconds), 18000.0, smoothing_settings)
        assert labels(segments) == ["Gaming: Valorant"]

    def test_dense_sampling_keeps_the_same_verdict(self, smoothing_settings):
        """At a 60s cadence the identical gap is well resolved, so the state is real."""
        samples = [make_sample(float(t), GAMING, "Valorant") for t in range(0, 600, 60)]
        samples.append(make_sample(700.0, CHATTING, ""))
        samples += [make_sample(float(t), GAMING, "Valorant") for t in range(1000, 1600, 60)]
        segments = smooth(sorted(samples, key=lambda s: s.offset_seconds), 1800.0, smoothing_settings)
        assert any(s.primary_category is CHATTING for s in segments)


class TestSegmentGeometry:
    def test_the_timeline_covers_the_whole_vod_without_gaps(self, smoothing_settings):
        samples = (
            [make_sample(t, GAMING, "Valorant") for t in (300, 400, 500)]
            + [make_sample(t, REACTION, "YouTube") for t in (900, 1000, 1100)]
        )
        segments = smooth(samples, 1800.0, smoothing_settings)
        assert segments[0].start_seconds == 0.0
        assert segments[-1].end_seconds == pytest.approx(1800.0)
        for previous, current in zip(segments, segments[1:]):
            assert previous.end_seconds == pytest.approx(current.start_seconds)

    def test_segments_are_ordered_and_non_overlapping(self, smoothing_settings):
        samples = (
            [make_sample(t, GAMING, "Valorant") for t in (0, 200, 400)]
            + [make_sample(t, CHATTING, "") for t in (600, 800, 1000)]
            + [make_sample(t, REACTION, "YouTube") for t in (1200, 1400, 1600)]
        )
        segments = smooth(samples, 1800.0, smoothing_settings)
        for previous, current in zip(segments, segments[1:]):
            assert previous.end_seconds <= current.start_seconds

    def test_no_segment_is_shorter_than_the_floor(self, smoothing_settings):
        samples = [
            make_sample(0, GAMING, "Valorant"),
            make_sample(100, GAMING, "Valorant"),
            make_sample(200, REACTION, "YouTube"),
            make_sample(210, REACTION, "YouTube"),
            make_sample(220, CHATTING, ""),
            make_sample(230, CHATTING, ""),
            make_sample(300, GAMING, "Valorant"),
            make_sample(400, GAMING, "Valorant"),
        ]
        segments = smooth(samples, 600.0, smoothing_settings)
        assert all(s.duration_seconds >= 60.0 for s in segments)

    def test_a_transition_starts_at_the_scene_trigger(self, smoothing_settings):
        samples = (
            [make_sample(t, GAMING, "Valorant") for t in (0, 200, 400)]
            + [make_sample(t, REACTION, "YouTube") for t in (600, 700, 800)]
        )
        segments = smooth(samples, 1200.0, smoothing_settings)
        assert segments[1].start_seconds == pytest.approx(600.0)


class TestSegmentMetadata:
    def test_afk_is_set_when_the_majority_of_samples_agree(self, smoothing_settings):
        samples = [
            make_sample(0, INTERMISSION, "", afk=True),
            make_sample(200, INTERMISSION, "", afk=True),
            make_sample(400, INTERMISSION, "", afk=False),
        ]
        segments = smooth(samples, 600.0, smoothing_settings)
        assert segments[0].is_afk_or_brb

    def test_afk_is_clear_when_the_majority_disagree(self, smoothing_settings):
        samples = [
            make_sample(0, GAMING, "Valorant", afk=True),
            make_sample(200, GAMING, "Valorant", afk=False),
            make_sample(400, GAMING, "Valorant", afk=False),
        ]
        segments = smooth(samples, 600.0, smoothing_settings)
        assert not segments[0].is_afk_or_brb

    def test_confidence_is_the_mean_across_the_run(self, smoothing_settings):
        samples = [
            make_sample(0, GAMING, "Valorant", confidence=0.6),
            make_sample(200, GAMING, "Valorant", confidence=0.8),
            make_sample(400, GAMING, "Valorant", confidence=1.0),
        ]
        segments = smooth(samples, 600.0, smoothing_settings)
        assert segments[0].confidence_score == pytest.approx(0.8, abs=0.001)

    def test_sample_counts_are_preserved_across_merging(self, smoothing_settings):
        samples = [make_sample(t * 100, GAMING, "Valorant") for t in range(10)]
        segments = smooth(samples, 1200.0, smoothing_settings)
        assert sum(s.sample_count for s in segments) == 10

    def test_sub_activity_is_the_most_common_reading(self, smoothing_settings):
        from tests.conftest import make_classification
        from kick_vod_analyser.models import SampleResult

        def sample(offset, sub):
            return SampleResult(
                offset_seconds=offset,
                trigger="scene",
                classification=make_classification(GAMING, "Valorant", sub_activity=sub),
            )

        samples = [sample(0, "Main Menu"), sample(200, "In-Game Match"), sample(400, "In-Game Match")]
        segments = smooth(samples, 600.0, smoothing_settings)
        assert segments[0].sub_activity == "In-Game Match"


class TestEdgeCases:
    def test_no_samples_yields_no_segments(self, smoothing_settings):
        assert smooth([], 3600.0, smoothing_settings) == []

    def test_a_single_sample_covers_the_whole_vod(self, smoothing_settings):
        segments = smooth([make_sample(500, GAMING, "Valorant")], 3600.0, smoothing_settings)
        assert len(segments) == 1
        assert segments[0].start_seconds == 0.0
        assert segments[0].end_seconds == pytest.approx(3600.0)

    def test_unordered_input_is_sorted_before_smoothing(self, smoothing_settings):
        samples = [
            make_sample(900, REACTION, "YouTube"),
            make_sample(0, GAMING, "Valorant"),
            make_sample(1000, REACTION, "YouTube"),
            make_sample(300, GAMING, "Valorant"),
            make_sample(1100, REACTION, "YouTube"),
            make_sample(600, GAMING, "Valorant"),
        ]
        segments = smooth(samples, 1500.0, smoothing_settings)
        assert labels(segments) == ["Gaming: Valorant", "Reaction / Media Share: YouTube"]

    def test_alternating_noise_does_not_fragment_the_timeline(self, smoothing_settings):
        samples = []
        for index in range(40):
            category = GAMING if index % 2 == 0 else CHATTING
            samples.append(make_sample(index * 30.0, category, "Valorant" if index % 2 == 0 else ""))
        segments = smooth(samples, 1200.0, smoothing_settings)
        assert len(segments) <= 3

    def test_a_ten_hour_multi_game_session_reconstructs_correctly(self, smoothing_settings):
        """The plan's headline case: ten hours, ten switches, plus alt-tab noise."""
        plan = [
            (0, 7200, GAMING, "Valorant"),
            (7200, 12600, REACTION, "YouTube"),
            (12600, 19800, GAMING, "Grand Theft Auto V"),
            (19800, 21600, INTERMISSION, ""),
            (21600, 28800, GAMING, "Counter-Strike 2"),
            (28800, 32400, CHATTING, ""),
            (32400, 36000, GAMING, "Valorant"),
        ]
        samples = []
        for start, end, category, title in plan:
            for offset in range(int(start) + 60, int(end), 600):
                samples.append(make_sample(float(offset), category, title))
        for offset in (3000.0, 15000.0, 25000.0):
            samples.append(make_sample(offset + 30, CHATTING, ""))

        segments = smooth(sorted(samples, key=lambda s: s.offset_seconds), 36000.0, smoothing_settings)

        assert labels(segments) == [
            "Gaming: Valorant",
            "Reaction / Media Share: YouTube",
            "Gaming: Grand Theft Auto V",
            "Intermission / AFK / BRB",
            "Gaming: Counter-Strike 2",
            "Just Chatting / Podcast",
            "Gaming: Valorant",
        ]
        for segment, (start, end, _, _) in zip(segments, plan):
            assert abs(segment.start_seconds - start) < 700
            assert abs(segment.end_seconds - end) < 700
