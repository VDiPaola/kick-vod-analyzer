from __future__ import annotations

import pytest
from pydantic import ValidationError

from kick_vod_analyser.models import (
    ChatMessage,
    ChatWindow,
    Classification,
    PrimaryCategory,
    SamplePoint,
    Segment,
    VodInfo,
)


class TestClassification:
    def test_accepts_full_schema_payload(self):
        result = Classification.model_validate(
            {
                "primary_category": "Gaming",
                "specific_title_or_context": "Valorant",
                "sub_activity": "In-Game Match",
                "is_streamer_on_screen": True,
                "is_afk_or_brb": False,
                "confidence_score": 0.92,
                "visual_evidence": "Valorant HUD visible in all four cells.",
            }
        )
        assert result.primary_category is PrimaryCategory.GAMING
        assert result.confidence_score == pytest.approx(0.92)

    @pytest.mark.parametrize("raw,expected", [(1.5, 1.0), (-0.2, 0.0), (0.5, 0.5)])
    def test_clamps_confidence(self, raw, expected):
        result = Classification(primary_category="Gaming", confidence_score=raw)
        assert result.confidence_score == expected

    def test_rejects_unknown_category(self):
        with pytest.raises(ValidationError):
            Classification(primary_category="Speedrunning", confidence_score=0.5)

    def test_every_plan_category_is_modelled(self):
        expected = {
            "Gaming",
            "Just Chatting / Podcast",
            "IRL / Outdoors",
            "Reaction / Media Share",
            "Coding / Creative",
            "Gambling / Slots",
            "Intermission / AFK / BRB",
            "Technical Difficulties / Offline",
        }
        assert {c.value for c in PrimaryCategory} == expected


class TestSamplePoint:
    def test_custom_id_rounds_to_whole_seconds(self):
        assert SamplePoint(offset_seconds=120.6, trigger="scene").custom_id == "t121"
        assert SamplePoint(offset_seconds=0.4, trigger="heartbeat").custom_id == "t0"

    def test_rejects_negative_offset(self):
        with pytest.raises(ValidationError):
            SamplePoint(offset_seconds=-1.0, trigger="scene")

    def test_rejects_unknown_trigger(self):
        with pytest.raises(ValidationError):
            SamplePoint(offset_seconds=1.0, trigger="magic")


class TestSegment:
    def test_duration_never_goes_negative(self):
        segment = Segment(start_seconds=100.0, end_seconds=50.0, primary_category="Gaming")
        assert segment.duration_seconds == 0.0

    def test_label_includes_title_when_specific(self):
        segment = Segment(
            start_seconds=0.0,
            end_seconds=10.0,
            primary_category="Gaming",
            specific_title_or_context="Valorant",
        )
        assert segment.label == "Gaming: Valorant"

    @pytest.mark.parametrize("title", ["", "  ", "unknown", "N/A", "none"])
    def test_label_drops_placeholder_titles(self, title):
        segment = Segment(
            start_seconds=0.0,
            end_seconds=10.0,
            primary_category="Just Chatting / Podcast",
            specific_title_or_context=title,
        )
        assert segment.label == "Just Chatting / Podcast"


class TestChatWindow:
    def test_empty_window_renders_a_placeholder(self):
        assert "no chat activity" in ChatWindow(offset_seconds=0.0).render()

    def test_render_includes_counts_and_lines(self):
        window = ChatWindow(
            offset_seconds=10.0,
            lines=["[00:00:10] what game is this"],
            message_count=4,
            unique_chatters=3,
        )
        rendered = window.render()
        assert "4 messages from 3 chatters" in rendered
        assert "what game is this" in rendered


class TestVodInfo:
    def test_duration_hours(self, vod):
        assert vod.duration_hours == pytest.approx(1.0)

    def test_rejects_zero_duration(self):
        with pytest.raises(ValidationError):
            VodInfo(vod_id="x", url="u", channel_slug="c", duration_seconds=0)

    def test_is_immutable(self, vod):
        with pytest.raises(ValidationError):
            vod.channel_slug = "other"


class TestChatMessage:
    def test_strips_surrounding_whitespace(self):
        assert ChatMessage(offset_seconds=1, username="a", text="  hi  ").text == "hi"
