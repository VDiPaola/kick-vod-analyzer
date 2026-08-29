from __future__ import annotations

import pytest

from kick_vod_analyser.chatwindow.slicer import (
    build_window,
    canonical_form,
    condense,
    format_offset,
    is_noise,
    relevance_score,
)
from kick_vod_analyser.config import ChatSettings
from kick_vod_analyser.models import ChatMessage


def msg(text: str, offset: float = 100.0, username: str = "viewer") -> ChatMessage:
    return ChatMessage(offset_seconds=offset, username=username, text=text)


class TestCanonicalForm:
    def test_case_and_punctuation_are_ignored(self):
        assert canonical_form("KEKW!!!") == canonical_form("kekw")

    def test_repeated_words_collapse(self):
        assert canonical_form("LUL LUL LUL LUL") == "lul"

    def test_urls_become_a_placeholder(self):
        assert canonical_form("check https://x.com/abc") == "check url"

    def test_whitespace_is_normalised(self):
        assert canonical_form("what   game    is  this") == "what game is this"

    def test_emoji_only_message_is_empty(self):
        assert canonical_form("!!!???") == ""


class TestIsNoise:
    @pytest.mark.parametrize("text", ["!discord", "!sens", "$tip", "#play"])
    def test_bot_commands_are_dropped(self, text):
        assert is_noise(msg(text), ChatSettings())

    def test_bot_commands_can_be_retained(self):
        assert not is_noise(msg("!sens"), ChatSettings(drop_bot_commands=False))

    @pytest.mark.parametrize("name", ["Botrix", "streamelements", "NightBot"])
    def test_known_bots_are_dropped(self, name):
        assert is_noise(msg("Follow the socials", username=name), ChatSettings())

    @pytest.mark.parametrize("text", ["lol", "W", "ez", "gg", "xd"])
    def test_content_free_single_tokens_are_dropped(self, text):
        assert is_noise(msg(text), ChatSettings())

    def test_short_messages_below_the_floor_are_dropped(self):
        assert is_noise(msg("a"), ChatSettings(min_message_length=2))

    def test_real_content_survives(self):
        assert not is_noise(msg("what game is this"), ChatSettings())

    def test_a_low_signal_token_with_an_emote_survives(self):
        message = ChatMessage(offset_seconds=1, username="v", text="lol", emotes=("KEKW",))
        assert not is_noise(message, ChatSettings())


class TestRelevanceScore:
    def test_signal_phrases_outrank_spam(self):
        assert relevance_score("what game is this", 1) > relevance_score("KEKW", 40)

    def test_repetition_raises_the_score(self):
        assert relevance_score("KEKW", 40) > relevance_score("KEKW", 1)

    def test_repetition_scales_logarithmically(self):
        """Equal ratios add equal amounts, not proportional ones."""
        first = relevance_score("KEKW", 5) - relevance_score("KEKW", 1)
        second = relevance_score("KEKW", 25) - relevance_score("KEKW", 5)
        assert first == pytest.approx(second, abs=0.01)

    def test_repetition_bonus_is_capped(self):
        assert relevance_score("KEKW", 10_000) == relevance_score("KEKW", 1_000_000)

    def test_extreme_spam_never_outranks_a_signal_phrase(self):
        assert relevance_score("KEKW", 10_000) < relevance_score("what game is this", 1)

    def test_questions_score_above_statements(self):
        assert relevance_score("is he done?", 1) > relevance_score("is he done", 1)

    def test_walls_of_text_are_penalised(self):
        assert relevance_score("word " * 60, 1) < relevance_score("a normal chat line", 1)


class TestCondense:
    def test_repeats_collapse_with_a_multiplier(self):
        messages = [msg("KEKW", 100 + i, f"u{i}") for i in range(45)]
        lines = condense(messages, ChatSettings())
        assert len(lines) == 1
        assert "(x45)" in lines[0]

    def test_output_respects_the_line_cap(self):
        messages = [msg(f"unique message number {i}", 100 + i, f"u{i}") for i in range(80)]
        lines = condense(messages, ChatSettings(max_lines=30))
        assert len(lines) == 30

    def test_high_signal_lines_survive_the_cap(self):
        messages = [msg(f"filler line {i}", 100 + i, f"u{i}") for i in range(60)]
        messages.append(msg("what game is this", 130, "asker"))
        lines = condense(messages, ChatSettings(max_lines=5))
        assert any("what game is this" in line for line in lines)

    def test_lines_stay_in_chronological_order(self):
        messages = [msg(f"message {i}", 100 + i * 2, f"u{i}") for i in range(10)]
        lines = condense(messages, ChatSettings())
        stamps = [line.split("]")[0] for line in lines]
        assert stamps == sorted(stamps)

    def test_each_line_carries_a_timestamp(self):
        lines = condense([msg("hello there friend", 3661.0)], ChatSettings())
        assert lines[0].startswith("[01:01:01]")

    def test_long_messages_are_truncated(self):
        lines = condense([msg("x" * 400)], ChatSettings())
        assert len(lines[0]) < 200 and lines[0].endswith("...")

    def test_the_longest_variant_of_a_repeat_is_kept(self):
        messages = [msg("KEKW", 100), msg("KEKW KEKW", 101), msg("kekw", 102)]
        lines = condense(messages, ChatSettings())
        assert len(lines) == 1

    def test_an_all_noise_window_condenses_to_nothing(self):
        messages = [msg("!discord", 100), msg("lol", 101), msg("W", 102)]
        assert condense(messages, ChatSettings()) == []

    def test_empty_input_is_safe(self):
        assert condense([], ChatSettings()) == []

    def test_a_large_window_stays_inside_the_token_budget(self):
        """A 45 second slice of a busy chat must not blow the prompt budget."""
        messages = []
        for i in range(2000):
            text = "KEKW" if i % 3 == 0 else f"random chat line {i % 120}"
            messages.append(msg(text, 100 + i * 0.02, f"u{i}"))
        lines = condense(messages, ChatSettings())
        rendered = "\n".join(lines)
        assert len(lines) <= 30
        assert len(rendered) < 6000


class TestBuildWindow:
    def test_reports_message_and_chatter_counts(self):
        messages = [msg("hello there", 100, "a"), msg("hello there", 101, "b"), msg("hi", 102, "a")]
        window = build_window(messages, 100.0, ChatSettings())
        assert window.message_count == 3
        assert window.unique_chatters == 2

    def test_an_empty_window_renders_the_placeholder(self):
        window = build_window([], 100.0, ChatSettings())
        assert window.is_empty
        assert "no chat activity" in window.render()

    def test_rendered_block_includes_the_counts_header(self):
        window = build_window([msg("what game is this", 100)], 100.0, ChatSettings())
        assert "1 messages from 1 chatters" in window.render()


class TestFormatOffset:
    @pytest.mark.parametrize(
        "seconds,expected",
        [(0, "00:00:00"), (61, "00:01:01"), (3661, "01:01:01"), (36000, "10:00:00")],
    )
    def test_formats_as_hours_minutes_seconds(self, seconds, expected):
        assert format_offset(seconds) == expected
