from __future__ import annotations

import json

import pytest

from kick_vod_analyser.classify.base import (
    ClassificationRequest,
    ClassificationResponse,
    parse_classification,
)
from kick_vod_analyser.classify.factory import MockClassifier, build_classifier
from kick_vod_analyser.classify.gemini import (
    extract_text_from_record,
    parse_batch_results as parse_gemini_results,
)
from kick_vod_analyser.classify.openai_provider import parse_batch_results as parse_openai_results
from kick_vod_analyser.classify.pricing import PRICES, estimate_cost, price_for
from kick_vod_analyser.classify.prompts import (
    CLASSIFICATION_SCHEMA,
    SYSTEM_PROMPT,
    build_user_prompt,
    gemini_response_schema,
    openai_response_format,
)
from kick_vod_analyser.models import ChatWindow, PrimaryCategory

VALID = {
    "primary_category": "Gaming",
    "specific_title_or_context": "Valorant",
    "sub_activity": "In-Game Match",
    "is_streamer_on_screen": True,
    "is_afk_or_brb": False,
    "confidence_score": 0.9,
    "visual_evidence": "Valorant HUD is visible.",
}


class TestParseClassification:
    def test_parses_bare_json(self):
        result, error = parse_classification(json.dumps(VALID))
        assert error is None
        assert result.primary_category is PrimaryCategory.GAMING

    def test_parses_a_markdown_fenced_block(self):
        result, error = parse_classification(f"```json\n{json.dumps(VALID)}\n```")
        assert error is None and result is not None

    def test_recovers_json_from_a_preamble(self):
        result, error = parse_classification(f"Here is the answer:\n{json.dumps(VALID)}")
        assert error is None and result is not None

    @pytest.mark.parametrize("text", [None, "", "   "])
    def test_empty_output_is_reported(self, text):
        result, error = parse_classification(text)
        assert result is None and "empty" in error

    def test_prose_without_json_is_reported(self):
        result, error = parse_classification("I cannot determine the activity.")
        assert result is None and "no JSON object" in error

    def test_malformed_json_is_reported(self):
        result, error = parse_classification('{"primary_category": "Gaming",}')
        assert result is None and error

    def test_a_wrong_enum_value_fails_validation(self):
        payload = dict(VALID, primary_category="Speedrunning")
        result, error = parse_classification(json.dumps(payload))
        assert result is None and "schema validation" in error

    def test_a_missing_optional_field_still_parses(self):
        payload = {"primary_category": "Gaming", "confidence_score": 0.4}
        result, error = parse_classification(json.dumps(payload))
        assert error is None and result.specific_title_or_context == ""


class TestPrompts:
    def test_system_prompt_explains_the_grid_layout(self):
        assert "2x2" in SYSTEM_PROMPT or "cell 1" in SYSTEM_PROMPT
        assert "T-6s" in SYSTEM_PROMPT and "T+6s" in SYSTEM_PROMPT

    def test_system_prompt_covers_the_documented_edge_cases(self):
        lowered = SYSTEM_PROMPT.lower()
        assert "reaction / media share" in lowered
        assert "loading screens" in lowered
        assert "brb" in lowered

    def test_schema_requires_every_documented_field(self):
        assert set(CLASSIFICATION_SCHEMA["required"]) == {
            "primary_category",
            "specific_title_or_context",
            "sub_activity",
            "is_streamer_on_screen",
            "is_afk_or_brb",
            "confidence_score",
            "visual_evidence",
        }

    def test_schema_enum_matches_the_model_enum(self):
        enum = CLASSIFICATION_SCHEMA["properties"]["primary_category"]["enum"]
        assert set(enum) == {c.value for c in PrimaryCategory}

    def test_confidence_is_bounded_in_the_schema(self):
        field = CLASSIFICATION_SCHEMA["properties"]["confidence_score"]
        assert field["minimum"] == 0.0 and field["maximum"] == 1.0

    def test_gemini_schema_drops_additional_properties(self):
        assert "additionalProperties" not in gemini_response_schema()

    def test_openai_response_format_is_strict(self):
        wrapper = openai_response_format()
        assert wrapper["json_schema"]["strict"] is True
        assert wrapper["json_schema"]["schema"]["additionalProperties"] is False

    def test_user_prompt_includes_the_timestamp_and_channel(self):
        prompt = build_user_prompt(3661.0, None, channel_slug="teststreamer")
        assert "01:01:01" in prompt and "teststreamer" in prompt

    def test_user_prompt_states_when_chat_is_unavailable(self):
        assert "unavailable" in build_user_prompt(100.0, None)

    def test_user_prompt_embeds_the_chat_excerpt(self):
        window = ChatWindow(
            offset_seconds=100.0,
            lines=["[00:01:40] what game is this"],
            message_count=1,
            unique_chatters=1,
        )
        prompt = build_user_prompt(100.0, window)
        assert "what game is this" in prompt

    def test_a_long_stream_title_is_truncated(self):
        prompt = build_user_prompt(0.0, None, stream_title="x" * 500)
        assert "x" * 201 not in prompt


class TestPricing:
    def test_known_models_use_the_table(self):
        assert price_for("gpt-4o-mini") is PRICES["gpt-4o-mini"]

    def test_a_dated_suffix_matches_by_prefix(self):
        assert price_for("gemini-2.5-flash-lite-preview-09-2025") is PRICES["gemini-2.5-flash-lite"]

    def test_the_longest_prefix_wins(self):
        assert price_for("gemini-2.5-flash-lite") is PRICES["gemini-2.5-flash-lite"]

    def test_an_unknown_model_falls_back_without_raising(self):
        assert price_for("some-future-model").input_per_million > 0

    def test_batch_mode_halves_the_cost(self):
        sync = estimate_cost(100, "gemini-2.5-flash-lite", batch=False)
        batch = estimate_cost(100, "gemini-2.5-flash-lite", batch=True)
        assert batch["total_cost_usd"] == pytest.approx(sync["total_cost_usd"] / 2)

    def test_dropping_chat_lowers_the_input_tokens(self):
        with_chat = estimate_cost(100, "gpt-4o-mini", with_chat=True)
        without = estimate_cost(100, "gpt-4o-mini", with_chat=False)
        assert without["input_tokens"] < with_chat["input_tokens"]

    def test_measured_usage_overrides_the_estimate(self):
        cost = estimate_cost(100, "gpt-4o-mini", input_tokens=1_000_000, output_tokens=0)
        assert cost["input_cost_usd"] == pytest.approx(0.15)
        assert cost["output_cost_usd"] == 0.0

    def test_a_ten_hour_vod_stays_under_five_cents_on_the_default_model(self):
        """The plan's headline cost target, checked against the live price table."""
        from kick_vod_analyser.config import Settings

        default_model = Settings.model_fields["gemini_model"].default
        cost = estimate_cost(100, default_model, batch=True)
        assert cost["total_cost_usd"] < 0.05

    def test_the_default_model_has_an_explicit_price(self):
        """A default that fell through to DEFAULT_PRICE would misreport every estimate."""
        from kick_vod_analyser.classify.pricing import DEFAULT_PRICE
        from kick_vod_analyser.config import Settings

        for field in ("gemini_model", "openai_model"):
            model = Settings.model_fields[field].default
            assert price_for(model) is not DEFAULT_PRICE, model

    def test_zero_requests_costs_nothing(self):
        assert estimate_cost(0, "gpt-4o-mini")["total_cost_usd"] == 0.0


class TestMockClassifier:
    def _request(self, path, custom_id="t100"):
        return ClassificationRequest(
            custom_id=custom_id,
            offset_seconds=100.0,
            image_path=path,
            user_prompt="classify this",
        )

    def test_returns_one_response_per_request(self, solid_frame):
        requests = [
            self._request(solid_frame((i * 40, 10, 10), f"f{i}.jpg"), f"t{i}") for i in range(5)
        ]
        responses = MockClassifier().classify(requests)
        assert [r.custom_id for r in responses] == [r.custom_id for r in requests]

    def test_is_deterministic_for_identical_input(self, solid_frame):
        request = self._request(solid_frame((77, 88, 99), "f.jpg"))
        first = MockClassifier().classify([request])[0]
        second = MockClassifier().classify([request])[0]
        assert first.classification == second.classification

    def test_different_images_can_yield_different_categories(self, solid_frame):
        requests = [
            self._request(solid_frame((i * 17 % 255, i * 31 % 255, i * 7 % 255), f"f{i}.jpg"), f"t{i}")
            for i in range(24)
        ]
        categories = {
            r.classification.primary_category for r in MockClassifier().classify(requests)
        }
        assert len(categories) > 1

    def test_every_response_validates(self, solid_frame):
        requests = [self._request(solid_frame((i, i, i), f"f{i}.jpg"), f"t{i}") for i in range(8)]
        for response in MockClassifier().classify(requests):
            assert response.classification is not None
            assert response.error is None

    def test_intermission_verdicts_set_the_afk_flag(self, solid_frame):
        requests = [self._request(solid_frame((i, 200, i), f"f{i}.jpg"), f"t{i}") for i in range(30)]
        for response in MockClassifier().classify(requests):
            classification = response.classification
            if classification.primary_category is PrimaryCategory.INTERMISSION:
                assert classification.is_afk_or_brb

    def test_empty_input_is_safe(self):
        assert MockClassifier().classify([]) == []

    def test_batch_is_unsupported(self):
        assert not MockClassifier().supports_batch()
        with pytest.raises(NotImplementedError):
            MockClassifier().submit_batch([], None)


class TestBuildClassifier:
    def test_mock_needs_no_credentials(self, settings):
        assert isinstance(build_classifier("mock", settings), MockClassifier)

    def test_gemini_without_a_key_is_rejected(self, settings):
        settings.gemini_api_key = None
        with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
            build_classifier("gemini", settings)

    def test_openai_without_a_key_is_rejected(self, settings):
        settings.openai_api_key = None
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            build_classifier("openai", settings)

    def test_an_unknown_provider_is_rejected(self, settings):
        with pytest.raises(ValueError, match="unknown provider"):
            build_classifier("telepathy", settings)


class TestGeminiBatchParsing:
    def _line(self, key, text):
        return json.dumps(
            {
                "key": key,
                "response": {"candidates": [{"content": {"parts": [{"text": text}]}}]},
            }
        )

    def test_parses_successful_results(self, tmp_path):
        path = tmp_path / "results.jsonl"
        path.write_text(
            "\n".join([self._line("t100", json.dumps(VALID)), self._line("t200", json.dumps(VALID))]),
            encoding="utf-8",
        )
        responses = parse_gemini_results(path)
        assert [r.custom_id for r in responses] == ["t100", "t200"]
        assert all(r.classification is not None for r in responses)

    def test_records_per_request_errors(self, tmp_path):
        path = tmp_path / "results.jsonl"
        path.write_text(
            json.dumps({"key": "t100", "error": {"code": 429, "message": "rate limited"}}),
            encoding="utf-8",
        )
        response = parse_gemini_results(path)[0]
        assert response.classification is None and "429" in response.error

    def test_a_corrupt_line_does_not_abort_the_file(self, tmp_path):
        path = tmp_path / "results.jsonl"
        path.write_text("{not json\n" + self._line("t200", json.dumps(VALID)), encoding="utf-8")
        responses = parse_gemini_results(path)
        assert len(responses) == 2
        assert responses[1].classification is not None

    def test_blank_lines_are_ignored(self, tmp_path):
        path = tmp_path / "results.jsonl"
        path.write_text("\n\n" + self._line("t100", json.dumps(VALID)) + "\n\n", encoding="utf-8")
        assert len(parse_gemini_results(path)) == 1

    def test_extracts_text_from_a_raw_response_dict(self):
        payload = {"candidates": [{"content": {"parts": [{"text": "hello"}]}}]}
        assert extract_text_from_record(payload) == "hello"

    def test_extraction_tolerates_missing_parts(self):
        assert extract_text_from_record({"candidates": [{}]}) is None


class TestOpenAIBatchParsing:
    def _line(self, custom_id, content):
        return json.dumps(
            {
                "custom_id": custom_id,
                "response": {"body": {"choices": [{"message": {"content": content}}]}},
            }
        )

    def test_parses_successful_results(self, tmp_path):
        path = tmp_path / "results.jsonl"
        path.write_text(self._line("t100", json.dumps(VALID)), encoding="utf-8")
        response = parse_openai_results(path)[0]
        assert response.custom_id == "t100"
        assert response.classification.primary_category is PrimaryCategory.GAMING

    def test_records_per_request_errors(self, tmp_path):
        path = tmp_path / "results.jsonl"
        path.write_text(
            json.dumps({"custom_id": "t100", "error": {"message": "context length"}}),
            encoding="utf-8",
        )
        assert "context length" in parse_openai_results(path)[0].error

    def test_a_missing_body_is_reported_not_raised(self, tmp_path):
        path = tmp_path / "results.jsonl"
        path.write_text(json.dumps({"custom_id": "t100", "response": {}}), encoding="utf-8")
        response = parse_openai_results(path)[0]
        assert response.classification is None and response.error


class TestClassificationResponse:
    def test_defaults_to_no_classification(self):
        response = ClassificationResponse(custom_id="t1")
        assert response.classification is None and response.error is None
