"""Backoff behaviour for rate-limited and transient provider errors."""

from __future__ import annotations

import json

import pytest

from kick_vod_analyser.classify.base import ClassificationRequest
from kick_vod_analyser.classify.gemini import GeminiClassifier
from kick_vod_analyser.classify.openai_provider import OpenAIClassifier
from kick_vod_analyser.classify.retry import call_with_retry, is_retryable, suggested_delay

VALID = {
    "primary_category": "Gaming",
    "specific_title_or_context": "Valorant",
    "sub_activity": "In-Game Match",
    "is_streamer_on_screen": True,
    "is_afk_or_brb": False,
    "confidence_score": 0.9,
    "visual_evidence": "HUD visible.",
}


class RateLimited(Exception):
    def __init__(self, message="429 RESOURCE_EXHAUSTED", code=429):
        super().__init__(message)
        self.code = code


class Forbidden(Exception):
    status_code = 403


class TestClassification:
    def test_status_codes(self):
        assert is_retryable(RateLimited())
        assert is_retryable(RateLimited("boom", code=503))
        assert not is_retryable(Forbidden())

    def test_message_markers_without_code(self):
        assert is_retryable(RuntimeError("Rate limit exceeded"))
        assert is_retryable(RuntimeError("You exceeded your current quota"))
        assert not is_retryable(RuntimeError("invalid api key"))

    def test_gemini_retry_delay_in_body(self):
        exc = RateLimited(
            "429 RESOURCE_EXHAUSTED. {'error': {'details': [{'retryDelay': '37s'}]}}"
        )
        assert suggested_delay(exc) == 37.0

    def test_openai_retry_in_text(self):
        assert suggested_delay(RuntimeError("Please try again in 1.5s.")) is None
        assert suggested_delay(RuntimeError("Rate limit reached. Retry after 800ms")) == 0.8
        assert suggested_delay(RuntimeError("retry in 2 minutes")) == 120.0

    def test_retry_after_header(self):
        exc = RateLimited()
        exc.response = type("R", (), {"status_code": 429, "headers": {"retry-after": "12"}})()
        assert suggested_delay(exc) == 12.0


class TestCallWithRetry:
    def test_returns_after_transient_failures(self):
        calls = []
        sleeps = []

        def fn():
            calls.append(1)
            if len(calls) < 3:
                raise RateLimited()
            return "ok"

        assert call_with_retry(fn, label="x", sleep=sleeps.append) == "ok"
        assert len(calls) == 3
        assert len(sleeps) == 2
        assert sleeps[0] < sleeps[1]

    def test_uses_provider_delay(self):
        sleeps = []
        calls = []

        def fn():
            calls.append(1)
            if len(calls) == 1:
                raise RateLimited("quota, retryDelay: 40s")
            return 1

        call_with_retry(fn, label="x", sleep=sleeps.append)
        assert 40.0 <= sleeps[0] <= 41.0

    def test_caps_delay(self):
        sleeps = []
        calls = []

        def fn():
            calls.append(1)
            if len(calls) == 1:
                raise RateLimited("retryDelay: 9999s")
            return 1

        call_with_retry(fn, label="x", max_delay=60, sleep=sleeps.append)
        assert sleeps[0] <= 61.0

    def test_non_retryable_raises_immediately(self):
        sleeps = []
        with pytest.raises(Forbidden):
            call_with_retry(lambda: (_ for _ in ()).throw(Forbidden()), label="x", sleep=sleeps.append)
        assert sleeps == []

    def test_gives_up_after_attempts(self):
        sleeps = []
        calls = []

        def fn():
            calls.append(1)
            raise RateLimited()

        with pytest.raises(RateLimited):
            call_with_retry(fn, label="x", attempts=3, sleep=sleeps.append)
        assert len(calls) == 3
        assert len(sleeps) == 2


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr("kick_vod_analyser.classify.retry.time.sleep", lambda s: None)


@pytest.fixture
def request_(solid_frame):
    return ClassificationRequest(
        custom_id="t0",
        offset_seconds=0.0,
        image_path=solid_frame((10, 20, 30), "grid0.jpg"),
        user_prompt="classify",
    )


def test_gemini_recovers_from_rate_limit(monkeypatch, request_):
    classifier = GeminiClassifier("gemini-2.5-flash-lite", api_key="fake", max_workers=1)
    calls = []

    class Response:
        text = json.dumps(VALID)
        usage_metadata = None

    class Models:
        def generate_content(self, **kwargs):
            calls.append(1)
            if len(calls) < 3:
                raise RateLimited()
            return Response()

    classifier._client = type("C", (), {"models": Models()})()
    [response] = classifier.classify([request_])
    assert response.classification is not None
    assert len(calls) == 3


def test_openai_recovers_from_rate_limit(monkeypatch, request_):
    classifier = OpenAIClassifier("gpt-4o-mini", api_key="fake", max_workers=1)
    calls = []

    class Message:
        content = json.dumps(VALID)

    class Choice:
        message = Message()

    class Completion:
        choices = [Choice()]
        usage = None

    class Completions:
        def create(self, **kwargs):
            calls.append(1)
            if len(calls) < 2:
                raise RateLimited("Rate limit reached", code=429)
            return Completion()

    chat = type("Chat", (), {"completions": Completions()})()
    classifier._client = type("C", (), {"chat": chat})()
    [response] = classifier.classify([request_])
    assert response.classification is not None
    assert len(calls) == 2


def test_gemini_exhausted_retries_become_error_response(monkeypatch, request_):
    classifier = GeminiClassifier("gemini-2.5-flash-lite", api_key="fake", max_workers=1, max_attempts=2)

    class Models:
        def generate_content(self, **kwargs):
            raise RateLimited()

    classifier._client = type("C", (), {"models": Models()})()
    [response] = classifier.classify([request_])
    assert response.classification is None
    assert "429" in response.error
