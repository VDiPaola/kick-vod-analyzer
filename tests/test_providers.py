"""Provider tests driven by fake SDK clients. No network is touched."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kick_vod_analyser.classify.base import ClassificationRequest
from kick_vod_analyser.classify.gemini import GeminiClassifier
from kick_vod_analyser.classify.openai_provider import OpenAIClassifier
from kick_vod_analyser.ingest.http import BROWSER_HEADERS, build_client, impersonation_available

VALID = {
    "primary_category": "Gaming",
    "specific_title_or_context": "Valorant",
    "sub_activity": "In-Game Match",
    "is_streamer_on_screen": True,
    "is_afk_or_brb": False,
    "confidence_score": 0.9,
    "visual_evidence": "HUD visible.",
}


@pytest.fixture
def requests(solid_frame):
    return [
        ClassificationRequest(
            custom_id=f"t{index}",
            offset_seconds=float(index * 100),
            image_path=solid_frame((index * 40 % 255, 20, 20), f"grid{index}.jpg"),
            user_prompt=f"classify offset {index}",
        )
        for index in range(3)
    ]


class Recorder:
    def __init__(self):
        self.calls = []


class TestGeminiSync:
    def _classifier(self, monkeypatch, responder):
        classifier = GeminiClassifier("gemini-2.5-flash-lite", api_key="fake-key", max_workers=2)

        class Models:
            def generate_content(self, **kwargs):
                return responder(kwargs)

        class Client:
            models = Models()

        classifier._client = Client()
        return classifier

    def test_maps_responses_back_to_custom_ids(self, monkeypatch, requests):
        class Response:
            text = json.dumps(VALID)
            usage_metadata = type("U", (), {"prompt_token_count": 700, "candidates_token_count": 80})()

        classifier = self._classifier(monkeypatch, lambda kwargs: Response())
        responses = classifier.classify(requests)

        assert [r.custom_id for r in responses] == ["t0", "t1", "t2"]
        assert all(r.classification is not None for r in responses)

    def test_accumulates_token_usage(self, monkeypatch, requests):
        class Response:
            text = json.dumps(VALID)
            usage_metadata = type("U", (), {"prompt_token_count": 700, "candidates_token_count": 80})()

        classifier = self._classifier(monkeypatch, lambda kwargs: Response())
        classifier.classify(requests)

        assert classifier.usage.input_tokens == 2100
        assert classifier.usage.output_tokens == 240

    def test_sends_the_image_and_the_prompt(self, monkeypatch, requests):
        recorder = Recorder()

        class Response:
            text = json.dumps(VALID)
            usage_metadata = None

        def responder(kwargs):
            recorder.calls.append(kwargs)
            return Response()

        self._classifier(monkeypatch, responder).classify(requests[:1])

        contents = recorder.calls[0]["contents"]
        assert len(contents) == 2
        assert contents[0].inline_data is not None
        assert "classify offset 0" in contents[1].text

    def test_the_schema_is_attached_to_the_request(self, monkeypatch, requests):
        recorder = Recorder()

        class Response:
            text = json.dumps(VALID)
            usage_metadata = None

        def responder(kwargs):
            recorder.calls.append(kwargs)
            return Response()

        self._classifier(monkeypatch, responder).classify(requests[:1])

        config = recorder.calls[0]["config"]
        assert config.response_mime_type == "application/json"
        assert config.response_json_schema is not None

    def test_a_request_failure_is_isolated(self, monkeypatch, requests):
        class Response:
            text = json.dumps(VALID)
            usage_metadata = None

        def responder(kwargs):
            if "offset 1" in kwargs["contents"][1].text:
                raise RuntimeError("rate limited")
            return Response()

        responses = self._classifier(monkeypatch, responder).classify(requests)

        assert responses[1].classification is None
        assert "rate limited" in responses[1].error
        assert responses[0].classification is not None
        assert responses[2].classification is not None

    def test_empty_input_makes_no_calls(self, monkeypatch):
        recorder = Recorder()
        classifier = self._classifier(monkeypatch, lambda k: recorder.calls.append(k))
        assert classifier.classify([]) == []
        assert recorder.calls == []


class FakeGeminiClient:
    def __init__(self, *, dest=None, state="JOB_STATE_SUCCEEDED"):
        self.uploaded = []
        self.created = []
        self.downloads = []
        self._dest = dest
        self._state = state
        outer = self

        class Files:
            def upload(self, *, file, config=None):
                outer.uploaded.append((file, config or {}))
                name = Path(file).name
                return type(
                    "File",
                    (),
                    {
                        "name": f"files/{name}",
                        "uri": f"https://files/{name}",
                        "mime_type": (config or {}).get("mime_type"),
                    },
                )()

            def download(self, *, file):
                outer.downloads.append(file)
                return outer._payload

        class Batches:
            def create(self, *, model, src, config=None):
                outer.created.append({"model": model, "src": src, "config": config})
                return type("Job", (), {"name": "batches/job-1"})()

            def get(self, *, name):
                return type(
                    "Job",
                    (),
                    {
                        "name": name,
                        "state": type("S", (), {"name": outer._state})(),
                        "dest": outer._dest,
                    },
                )()

        self.files = Files()
        self.batches = Batches()
        self._payload = b""


class TestGeminiBatch:
    def _classifier(self, client):
        classifier = GeminiClassifier("gemini-2.5-flash-lite", api_key="fake-key")
        classifier._client = client
        return classifier

    def test_supports_batch(self):
        assert self._classifier(FakeGeminiClient()).supports_batch()

    def test_uploads_every_grid_before_creating_the_job(self, requests, tmp_path):
        client = FakeGeminiClient()
        job_id = self._classifier(client).submit_batch(requests, tmp_path)

        grid_uploads = [u for u in client.uploaded if u[1].get("mime_type") == "image/jpeg"]
        assert len(grid_uploads) == 3
        assert job_id == "batches/job-1"

    def test_writes_a_jsonl_manifest_keyed_by_custom_id(self, requests, tmp_path):
        self._classifier(FakeGeminiClient()).submit_batch(requests, tmp_path)

        manifest = tmp_path / "batch_requests.jsonl"
        lines = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
        assert [line["key"] for line in lines] == ["t0", "t1", "t2"]

    def test_manifest_lines_reference_uploaded_files_not_inline_bytes(self, requests, tmp_path):
        self._classifier(FakeGeminiClient()).submit_batch(requests, tmp_path)

        line = json.loads(
            (tmp_path / "batch_requests.jsonl").read_text(encoding="utf-8").splitlines()[0]
        )
        parts = line["request"]["contents"][0]["parts"]
        assert parts[0]["file_data"]["file_uri"].startswith("https://files/")
        assert "inline_data" not in json.dumps(parts)

    def test_manifest_carries_the_system_prompt_and_schema(self, requests, tmp_path):
        self._classifier(FakeGeminiClient()).submit_batch(requests, tmp_path)

        line = json.loads(
            (tmp_path / "batch_requests.jsonl").read_text(encoding="utf-8").splitlines()[0]
        )
        assert "activity classifier" in line["request"]["system_instruction"]["parts"][0]["text"]
        assert line["request"]["generation_config"]["response_json_schema"]["required"]

    def test_the_job_id_is_persisted_for_resume(self, requests, tmp_path):
        self._classifier(FakeGeminiClient()).submit_batch(requests, tmp_path)
        assert (tmp_path / "batch_job.txt").read_text(encoding="utf-8") == "batches/job-1"

    def test_submitting_nothing_is_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            self._classifier(FakeGeminiClient()).submit_batch([], tmp_path)

    def test_poll_returns_a_bare_state_name(self):
        client = FakeGeminiClient(state="JOB_STATE_RUNNING")
        assert self._classifier(client).poll_batch("batches/job-1") == "JOB_STATE_RUNNING"

    def test_fetch_reads_a_file_backed_result(self, tmp_path):
        dest = type("Dest", (), {"inlined_responses": None, "file_name": "files/out.jsonl"})()
        client = FakeGeminiClient(dest=dest)
        client._payload = (
            json.dumps(
                {
                    "key": "t0",
                    "response": {
                        "candidates": [{"content": {"parts": [{"text": json.dumps(VALID)}]}}]
                    },
                }
            ).encode()
            + b"\n"
        )

        responses = self._classifier(client).fetch_batch("batches/job-1", tmp_path)

        assert [r.custom_id for r in responses] == ["t0"]
        assert (tmp_path / "batch_results.jsonl").exists()

    def test_fetch_reads_inline_results(self, tmp_path):
        inline = [
            type(
                "Item",
                (),
                {
                    "metadata": {"key": "t0"},
                    "error": None,
                    "response": type("R", (), {"text": json.dumps(VALID)})(),
                },
            )()
        ]
        dest = type("Dest", (), {"inlined_responses": inline, "file_name": None})()

        responses = self._classifier(FakeGeminiClient(dest=dest)).fetch_batch("job", tmp_path)

        assert responses[0].custom_id == "t0"
        assert responses[0].classification is not None

    def test_an_inline_error_is_surfaced(self, tmp_path):
        inline = [
            type("Item", (), {"metadata": {"key": "t0"}, "error": "quota exceeded", "response": None})()
        ]
        dest = type("Dest", (), {"inlined_responses": inline, "file_name": None})()

        response = self._classifier(FakeGeminiClient(dest=dest)).fetch_batch("job", tmp_path)[0]

        assert response.classification is None and "quota" in response.error

    def test_a_job_without_a_destination_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match="no destination"):
            self._classifier(FakeGeminiClient(dest=None)).fetch_batch("job", tmp_path)

    def test_a_destination_with_neither_form_raises(self, tmp_path):
        dest = type("Dest", (), {"inlined_responses": None, "file_name": None})()
        with pytest.raises(RuntimeError, match="neither inline nor file"):
            self._classifier(FakeGeminiClient(dest=dest)).fetch_batch("job", tmp_path)


class FakeOpenAIClient:
    def __init__(self, *, status="completed", output=b"", errors=b""):
        self.created_batches = []
        self.uploaded = []
        outer = self

        class Files:
            def create(self, *, file, purpose):
                outer.uploaded.append(purpose)
                return type("F", (), {"id": "file-in"})()

            def content(self, file_id):
                payload = outer._output if file_id == "file-out" else outer._errors
                return type("C", (), {"read": lambda self: payload})()

        class Batches:
            def create(self, **kwargs):
                outer.created_batches.append(kwargs)
                return type("B", (), {"id": "batch_123"})()

            def retrieve(self, job_id):
                return type(
                    "B",
                    (),
                    {
                        "id": job_id,
                        "status": outer._status,
                        "output_file_id": "file-out" if outer._output else None,
                        "error_file_id": "file-err" if outer._errors else None,
                    },
                )()

        self.files = Files()
        self.batches = Batches()
        self._status = status
        self._output = output
        self._errors = errors


class TestOpenAIProvider:
    def _classifier(self, client=None):
        classifier = OpenAIClassifier("gpt-4o-mini", api_key="fake-key")
        if client is not None:
            classifier._client = client
        return classifier

    def test_batch_line_targets_the_chat_completions_endpoint(self, requests):
        line = self._classifier().build_batch_line(requests[0])
        assert line["url"] == "/v1/chat/completions"
        assert line["method"] == "POST"
        assert line["custom_id"] == "t0"

    def test_batch_line_embeds_the_grid_as_a_data_uri(self, requests):
        line = self._classifier().build_batch_line(requests[0])
        image = line["body"]["messages"][1]["content"][0]
        assert image["image_url"]["url"].startswith("data:image/jpeg;base64,")

    def test_batch_line_requests_strict_structured_output(self, requests):
        body = self._classifier().build_batch_line(requests[0])["body"]
        assert body["response_format"]["json_schema"]["strict"] is True

    def test_batch_line_carries_the_system_prompt(self, requests):
        body = self._classifier().build_batch_line(requests[0])["body"]
        assert body["messages"][0]["role"] == "system"
        assert "activity classifier" in body["messages"][0]["content"]

    def test_submit_uploads_the_manifest_then_creates_the_job(self, requests, tmp_path):
        client = FakeOpenAIClient()
        job_id = self._classifier(client).submit_batch(requests, tmp_path)

        assert client.uploaded == ["batch"]
        assert client.created_batches[0]["endpoint"] == "/v1/chat/completions"
        assert client.created_batches[0]["completion_window"] == "24h"
        assert job_id == "batch_123"
        assert (tmp_path / "batch_requests.jsonl").exists()

    def test_manifest_has_one_line_per_request(self, requests, tmp_path):
        self._classifier(FakeOpenAIClient()).submit_batch(requests, tmp_path)
        lines = (tmp_path / "batch_requests.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 3

    def test_poll_returns_the_status_string(self):
        assert self._classifier(FakeOpenAIClient(status="in_progress")).poll_batch("b") == "in_progress"

    def test_fetch_parses_the_output_file(self, tmp_path):
        output = (
            json.dumps(
                {
                    "custom_id": "t0",
                    "response": {"body": {"choices": [{"message": {"content": json.dumps(VALID)}}]}},
                }
            ).encode()
            + b"\n"
        )
        responses = self._classifier(FakeOpenAIClient(output=output)).fetch_batch("b", tmp_path)

        assert [r.custom_id for r in responses] == ["t0"]
        assert (tmp_path / "batch_results.jsonl").exists()

    def test_fetch_saves_the_error_file_alongside(self, tmp_path):
        self._classifier(FakeOpenAIClient(errors=b'{"custom_id":"t0"}\n')).fetch_batch("b", tmp_path)
        assert (tmp_path / "batch_errors.jsonl").exists()

    def test_a_job_with_no_output_returns_nothing(self, tmp_path):
        assert self._classifier(FakeOpenAIClient()).fetch_batch("b", tmp_path) == []

    def test_sync_classification_maps_responses(self, requests):
        classifier = self._classifier()

        class Completions:
            def create(self, **kwargs):
                message = type("M", (), {"content": json.dumps(VALID)})()
                return type(
                    "C",
                    (),
                    {
                        "choices": [type("Ch", (), {"message": message})()],
                        "usage": type("U", (), {"prompt_tokens": 800, "completion_tokens": 90})(),
                    },
                )()

        classifier._client = type(
            "Client", (), {"chat": type("Chat", (), {"completions": Completions()})()}
        )()

        responses = classifier.classify(requests)

        assert [r.custom_id for r in responses] == ["t0", "t1", "t2"]
        assert classifier.usage.input_tokens == 2400

    def test_a_sync_failure_is_isolated(self, requests):
        classifier = self._classifier()

        class Completions:
            def create(self, **kwargs):
                raise RuntimeError("server error")

        classifier._client = type(
            "Client", (), {"chat": type("Chat", (), {"completions": Completions()})()}
        )()

        responses = classifier.classify(requests)
        assert all(r.classification is None and "server error" in r.error for r in responses)


class TestHttpClientFactory:
    def test_reports_whether_impersonation_is_available(self):
        assert isinstance(impersonation_available(), bool)

    def test_headers_present_a_browser_identity(self):
        assert "Mozilla" in BROWSER_HEADERS["User-Agent"]
        assert BROWSER_HEADERS["Referer"] == "https://kick.com/"

    def test_returns_a_client_with_get_and_close(self):
        client = build_client(timeout=5.0)
        try:
            assert hasattr(client, "get") and hasattr(client, "close")
        finally:
            client.close()

    def test_falls_back_to_httpx_without_impersonation(self, monkeypatch):
        import httpx

        from kick_vod_analyser.ingest import http as http_module

        monkeypatch.setattr(http_module, "impersonation_available", lambda: False)
        client = build_client(timeout=5.0)
        try:
            assert isinstance(client, httpx.Client)
        finally:
            client.close()
