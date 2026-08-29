"""OpenAI classifier: synchronous and Batch API paths."""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..sampling.grid import encode_base64
from .base import (
    Classifier,
    ClassificationRequest,
    ClassificationResponse,
    parse_classification,
)
from .retry import call_with_retry
from .prompts import SYSTEM_PROMPT, openai_response_format

log = logging.getLogger(__name__)

TERMINAL_STATES = {"completed", "failed", "expired", "cancelled"}


class OpenAIClassifier(Classifier):
    """Vision classification through the OpenAI chat completions API.

    Batch mode embeds each grid as a data URI directly in the JSONL, which the
    Files endpoint accepts. There is no separate upload step to coordinate.
    """

    provider = "openai"

    def __init__(self, model: str, api_key: str | None = None, *, max_workers: int = 8, max_attempts: int = 8) -> None:
        super().__init__(model)
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai is not installed; pip install openai") from exc

        self._client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self.max_workers = max_workers
        self.max_attempts = max_attempts

    def _messages(self, request: ClassificationRequest) -> list[dict]:
        data_uri = f"data:{request.mime_type};base64,{encode_base64(request.image_path)}"
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri, "detail": "low"}},
                    {"type": "text", "text": request.user_prompt},
                ],
            },
        ]

    def classify(self, requests: list[ClassificationRequest]) -> list[ClassificationResponse]:
        if not requests:
            return []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            return list(pool.map(self._classify_one, requests))

    def _classify_one(self, request: ClassificationRequest) -> ClassificationResponse:
        try:
            messages = self._messages(request)
            completion = call_with_retry(
                lambda: self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format=openai_response_format(),
                    temperature=0.1,
                    max_tokens=400,
                ),
                label=f"openai request {request.custom_id}",
                attempts=self.max_attempts,
            )
        except Exception as exc:
            log.warning("openai request %s failed: %s", request.custom_id, exc)
            return ClassificationResponse(custom_id=request.custom_id, error=str(exc))

        usage = getattr(completion, "usage", None)
        if usage is not None:
            self.usage.requests += 1
            self.usage.input_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
            self.usage.output_tokens += int(getattr(usage, "completion_tokens", 0) or 0)

        text = completion.choices[0].message.content if completion.choices else None
        classification, error = parse_classification(text)
        return ClassificationResponse(
            custom_id=request.custom_id,
            classification=classification,
            error=error,
            raw_text=text,
        )

    def supports_batch(self) -> bool:
        return True

    def build_batch_line(self, request: ClassificationRequest) -> dict:
        return {
            "custom_id": request.custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": self.model,
                "messages": self._messages(request),
                "response_format": openai_response_format(),
                "temperature": 0.1,
                "max_tokens": 400,
            },
        }

    def submit_batch(self, requests: list[ClassificationRequest], work_dir: Path) -> str:
        if not requests:
            raise ValueError("no requests to submit")
        work_dir.mkdir(parents=True, exist_ok=True)

        manifest = work_dir / "batch_requests.jsonl"
        with manifest.open("w", encoding="utf-8") as handle:
            for request in requests:
                handle.write(json.dumps(self.build_batch_line(request)) + "\n")

        with manifest.open("rb") as handle:
            uploaded = self._client.files.create(file=handle, purpose="batch")

        job = self._client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"source": "kick-vod-analyser", "run": work_dir.name},
        )
        log.info("submitted openai batch job %s with %d requests", job.id, len(requests))
        (work_dir / "batch_job.txt").write_text(job.id, encoding="utf-8")
        return job.id

    def poll_batch(self, job_id: str) -> str:
        return self._client.batches.retrieve(job_id).status

    def fetch_batch(self, job_id: str, work_dir: Path) -> list[ClassificationResponse]:
        job = self._client.batches.retrieve(job_id)
        responses: list[ClassificationResponse] = []

        if job.output_file_id:
            content = self._client.files.content(job.output_file_id).read()
            results_path = work_dir / "batch_results.jsonl"
            results_path.write_bytes(content)
            responses.extend(parse_batch_results(results_path))

        if job.error_file_id:
            errors = self._client.files.content(job.error_file_id).read()
            (work_dir / "batch_errors.jsonl").write_bytes(errors)

        return responses


def parse_batch_results(path: Path) -> list[ClassificationResponse]:
    """Parse an OpenAI batch output JSONL into responses keyed by custom id."""
    responses: list[ClassificationResponse] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                responses.append(
                    ClassificationResponse(custom_id=f"line{index}", error=f"bad JSONL: {exc}")
                )
                continue

            custom_id = str(record.get("custom_id") or f"line{index}")
            if record.get("error"):
                responses.append(
                    ClassificationResponse(custom_id=custom_id, error=json.dumps(record["error"]))
                )
                continue

            body = ((record.get("response") or {}).get("body")) or {}
            choices = body.get("choices") or []
            text = ((choices[0] if choices else {}).get("message") or {}).get("content")
            classification, error = parse_classification(text)
            responses.append(
                ClassificationResponse(
                    custom_id=custom_id,
                    classification=classification,
                    error=error,
                    raw_text=text,
                )
            )
    return responses
