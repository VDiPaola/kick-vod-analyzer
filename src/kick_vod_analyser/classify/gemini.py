"""Gemini classifier: synchronous and Batch API paths."""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .base import (
    Classifier,
    ClassificationRequest,
    ClassificationResponse,
    parse_classification,
)
from .retry import call_with_retry
from .prompts import SYSTEM_PROMPT, gemini_response_schema

log = logging.getLogger(__name__)

TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
    "JOB_STATE_PARTIALLY_SUCCEEDED",
}


class GeminiClassifier(Classifier):
    """Vision classification through google-genai.

    Batch mode uploads each grid through the Files API and references it by URI
    from a JSONL manifest. Embedding base64 image bytes inline would breach the
    inline request size ceiling well before a full-length VOD is covered.
    """

    provider = "gemini"

    def __init__(self, model: str, api_key: str | None = None, *, max_workers: int = 8, max_attempts: int = 8) -> None:
        super().__init__(model)
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError("google-genai is not installed; pip install google-genai") from exc

        self._genai = genai
        from google.genai import types

        self._types = types
        self._client = genai.Client(api_key=api_key) if api_key else genai.Client()
        self.max_workers = max_workers
        self.max_attempts = max_attempts

    def _generation_config(self):
        return self._types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_json_schema=gemini_response_schema(),
            temperature=0.1,
        )

    def classify(self, requests: list[ClassificationRequest]) -> list[ClassificationResponse]:
        if not requests:
            return []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            return list(pool.map(self._classify_one, requests))

    def _classify_one(self, request: ClassificationRequest) -> ClassificationResponse:
        try:
            contents = [
                self._types.Part.from_bytes(
                    data=request.image_path.read_bytes(), mime_type=request.mime_type
                ),
                self._types.Part.from_text(text=request.user_prompt),
            ]
            response = call_with_retry(
                lambda: self._client.models.generate_content(
                    model=self.model, contents=contents, config=self._generation_config()
                ),
                label=f"gemini request {request.custom_id}",
                attempts=self.max_attempts,
            )
        except Exception as exc:
            log.warning("gemini request %s failed: %s", request.custom_id, exc)
            return ClassificationResponse(custom_id=request.custom_id, error=str(exc))

        self._record_usage(getattr(response, "usage_metadata", None))
        text = getattr(response, "text", None)
        classification, error = parse_classification(text)
        return ClassificationResponse(
            custom_id=request.custom_id,
            classification=classification,
            error=error,
            raw_text=text,
        )

    def _record_usage(self, metadata) -> None:
        if metadata is None:
            return
        self.usage.requests += 1
        self.usage.input_tokens += int(getattr(metadata, "prompt_token_count", 0) or 0)
        self.usage.output_tokens += int(getattr(metadata, "candidates_token_count", 0) or 0)

    def supports_batch(self) -> bool:
        return True

    def submit_batch(self, requests: list[ClassificationRequest], work_dir: Path) -> str:
        if not requests:
            raise ValueError("no requests to submit")
        work_dir.mkdir(parents=True, exist_ok=True)

        uploads: dict[str, tuple[str, str]] = {}
        for request in requests:
            uploaded = self._client.files.upload(
                file=str(request.image_path),
                config={"mime_type": request.mime_type, "display_name": request.custom_id},
            )
            uploads[request.custom_id] = (uploaded.uri, uploaded.mime_type or request.mime_type)
            log.debug("uploaded %s as %s", request.custom_id, uploaded.uri)

        manifest = work_dir / "batch_requests.jsonl"
        with manifest.open("w", encoding="utf-8") as handle:
            for request in requests:
                uri, mime = uploads[request.custom_id]
                line = {
                    "key": request.custom_id,
                    "request": {
                        "contents": [
                            {
                                "role": "user",
                                "parts": [
                                    {"file_data": {"file_uri": uri, "mime_type": mime}},
                                    {"text": request.user_prompt},
                                ],
                            }
                        ],
                        "generation_config": {
                            "response_mime_type": "application/json",
                            "response_json_schema": gemini_response_schema(),
                            "temperature": 0.1,
                        },
                        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                    },
                }
                handle.write(json.dumps(line) + "\n")

        manifest_file = self._client.files.upload(
            file=str(manifest),
            config={"mime_type": "application/jsonl", "display_name": "kva-batch-requests"},
        )
        job = self._client.batches.create(
            model=self.model,
            src=manifest_file.name,
            config={"display_name": f"kva-{work_dir.name}"},
        )
        log.info("submitted gemini batch job %s with %d requests", job.name, len(requests))
        (work_dir / "batch_job.txt").write_text(job.name, encoding="utf-8")
        return job.name

    def poll_batch(self, job_id: str) -> str:
        job = self._client.batches.get(name=job_id)
        state = getattr(job.state, "name", None) or str(job.state)
        return state.rsplit(".", 1)[-1]

    def fetch_batch(self, job_id: str, work_dir: Path) -> list[ClassificationResponse]:
        job = self._client.batches.get(name=job_id)
        dest = getattr(job, "dest", None)
        if dest is None:
            raise RuntimeError(f"batch job {job_id} has no destination")

        if getattr(dest, "inlined_responses", None):
            return [
                self._response_from_inline(index, item)
                for index, item in enumerate(dest.inlined_responses)
            ]

        file_name = getattr(dest, "file_name", None)
        if not file_name:
            raise RuntimeError(f"batch job {job_id} produced neither inline nor file results")

        payload = self._client.files.download(file=file_name)
        results_path = work_dir / "batch_results.jsonl"
        results_path.write_bytes(payload)
        return parse_batch_results(results_path)

    def _response_from_inline(self, index: int, item) -> ClassificationResponse:
        metadata = getattr(item, "metadata", None) or {}
        custom_id = str(metadata.get("key") or index)
        error = getattr(item, "error", None)
        if error:
            return ClassificationResponse(custom_id=custom_id, error=str(error))
        text = _text_from_response(getattr(item, "response", None))
        classification, parse_error = parse_classification(text)
        return ClassificationResponse(
            custom_id=custom_id, classification=classification, error=parse_error, raw_text=text
        )


def _text_from_response(response) -> str | None:
    if response is None:
        return None
    text = getattr(response, "text", None)
    if text:
        return text
    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            part_text = getattr(part, "text", None)
            if part_text:
                return part_text
    return None


def parse_batch_results(path: Path) -> list[ClassificationResponse]:
    """Parse a Gemini batch results JSONL into responses keyed by custom id.

    Kept module-level and dependency-free so results can be reparsed offline
    without constructing a client.
    """
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

            custom_id = str(record.get("key") or record.get("custom_id") or f"line{index}")
            if record.get("error"):
                responses.append(
                    ClassificationResponse(custom_id=custom_id, error=json.dumps(record["error"]))
                )
                continue

            text = extract_text_from_record(record.get("response") or record)
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


def extract_text_from_record(payload: dict) -> str | None:
    """Pull the first text part out of a raw GenerateContentResponse dict."""
    if not isinstance(payload, dict):
        return None
    for candidate in payload.get("candidates") or []:
        parts = ((candidate or {}).get("content") or {}).get("parts") or []
        for part in parts:
            text = (part or {}).get("text")
            if text:
                return text
    return payload.get("text")
