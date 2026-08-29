"""Provider-agnostic classification interface."""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from ..models import Classification

log = logging.getLogger(__name__)

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class ClassificationRequest:
    """One grid image plus its prompt, keyed by VOD offset."""

    custom_id: str
    offset_seconds: float
    image_path: Path
    user_prompt: str
    mime_type: str = "image/jpeg"


@dataclass
class ClassificationResponse:
    """Provider output for a single request."""

    custom_id: str
    classification: Classification | None = None
    error: str | None = None
    raw_text: str | None = None


@dataclass
class UsageEstimate:
    """Token counts and derived cost for a completed run."""

    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    input_cost_usd: float = 0.0
    output_cost_usd: float = 0.0

    @property
    def total_cost_usd(self) -> float:
        return self.input_cost_usd + self.output_cost_usd


class Classifier(ABC):
    """Synchronous and batch classification over grid images."""

    provider = "base"

    def __init__(self, model: str) -> None:
        self.model = model
        self.usage = UsageEstimate()

    @abstractmethod
    def classify(self, requests: list[ClassificationRequest]) -> list[ClassificationResponse]: ...

    def supports_batch(self) -> bool:
        return False

    def submit_batch(self, requests: list[ClassificationRequest], work_dir: Path) -> str:
        raise NotImplementedError(f"{self.provider} does not implement batch submission")

    def poll_batch(self, job_id: str) -> str:
        raise NotImplementedError(f"{self.provider} does not implement batch polling")

    def fetch_batch(self, job_id: str, work_dir: Path) -> list[ClassificationResponse]:
        raise NotImplementedError(f"{self.provider} does not implement batch retrieval")


def parse_classification(text: str | None) -> tuple[Classification | None, str | None]:
    """Parse model output into a Classification, tolerating fenced JSON.

    Structured output modes normally return bare JSON, but a model occasionally
    wraps it in a markdown fence or adds a preamble; recovering the first JSON
    object avoids discarding an otherwise valid answer.
    """
    if not text or not text.strip():
        return None, "empty response"

    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        if candidate.lower().startswith("json"):
            candidate = candidate[4:]
        candidate = candidate.strip()

    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        match = _JSON_BLOCK_RE.search(candidate)
        if not match:
            return None, "no JSON object in response"
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            return None, f"invalid JSON: {exc}"

    try:
        return Classification.model_validate(payload), None
    except Exception as exc:
        return None, f"schema validation failed: {exc}"
