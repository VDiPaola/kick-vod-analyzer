"""Classifier construction and the offline mock provider."""

from __future__ import annotations

import hashlib
import logging

from ..config import Settings
from ..models import Classification, PrimaryCategory
from .base import Classifier, ClassificationRequest, ClassificationResponse

log = logging.getLogger(__name__)

PROVIDERS = ("gemini", "openai", "mock")


class MockClassifier(Classifier):
    """Deterministic offline classifier for dry runs and tests.

    Derives a stable category from a hash of the grid bytes, so repeated runs
    over the same VOD produce an identical timeline. Costs nothing and needs no
    credentials, which makes the full pipeline exercisable end to end.
    """

    provider = "mock"

    _CATEGORIES = (
        PrimaryCategory.GAMING,
        PrimaryCategory.JUST_CHATTING,
        PrimaryCategory.REACTION,
        PrimaryCategory.INTERMISSION,
    )
    _TITLES = ("Valorant", "", "YouTube", "")

    def __init__(self, model: str = "mock-v1") -> None:
        super().__init__(model)

    def classify(self, requests: list[ClassificationRequest]) -> list[ClassificationResponse]:
        responses: list[ClassificationResponse] = []
        for request in requests:
            digest = hashlib.sha256(request.image_path.read_bytes()).digest()
            index = digest[0] % len(self._CATEGORIES)
            category = self._CATEGORIES[index]
            classification = Classification(
                primary_category=category,
                specific_title_or_context=self._TITLES[index],
                sub_activity="mock sub-activity",
                is_streamer_on_screen=bool(digest[1] & 1),
                is_afk_or_brb=category is PrimaryCategory.INTERMISSION,
                confidence_score=0.5 + (digest[2] % 50) / 100.0,
                visual_evidence="mock provider; no model was called",
            )
            self.usage.requests += 1
            responses.append(
                ClassificationResponse(custom_id=request.custom_id, classification=classification)
            )
        return responses


def build_classifier(provider: str, settings: Settings, *, model: str | None = None) -> Classifier:
    """Instantiate the requested provider, validating credentials up front."""
    if provider == "mock":
        return MockClassifier(model or "mock-v1")

    if provider == "gemini":
        from .gemini import GeminiClassifier

        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        return GeminiClassifier(model or settings.gemini_model, settings.gemini_api_key)

    if provider == "openai":
        from .openai_provider import OpenAIClassifier

        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        return OpenAIClassifier(model or settings.openai_model, settings.openai_api_key)

    raise ValueError(f"unknown provider: {provider}. Expected one of {', '.join(PROVIDERS)}")
