"""Domain models shared across the pipeline stages."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PrimaryCategory(str, Enum):
    GAMING = "Gaming"
    JUST_CHATTING = "Just Chatting / Podcast"
    IRL = "IRL / Outdoors"
    REACTION = "Reaction / Media Share"
    CREATIVE = "Coding / Creative"
    GAMBLING = "Gambling / Slots"
    INTERMISSION = "Intermission / AFK / BRB"
    OFFLINE = "Technical Difficulties / Offline"


TriggerKind = Literal["scene", "heartbeat", "boundary"]


class VodInfo(BaseModel):
    """Resolved metadata for a single Kick VOD."""

    model_config = ConfigDict(frozen=True)

    vod_id: str
    url: str
    channel_slug: str
    channel_id: int | None = None
    title: str = ""
    duration_seconds: float = Field(gt=0)
    started_at_epoch: float | None = None
    playback_url: str | None = None

    @property
    def duration_hours(self) -> float:
        return self.duration_seconds / 3600.0


class ChatMessage(BaseModel):
    """A single normalised chat line positioned on the VOD timeline."""

    model_config = ConfigDict(frozen=True)

    offset_seconds: float = Field(ge=0)
    username: str
    text: str
    emotes: tuple[str, ...] = ()

    @field_validator("text")
    @classmethod
    def _strip_text(cls, value: str) -> str:
        return value.strip()


class SamplePoint(BaseModel):
    """A timestamp the sampler decided is worth classifying."""

    model_config = ConfigDict(frozen=True)

    offset_seconds: float = Field(ge=0)
    trigger: TriggerKind
    scene_score: float | None = None

    @property
    def custom_id(self) -> str:
        return f"t{int(round(self.offset_seconds))}"


class ChatWindow(BaseModel):
    """Condensed chat context surrounding a sample point."""

    offset_seconds: float
    lines: list[str] = Field(default_factory=list)
    message_count: int = 0
    unique_chatters: int = 0

    @property
    def is_empty(self) -> bool:
        return not self.lines

    def render(self) -> str:
        if self.is_empty:
            return "(no chat activity in window)"
        header = f"{self.message_count} messages from {self.unique_chatters} chatters:"
        return header + "\n" + "\n".join(self.lines)


class Classification(BaseModel):
    """Structured LLM verdict for one sample point."""

    primary_category: PrimaryCategory
    specific_title_or_context: str = ""
    sub_activity: str = ""
    is_streamer_on_screen: bool = False
    is_afk_or_brb: bool = False
    confidence_score: float = Field(ge=0.0, le=1.0)
    visual_evidence: str = ""

    @field_validator("confidence_score", mode="before")
    @classmethod
    def _clamp_confidence(cls, value: object) -> object:
        if isinstance(value, (int, float)):
            return min(1.0, max(0.0, float(value)))
        return value


class SampleResult(BaseModel):
    """A sample point joined with its classification."""

    offset_seconds: float
    trigger: TriggerKind
    classification: Classification
    grid_path: str | None = None
    error: str | None = None


class Segment(BaseModel):
    """A contiguous span of the timeline holding one activity state."""

    start_seconds: float
    end_seconds: float
    primary_category: PrimaryCategory
    specific_title_or_context: str = ""
    sub_activity: str = ""
    is_afk_or_brb: bool = False
    confidence_score: float = 0.0
    sample_count: int = 1
    absorbed_samples: int = 0

    @property
    def duration_seconds(self) -> float:
        return max(0.0, self.end_seconds - self.start_seconds)

    @property
    def label(self) -> str:
        title = self.specific_title_or_context.strip()
        if title and title.lower() not in {"n/a", "none", "unknown"}:
            return f"{self.primary_category.value}: {title}"
        return self.primary_category.value


class Timeline(BaseModel):
    """Final pipeline artefact."""

    vod: VodInfo
    segments: list[Segment] = Field(default_factory=list)
    samples: list[SampleResult] = Field(default_factory=list)
    model: str = ""
    provider: str = ""
