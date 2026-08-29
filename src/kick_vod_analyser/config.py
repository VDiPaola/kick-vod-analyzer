"""Runtime configuration loaded from environment and .env files."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SamplingSettings(BaseSettings):
    """Controls which timestamps get classified."""

    model_config = SettingsConfigDict(env_prefix="KVA_SAMPLING_", extra="ignore")

    scene_threshold: float = 0.35
    heartbeat_seconds: float = 900.0
    min_gap_seconds: float = 45.0
    burst_offsets: tuple[float, ...] = (-6.0, -2.0, 2.0, 6.0)
    grid_width: int = 1280
    grid_height: int = 720
    jpeg_quality: int = 78
    phash_distance: int = 6
    max_samples: int = 0


class ChatSettings(BaseSettings):
    """Controls chat window construction."""

    model_config = SettingsConfigDict(env_prefix="KVA_CHAT_", extra="ignore")

    window_seconds: float = 45.0
    max_lines: int = 30
    min_message_length: int = 2
    drop_bot_commands: bool = True


class SmoothingSettings(BaseSettings):
    """Controls timeline debouncing."""

    model_config = SettingsConfigDict(env_prefix="KVA_SMOOTHING_", extra="ignore")

    min_segment_seconds: float = 60.0
    alt_tab_window_seconds: float = 90.0
    confirm_consecutive: int = 2
    min_confidence: float = 0.35


class Settings(BaseSettings):
    """Top-level settings object."""

    model_config = SettingsConfigDict(
        env_prefix="KVA_",
        env_file=(".env", ".env.local"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")

    gemini_model: str = "gemini-3.5-flash-lite"
    openai_model: str = "gpt-4o-mini"

    work_dir: Path = Path("./work")
    out_dir: Path = Path("./out")
    ffmpeg_binary: str = "ffmpeg"
    ffprobe_binary: str = "ffprobe"
    http_timeout: float = 30.0
    kick_auth_token: str | None = None
    kick_chat_workers: int = 8
    batch_poll_seconds: float = 60.0

    sampling: SamplingSettings = Field(default_factory=SamplingSettings)
    chat: ChatSettings = Field(default_factory=ChatSettings)
    smoothing: SmoothingSettings = Field(default_factory=SmoothingSettings)

    def vod_work_dir(self, vod_id: str) -> Path:
        return self.work_dir / vod_id

    def vod_out_dir(self, vod_id: str) -> Path:
        return self.out_dir / vod_id


def load_settings(**overrides: object) -> Settings:
    return Settings(**overrides)
