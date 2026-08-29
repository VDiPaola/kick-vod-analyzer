from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from kick_vod_analyser.config import ChatSettings, SmoothingSettings, load_settings
from kick_vod_analyser.models import (
    ChatMessage,
    Classification,
    PrimaryCategory,
    SampleResult,
    VodInfo,
)

HAS_FFMPEG = shutil.which("ffmpeg") is not None
requires_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not on PATH")


@pytest.fixture
def settings(tmp_path):
    return load_settings(work_dir=tmp_path / "work", out_dir=tmp_path / "out")


@pytest.fixture
def chat_settings():
    return ChatSettings()


@pytest.fixture
def smoothing_settings():
    return SmoothingSettings()


@pytest.fixture
def vod():
    return VodInfo(
        vod_id="test-vod",
        url="https://kick.com/teststreamer/videos/test-vod",
        channel_slug="teststreamer",
        channel_id=42,
        title="10 hour marathon",
        duration_seconds=3600.0,
        started_at_epoch=1_700_000_000.0,
        playback_url="https://cdn.example/test.m3u8",
    )


def make_classification(
    category: PrimaryCategory | str = PrimaryCategory.GAMING,
    title: str = "Valorant",
    *,
    confidence: float = 0.9,
    afk: bool = False,
    sub_activity: str = "In-Game Match",
) -> Classification:
    return Classification(
        primary_category=category,
        specific_title_or_context=title,
        sub_activity=sub_activity,
        is_streamer_on_screen=False,
        is_afk_or_brb=afk,
        confidence_score=confidence,
        visual_evidence="test fixture",
    )


def make_sample(
    offset: float,
    category: PrimaryCategory | str = PrimaryCategory.GAMING,
    title: str = "Valorant",
    *,
    confidence: float = 0.9,
    afk: bool = False,
    trigger: str = "scene",
) -> SampleResult:
    return SampleResult(
        offset_seconds=offset,
        trigger=trigger,
        classification=make_classification(category, title, confidence=confidence, afk=afk),
    )


def make_chat(offset: float, text: str, username: str = "viewer") -> ChatMessage:
    return ChatMessage(offset_seconds=offset, username=username, text=text)


@pytest.fixture
def sample_factory():
    return make_sample


@pytest.fixture
def chat_factory():
    return make_chat


@pytest.fixture
def solid_frame(tmp_path):
    def build(color: tuple[int, int, int], name: str = "frame.jpg", size=(320, 180)) -> Path:
        path = tmp_path / name
        Image.new("RGB", size, color).save(path, format="JPEG", quality=90)
        return path

    return build


@pytest.fixture(scope="session")
def synthetic_video(tmp_path_factory) -> Path:
    """A 90 second clip with three visually distinct 30 second acts.

    Each act uses a different ffmpeg source so the scene filter registers a real
    cut at 30s and 60s, which lets the sampling stage be exercised for real
    rather than against parsed log text.
    """
    if not HAS_FFMPEG:
        pytest.skip("ffmpeg not on PATH")

    out = tmp_path_factory.mktemp("media") / "synthetic.mp4"
    filtergraph = (
        "testsrc=size=640x360:rate=10:duration=30[a];"
        "smptebars=size=640x360:rate=10:duration=30[b];"
        "color=c=navy:size=640x360:rate=10:duration=30[c];"
        "[a][b][c]concat=n=3:v=1:a=0[v]"
    )
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-filter_complex", filtergraph,
            "-map", "[v]",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-g", "20",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    return out
