"""Kick VOD activity detection and segmentation."""

from .config import Settings, load_settings
from .models import (
    ChatMessage,
    ChatWindow,
    Classification,
    PrimaryCategory,
    SamplePoint,
    SampleResult,
    Segment,
    Timeline,
    VodInfo,
)
from .pipeline import Pipeline, RunOptions, RunReport

__version__ = "0.1.0"

__all__ = [
    "ChatMessage",
    "ChatWindow",
    "Classification",
    "Pipeline",
    "PrimaryCategory",
    "RunOptions",
    "RunReport",
    "SamplePoint",
    "SampleResult",
    "Segment",
    "Settings",
    "Timeline",
    "VodInfo",
    "load_settings",
]
