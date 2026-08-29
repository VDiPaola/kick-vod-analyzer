"""Retry transient provider failures (rate limits, overload) with backoff."""

from __future__ import annotations

import logging
import random
import re
import time
from collections.abc import Callable
from typing import TypeVar

log = logging.getLogger(__name__)

T = TypeVar("T")

RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}
RETRYABLE_MARKERS = (
    "resource_exhausted",
    "rate limit",
    "ratelimit",
    "quota",
    "too many requests",
    "overloaded",
    "unavailable",
    "try again later",
)
_RETRY_DELAY_RE = re.compile(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s", re.IGNORECASE)
_RETRY_IN_RE = re.compile(r"retry (?:in|after) (\d+(?:\.\d+)?)\s*(ms|s|seconds?|m|minutes?)?", re.IGNORECASE)


def status_code_of(exc: BaseException) -> int | None:
    for attr in ("status_code", "code", "status"):
        value = getattr(exc, attr, None)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def is_retryable(exc: BaseException) -> bool:
    code = status_code_of(exc)
    if code is not None:
        return code in RETRYABLE_STATUS_CODES
    text = str(exc).lower()
    return any(marker in text for marker in RETRYABLE_MARKERS)


def suggested_delay(exc: BaseException) -> float | None:
    """Delay the provider asked for, from headers or the error body."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None:
        try:
            header = headers.get("retry-after")
        except AttributeError:
            header = None
        if header:
            try:
                return float(header)
            except ValueError:
                pass

    text = str(exc)
    match = _RETRY_DELAY_RE.search(text)
    if match:
        return float(match.group(1))
    match = _RETRY_IN_RE.search(text)
    if match:
        value = float(match.group(1))
        unit = (match.group(2) or "s").lower()
        if unit == "ms":
            return value / 1000
        if unit.startswith("m") and unit != "ms":
            return value * 60
        return value
    return None


def call_with_retry(
    fn: Callable[[], T],
    *,
    label: str,
    attempts: int = 8,
    base_delay: float = 2.0,
    max_delay: float = 120.0,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call `fn`, retrying rate-limit and transient errors with exponential backoff.

    Non-retryable errors and the final failed attempt propagate to the caller.
    """
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            if attempt >= attempts or not is_retryable(exc):
                raise
            delay = suggested_delay(exc)
            if delay is None:
                delay = min(max_delay, base_delay * 2 ** (attempt - 1))
            delay = min(max_delay, delay) + random.uniform(0, min(1.0, delay * 0.1))
            log.warning(
                "%s hit a transient error (attempt %d/%d), retrying in %.0fs: %s",
                label,
                attempt,
                attempts,
                delay,
                _brief(exc),
            )
            sleep(delay)
    raise AssertionError("unreachable")


def _brief(exc: BaseException) -> str:
    text = str(exc).replace("\n", " ")
    return text if len(text) <= 200 else text[:197] + "..."
