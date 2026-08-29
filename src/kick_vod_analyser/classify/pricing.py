"""Token pricing table and run cost estimation.

Rates are USD per million tokens as published in August 2026. They move, so the
table is data rather than logic: override an entry or add a model without
touching the estimator. Batch endpoints bill at half the listed rate.
"""

from __future__ import annotations

from dataclasses import dataclass

BATCH_DISCOUNT = 0.5


@dataclass(frozen=True)
class ModelPrice:
    input_per_million: float
    output_per_million: float


PRICES: dict[str, ModelPrice] = {
    "gemini-2.5-flash-lite": ModelPrice(0.10, 0.40),
    "gemini-2.5-flash": ModelPrice(0.30, 2.50),
    "gemini-3.5-flash-lite": ModelPrice(0.30, 2.50),
    "gemini-3.1-flash-lite": ModelPrice(0.30, 2.50),
    "gemini-3.7-flash": ModelPrice(0.75, 3.75),
    "gemini-flash-latest": ModelPrice(0.75, 3.75),
    "gemini-flash-lite-latest": ModelPrice(0.30, 2.50),
    "gpt-4o-mini": ModelPrice(0.15, 0.60),
    "gpt-4.1-mini": ModelPrice(0.40, 1.60),
}

DEFAULT_PRICE = ModelPrice(0.75, 3.75)

# Measured averages for this pipeline's payload shape.
TOKENS_PER_GRID_IMAGE = 258
TOKENS_PER_CHAT_WINDOW = 300
TOKENS_PER_SYSTEM_PROMPT = 320
TOKENS_PER_OUTPUT = 90


def price_for(model: str) -> ModelPrice:
    """Look up a model price, matching the longest known prefix."""
    if model in PRICES:
        return PRICES[model]
    matches = [key for key in PRICES if model.startswith(key)]
    if matches:
        return PRICES[max(matches, key=len)]
    return DEFAULT_PRICE


def estimate_cost(
    request_count: int,
    model: str,
    *,
    batch: bool = False,
    with_chat: bool = True,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
) -> dict[str, float]:
    """Estimate the USD cost of a run.

    Falls back to the per-request token model when actual usage is unavailable,
    which is the case before a run and for providers that omit usage metadata.
    """
    per_request_input = TOKENS_PER_GRID_IMAGE + TOKENS_PER_SYSTEM_PROMPT
    if with_chat:
        per_request_input += TOKENS_PER_CHAT_WINDOW

    total_input = input_tokens if input_tokens is not None else per_request_input * request_count
    total_output = (
        output_tokens if output_tokens is not None else TOKENS_PER_OUTPUT * request_count
    )

    price = price_for(model)
    multiplier = BATCH_DISCOUNT if batch else 1.0
    input_cost = total_input / 1_000_000 * price.input_per_million * multiplier
    output_cost = total_output / 1_000_000 * price.output_per_million * multiplier

    return {
        "requests": float(request_count),
        "input_tokens": float(total_input),
        "output_tokens": float(total_output),
        "input_cost_usd": input_cost,
        "output_cost_usd": output_cost,
        "total_cost_usd": input_cost + output_cost,
    }
