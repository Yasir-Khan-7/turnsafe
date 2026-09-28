"""Offline token estimates.

Neither Anthropic nor Gemini ship a local tokenizer, so the default counter is
a character heuristic (about 4 characters per token for English and code).
It is deliberately a little pessimistic. For exact numbers pass your own
``counter`` to ``fit()``, for example one built on ``tiktoken`` or on
Anthropic's ``messages.count_tokens`` endpoint.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

CHARS_PER_TOKEN = 4
MESSAGE_OVERHEAD = 4
# Images are billed by size, not by the length of their base64 text.
IMAGE_TOKENS = 1_600


def approx_tokens(message: Mapping[str, Any]) -> int:
    """Estimate the tokens one message costs. Never returns less than 1."""
    chars, images = _walk(message)
    return max(1, math.ceil(chars / CHARS_PER_TOKEN) + images * IMAGE_TOKENS + MESSAGE_OVERHEAD)


def _walk(value: Any) -> tuple[int, int]:
    if isinstance(value, str):
        if value.startswith("data:") and ";base64," in value[:100]:
            return 0, 1
        return len(value), 0
    if isinstance(value, Mapping):
        if value.get("type") == "base64" and "data" in value:
            return 0, 1
        chars = images = 0
        for key, item in value.items():
            c, i = _walk(item)
            chars += c + len(str(key))
            images += i
        return chars, images
    if isinstance(value, (list, tuple)):
        chars = images = 0
        for item in value:
            c, i = _walk(item)
            chars += c
            images += i
        return chars, images
    if value is None or isinstance(value, bool):
        return 0, 0
    return len(str(value)), 0
