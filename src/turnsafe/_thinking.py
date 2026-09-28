"""Keep Anthropic thinking blocks valid after the history has been changed.

On models with thinking-block prefix binding (Claude Fable 5.1 and Opus 5.5,
enforced by default for accounts created on or after 2026-08-31), a replayed
``thinking`` or ``redacted_thinking`` block is only valid while every message
before it is unchanged. Trimming from the front, dropping a message or
editing one invalidates that block and every later one, and the API answers
"Invalid `signature` in `thinking` block. The block is bound to a different
conversation."

The documented recovery is to remove the thinking blocks from the first
changed position onward, leaving everything else in place, or to set
``prefix_mismatch_behavior: "drop_block"`` and let the API drop them.
https://platform.claude.com/docs/en/build-with-claude/thinking-troubleshooting

Models without prefix binding differ: they want the latest assistant turn's
thinking replayed unchanged, and manual extended thinking requires it in a
tool loop. That is why ``"keep"`` is the default and stripping is opt-in.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from ._types import Message

ThinkingPolicy = Literal["strip_after_change", "keep"]
THINKING_BLOCKS = ("thinking", "redacted_thinking")


def first_change(before: Sequence[Message], after: Sequence[Message]) -> int:
    """Index of the first message in ``after`` that differs from ``before``."""
    for n, (a, b) in enumerate(zip(before, after, strict=False)):
        if a != b:
            return n
    return min(len(before), len(after))


def strip_after_change(
    before: Sequence[Message], after: list[Message], src: Sequence[int] | None = None
) -> tuple[list[Message], list[int]]:
    """Remove every thinking block in ``after`` that follows its first change.

    ``src[n]`` is the index in ``before`` that ``after[n]`` was derived from.
    When the first changed message is an edited version of the message at the
    same position, thinking blocks that come before its first changed block are
    kept: nothing before them changed.
    """
    n = first_change(before, after)
    if n >= len(after):
        return after, []
    msg = after[n]
    content = msg.get("content")
    same = src is not None and n < len(src) and src[n] == n and n < len(before)
    old = before[n].get("content") if same else None
    if msg.get("role") == "assistant" and isinstance(content, list) and isinstance(old, list):
        b = next(
            (k for k, (x, y) in enumerate(zip(old, content, strict=False)) if x != y),
            min(len(old), len(content)),
        )
        kept = content[:b] + [x for x in content[b:] if x.get("type") not in THINKING_BLOCKS]
        head: list[Message] = after[:n]
        touched = [n] if len(kept) != len(content) else []
        if kept:
            head.append({**msg, "content": kept})
        rest, more = strip_from(after[n + 1 :], 0)
        return head + rest, touched + [n + 1 + k for k in more]
    return strip_from(after, n)


def strip_from(msgs: list[Message], start: int) -> tuple[list[Message], list[int]]:
    """Remove thinking blocks from assistant messages at ``start`` and later.

    Returns the new list and the output positions of the messages that lost
    blocks. An assistant message left with nothing but thinking is dropped;
    its position is reported too.
    """
    out: list[Message] = msgs[:start]
    touched: list[int] = []
    for n in range(start, len(msgs)):
        msg = msgs[n]
        content = msg.get("content")
        if msg.get("role") != "assistant" or not isinstance(content, list):
            out.append(msg)
            continue
        kept = [b for b in content if b.get("type") not in THINKING_BLOCKS]
        if len(kept) == len(content):
            out.append(msg)
            continue
        touched.append(n)
        if kept:
            out.append({**msg, "content": kept})
    return out, touched


def check_policy(thinking: str) -> None:
    if thinking not in ("strip_after_change", "keep"):
        raise ValueError(f"thinking must be 'strip_after_change' or 'keep', got {thinking!r}")
