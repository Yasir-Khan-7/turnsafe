"""``validate()``: report every rule a chat history breaks, before the API does."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from . import _types as t
from ._match import Match, match
from ._providers import get_adapter, normalize
from ._types import Message, Report, Violation


def validate(messages: Iterable[Any], provider: str = "auto") -> Report:
    """Check a chat history against the provider's tool-call rules.

    Args:
        messages: The ``messages`` list you would send to the API. Dicts or
            SDK message objects.
        provider: ``"openai"`` (Chat Completions), ``"anthropic"`` (Messages)
            or ``"auto"`` to detect from the tool-call format.

    Returns:
        A ``Report``. It is truthy when there are no errors; call
        ``report.raise_for_errors()`` to turn errors into an exception.
    """
    msgs = normalize(messages)
    adapter = get_adapter(provider, msgs)
    return Report(adapter.name, tuple(_violations(adapter.name, adapter, msgs)))


def _violations(name: str, adapter: Any, msgs: Sequence[Message]) -> list[Violation]:
    m = match(adapter, msgs)
    out: list[Violation] = []

    for i, cid in m.duplicate_calls:
        out.append(
            Violation(
                t.DUPLICATE_TOOL_CALL_ID,
                f"tool call id {cid!r} appears more than once in this assistant turn",
                i,
                tool_call_id=cid,
            )
        )
    for i, ids in m.missing.items():
        for cid in ids:
            out.append(
                Violation(
                    t.UNANSWERED_TOOL_CALL,
                    f"tool call {cid!r} has no tool result"
                    + (" after it" if name == "openai" else " in the next user message"),
                    i,
                    tool_call_id=cid,
                )
            )
    for j, pos in sorted(m.misplaced):
        cid = m.result_ids[(j, pos)]
        where = (
            "in the tool messages directly after the call"
            if name == "openai"
            else "in the user message directly after the call"
        )
        out.append(
            Violation(
                t.MISPLACED_TOOL_RESULT,
                f"result for tool call {cid!r} must be {where}",
                j,
                tool_call_id=cid,
            )
        )
    for j, pos in sorted(m.duplicates):
        cid = m.result_ids[(j, pos)]
        out.append(
            Violation(
                t.DUPLICATE_TOOL_RESULT,
                f"tool call {cid!r} already has a result",
                j,
                tool_call_id=cid,
            )
        )
    for j, pos in sorted(m.orphans):
        cid = m.result_ids[(j, pos)]
        out.append(
            Violation(
                t.ORPHAN_TOOL_RESULT,
                f"tool result {cid!r} does not answer any call in the preceding assistant message",
                j,
                tool_call_id=cid,
            )
        )

    if name == "anthropic":
        out += _anthropic_rules(adapter, msgs, m)
    out.sort(key=lambda v: v.index)
    return out


def _anthropic_rules(adapter: Any, msgs: Sequence[Message], m: Match) -> list[Violation]:
    out: list[Violation] = []
    for i, msg in enumerate(msgs):
        role = msg.get("role")
        if role == "system":
            if i == 0:
                out.append(
                    Violation(
                        t.SYSTEM_IN_MESSAGES,
                        "a system message cannot open the conversation; pass it as the "
                        "top-level `system` parameter",
                        i,
                    )
                )
            else:
                out.append(
                    Violation(
                        t.SYSTEM_IN_MESSAGES,
                        "mid-conversation system messages work on Claude Opus 4.8+, Opus 5, "
                        "Fable and Mythos models, and are rejected by Sonnet 5 and older models",
                        i,
                        severity="warning",
                    )
                )
            continue
        if _has_empty(role, msg.get("content")):
            out.append(Violation(t.EMPTY_CONTENT, "empty user message or blank text block", i))

    # tool_use ids must be unique across the whole history. Repeats inside one
    # assistant turn are already reported by the matcher.
    first_turn: dict[str, int] = {}
    for start, ids in m.calls.items():
        for cid in ids:
            if cid in first_turn:
                out.append(
                    Violation(
                        t.DUPLICATE_TOOL_CALL_ID,
                        f"tool_use id {cid!r} was already used in messages[{first_turn[cid]}]; "
                        "ids must be unique",
                        start,
                        tool_call_id=cid,
                    )
                )
            else:
                first_turn[cid] = start
        zone = adapter.zone(msgs, m.runs[start][-1])
        if _results_not_first([msgs[j] for j in zone]):
            out.append(
                Violation(
                    t.TOOL_RESULT_NOT_FIRST,
                    "tool_result blocks must come before any other content in the user turn "
                    "(consecutive user messages count as one turn)",
                    zone[0],
                )
            )

    if msgs and msgs[0].get("role") not in ("user", "system"):
        out.append(
            Violation(
                t.FIRST_MESSAGE_NOT_USER,
                "the first message should have role 'user'; Bedrock and some deployments "
                "reject anything else",
                0,
                severity="warning",
            )
        )
    return out


def _results_not_first(zone_msgs: list[Message]) -> bool:
    seen_other = False
    for msg in zone_msgs:
        content = msg.get("content")
        blocks = content if isinstance(content, list) else [{"type": "text"}] if content else []
        for block in blocks:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                if seen_other:
                    return True
            else:
                seen_other = True
    return False


def _has_empty(role: Any, content: Any) -> bool:
    """What the API rejects: an empty user message, or a blank text block anywhere
    except inside a tool_result. Empty assistant content is accepted."""
    if isinstance(content, list) and any(_blank_text(b) for b in content):
        return True
    if role != "user":
        return False
    return content is None or content == [] or (isinstance(content, str) and not content.strip())


def _blank_text(block: Any) -> bool:
    return (
        isinstance(block, dict)
        and block.get("type") == "text"
        and not str(block.get("text", "")).strip()
    )
