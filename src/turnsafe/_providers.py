"""Per-provider knowledge: where tool calls and tool results live in a message.

Everything format-specific is here. ``validate``, ``repair`` and ``fit`` only
talk to the ``Adapter`` interface, so adding a provider means adding a class.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from ._types import Message, Provider, ProviderDetectionError


def to_dict(obj: Any, what: str = "message") -> dict[str, Any]:
    """Turn a message or block into a plain dict.

    Accepts mappings and SDK objects with ``model_dump`` (openai / anthropic
    pydantic models). ``None`` fields are dropped, which both APIs accept.
    """
    if isinstance(obj, Mapping):
        return dict(obj)
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        result = dump(exclude_none=True)
        if isinstance(result, dict):
            return result
    raise TypeError(
        f"Expected a {what} dict (or an SDK object with model_dump()), got {type(obj).__name__}"
    )


def normalize(messages: Iterable[Any]) -> list[Message]:
    """Deep-copy the history into plain dicts so callers' objects are never mutated.

    Tuples become lists, and tool call ids are checked: a call or result
    without a usable id would silently escape every rule.
    """
    if isinstance(messages, (str, bytes, Mapping)):
        raise TypeError("messages must be a list of message dicts")
    out: list[Message] = []
    for i, raw in enumerate(messages):
        msg = copy.deepcopy(to_dict(raw))
        content = msg.get("content")
        if isinstance(content, (list, tuple)):
            msg["content"] = [to_dict(block, "content block") for block in content]
            for k, block in enumerate(msg["content"]):
                if block.get("type") in ("tool_use", "server_tool_use", "mcp_tool_use"):
                    _check_id(block.get("id"), f"messages[{i}].content[{k}].id")
                elif block.get("type") == "tool_result":
                    _check_id(block.get("tool_use_id"), f"messages[{i}].content[{k}].tool_use_id")
        calls = msg.get("tool_calls")
        if isinstance(calls, (list, tuple)):
            msg["tool_calls"] = [to_dict(call, "tool call") for call in calls]
            for k, call in enumerate(msg["tool_calls"]):
                _check_id(call.get("id"), f"messages[{i}].tool_calls[{k}].id")
        if msg.get("role") == "tool":
            _check_id(msg.get("tool_call_id"), f"messages[{i}].tool_call_id")
        out.append(msg)
    return out


def _check_id(value: Any, where: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{where} must be a non-empty string, got {value!r}")


class Adapter:
    """How one provider's message format stores tool calls and results.

    A "result item" is whatever unit carries one tool result: a whole
    ``role: tool`` message for OpenAI, a ``tool_result`` block for Anthropic.
    """

    name: Provider
    # Whether the API treats consecutive assistant messages as one turn.
    merges_assistant_runs = False

    def call_ids(self, msg: Message) -> list[str]:
        """Ids of client tool calls made by this message (assistant only)."""
        raise NotImplementedError

    def result_items(self, msg: Message) -> list[tuple[str, Any]]:
        """(tool_call_id, item) for each tool result carried by this message."""
        raise NotImplementedError

    def zone(self, messages: Sequence[Message], i: int) -> list[int]:
        """Indices where results for the calls in ``messages[i]`` are allowed."""
        raise NotImplementedError

    def is_turn_start(self, msg: Message) -> bool:
        """True for a user message that starts a new turn (not just tool results)."""
        raise NotImplementedError

    def placeholder(self, tool_call_id: str, text: str) -> Any:
        """A synthetic result item for a call whose real result is missing."""
        raise NotImplementedError

    def strip_calls(self, msg: Message, ids: set[str]) -> Message | None:
        """``msg`` without the given calls, or None if nothing meaningful is left."""
        raise NotImplementedError

    def emit_group(
        self, assistants: list[Message], items: list[Any], zone_msgs: list[Message]
    ) -> list[Message]:
        """The assistant turn followed by messages that carry ``items``.

        ``zone_msgs`` are the input messages in the assistant's zone. Their
        result items are replaced by ``items``; everything else they hold,
        content and keys, must be kept.
        """
        raise NotImplementedError

    def rename_call(self, msg: Message, old: str, new: str) -> Message:
        """``msg`` with the tool call ``old`` renamed to ``new``."""
        raise NotImplementedError

    def rename_result(self, item: Any, new: str) -> Any:
        """A copy of result ``item`` answering ``new`` instead."""
        raise NotImplementedError

    def without_results(self, msg: Message) -> Message | None:
        """``msg`` with every tool result removed, or None if it becomes empty."""
        raise NotImplementedError


class OpenAIAdapter(Adapter):
    """OpenAI Chat Completions (``role: tool`` messages answer ``tool_calls``)."""

    name: Provider = "openai"

    def call_ids(self, msg: Message) -> list[str]:
        if msg.get("role") != "assistant":
            return []
        calls = msg.get("tool_calls") or []
        return [c["id"] for c in calls if isinstance(c, dict) and isinstance(c.get("id"), str)]

    def result_items(self, msg: Message) -> list[tuple[str, Any]]:
        if msg.get("role") == "tool":
            return [(str(msg.get("tool_call_id", "")), msg)]
        return []

    def zone(self, messages: Sequence[Message], i: int) -> list[int]:
        out = []
        j = i + 1
        while j < len(messages) and messages[j].get("role") == "tool":
            out.append(j)
            j += 1
        return out

    def is_turn_start(self, msg: Message) -> bool:
        return msg.get("role") == "user"

    def placeholder(self, tool_call_id: str, text: str) -> Any:
        return {"role": "tool", "tool_call_id": tool_call_id, "content": text}

    def strip_calls(self, msg: Message, ids: set[str]) -> Message | None:
        out = dict(msg)
        kept = [c for c in msg.get("tool_calls") or [] if c.get("id") not in ids]
        if kept:
            out["tool_calls"] = kept
            return out
        out.pop("tool_calls", None)
        out.pop("parallel_tool_calls", None)
        if _is_empty(out.get("content")) and not out.get("audio") and not out.get("refusal"):
            return None
        return out

    def emit_group(
        self, assistants: list[Message], items: list[Any], zone_msgs: list[Message]
    ) -> list[Message]:
        # The zone only ever holds tool messages, which are all in ``items`` or dropped.
        return [*assistants, *items]

    def rename_call(self, msg: Message, old: str, new: str) -> Message:
        calls = [{**c, "id": new} if c.get("id") == old else c for c in msg["tool_calls"]]
        return {**msg, "tool_calls": calls}

    def rename_result(self, item: Any, new: str) -> Any:
        return {**item, "tool_call_id": new}

    def without_results(self, msg: Message) -> Message | None:
        return None if msg.get("role") == "tool" else msg


# Anthropic blocks that pair with a later client-side tool_result. Server tool
# blocks (server_tool_use, web_search_tool_result, mcp_tool_use, ...) are
# answered inside the same assistant message, so they are left alone.
_ANTHROPIC_CALL = "tool_use"
_ANTHROPIC_RESULT = "tool_result"


def _blocks(msg: Message) -> list[Any]:
    content = msg.get("content")
    return content if isinstance(content, list) else []


class AnthropicAdapter(Adapter):
    """Anthropic Messages (``tool_result`` blocks answer ``tool_use`` blocks)."""

    name: Provider = "anthropic"
    # "Consecutive user or assistant turns in your request will be combined
    # into a single turn." https://platform.claude.com/docs/en/api/messages
    merges_assistant_runs = True

    def call_ids(self, msg: Message) -> list[str]:
        if msg.get("role") != "assistant":
            return []
        return [
            b["id"]
            for b in _blocks(msg)
            if isinstance(b, dict)
            and b.get("type") == _ANTHROPIC_CALL
            and isinstance(b.get("id"), str)
        ]

    def result_items(self, msg: Message) -> list[tuple[str, Any]]:
        if msg.get("role") != "user":
            return []
        return [
            (str(b.get("tool_use_id", "")), b)
            for b in _blocks(msg)
            if isinstance(b, dict) and b.get("type") == _ANTHROPIC_RESULT
        ]

    def zone(self, messages: Sequence[Message], i: int) -> list[int]:
        # The API merges consecutive user messages into one turn before
        # checking, so results may be spread over the whole run.
        out = []
        j = i + 1
        while j < len(messages) and messages[j].get("role") == "user":
            out.append(j)
            j += 1
        return out

    def is_turn_start(self, msg: Message) -> bool:
        return msg.get("role") == "user" and not self.result_items(msg)

    def placeholder(self, tool_call_id: str, text: str) -> Any:
        return {
            "type": "tool_result",
            "tool_use_id": tool_call_id,
            "content": text,
            "is_error": True,
        }

    def strip_calls(self, msg: Message, ids: set[str]) -> Message | None:
        kept = [
            b for b in _blocks(msg) if not (b.get("type") == _ANTHROPIC_CALL and b.get("id") in ids)
        ]
        # Thinking blocks alone are not a meaningful assistant turn.
        if not any(b.get("type") not in ("thinking", "redacted_thinking") for b in kept):
            return None
        return {**msg, "content": kept}

    def emit_group(
        self, assistants: list[Message], items: list[Any], zone_msgs: list[Message]
    ) -> list[Message]:
        # One user message: every tool_result first, then whatever else the zone
        # held, in order. Other keys from every zone message are kept (first wins).
        if not items:
            kept = [self.without_results(z) for z in zone_msgs]
            return [*assistants, *(z for z in kept if z is not None)]
        base: Message = {}
        rest: list[Any] = []
        for msg in zone_msgs:
            for key, value in msg.items():
                if key != "content":
                    base.setdefault(key, value)
            content = msg.get("content")
            if isinstance(content, str):
                rest += [{"type": "text", "text": content}] if content else []
            elif isinstance(content, list):
                rest += [b for b in content if b.get("type") != _ANTHROPIC_RESULT]
        base.setdefault("role", "user")
        return [*assistants, {**base, "content": [*items, *rest]}]

    def rename_call(self, msg: Message, old: str, new: str) -> Message:
        content = [
            {**b, "id": new} if b.get("type") == _ANTHROPIC_CALL and b.get("id") == old else b
            for b in _blocks(msg)
        ]
        return {**msg, "content": content}

    def rename_result(self, item: Any, new: str) -> Any:
        return {**item, "tool_use_id": new}

    def without_results(self, msg: Message) -> Message | None:
        if not self.result_items(msg):
            return msg
        kept = [b for b in _blocks(msg) if b.get("type") != _ANTHROPIC_RESULT]
        if not kept:
            return None
        return {**msg, "content": kept}


def _is_empty(content: Any) -> bool:
    return content is None or content == "" or content == []


ADAPTERS: dict[str, Adapter] = {"openai": OpenAIAdapter(), "anthropic": AnthropicAdapter()}

_OPENAI_ROLES = {"tool", "developer", "function"}
_ANTHROPIC_BLOCKS = {
    "tool_use",
    "tool_result",
    "thinking",
    "redacted_thinking",
    "server_tool_use",
    "web_search_tool_result",
    "mcp_tool_use",
    "mcp_tool_result",
}


def detect(messages: Sequence[Message]) -> Provider:
    """Guess the message format from tool-call markers.

    Plain text histories with no markers are reported as ``"openai"``; the
    two formats validate identically when there are no tools involved, except
    for Anthropic's content rules, so pass ``provider=`` when you know it.
    """
    # "system" is deliberately not an OpenAI signal: in an Anthropic history it
    # is a mistake that validate() should report, not a reason to give up.
    openai = anthropic = False
    for msg in messages:
        if (
            msg.get("role") in _OPENAI_ROLES
            or msg.get("tool_calls")
            or msg.get("tool_call_id") is not None
        ):
            openai = True
        if any(isinstance(b, dict) and b.get("type") in _ANTHROPIC_BLOCKS for b in _blocks(msg)):
            anthropic = True
    if openai and anthropic:
        raise ProviderDetectionError(
            "History mixes OpenAI and Anthropic tool formats; pass provider= explicitly."
        )
    return "anthropic" if anthropic else "openai"


def get_adapter(provider: str, messages: Sequence[Message]) -> Adapter:
    if provider == "auto":
        provider = detect(messages)
    try:
        return ADAPTERS[provider]
    except KeyError:
        raise ValueError(
            f"Unknown provider {provider!r}; expected 'openai', 'anthropic' or 'auto'."
        ) from None
