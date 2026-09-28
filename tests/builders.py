"""Tiny constructors for provider-shaped messages, used across the tests."""

from __future__ import annotations

from typing import Any


# OpenAI Chat Completions
def ou(text: str = "hi") -> dict[str, Any]:
    return {"role": "user", "content": text}


def osys(text: str = "be helpful") -> dict[str, Any]:
    return {"role": "system", "content": text}


def oa(*call_ids: str, text: str | None = None) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": "assistant", "content": text}
    if call_ids:
        msg["tool_calls"] = [
            {"id": cid, "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
            for cid in call_ids
        ]
    return msg


def ot(call_id: str, text: str = "result") -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": call_id, "content": text}


# Anthropic Messages
def au(*blocks: Any) -> dict[str, Any]:
    if len(blocks) == 1 and isinstance(blocks[0], str):
        return {"role": "user", "content": blocks[0]}
    return {"role": "user", "content": [_block(b) for b in blocks] or "hi"}


def aa(*blocks: Any) -> dict[str, Any]:
    if len(blocks) == 1 and isinstance(blocks[0], str):
        return {"role": "assistant", "content": blocks[0]}
    return {"role": "assistant", "content": [_block(b) for b in blocks]}


def use(call_id: str) -> dict[str, Any]:
    return {"type": "tool_use", "id": call_id, "name": "lookup", "input": {"q": call_id}}


def res(call_id: str, text: str = "result") -> dict[str, Any]:
    return {"type": "tool_result", "tool_use_id": call_id, "content": text}


def thinking(text: str = "let me think") -> dict[str, Any]:
    return {"type": "thinking", "thinking": text, "signature": "sig-" + text}


def _block(b: Any) -> Any:
    return {"type": "text", "text": b} if isinstance(b, str) else b
