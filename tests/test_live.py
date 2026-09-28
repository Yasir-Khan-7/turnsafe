"""Contract tests against the real APIs. Skipped unless TURNSAFE_LIVE=1.

Each case is a history the API rejects as-is. We assert that the raw history
gets a 400 (so the case really is broken) and that ``repair()`` output does not.

    TURNSAFE_LIVE=1 OPENAI_API_KEY=... ANTHROPIC_API_KEY=... pytest tests/test_live.py
"""

from __future__ import annotations

import os
from typing import Any

import pytest

import turnsafe as ts
from tests.builders import aa, au, oa, ot, ou, res, use

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("TURNSAFE_LIVE") != "1", reason="set TURNSAFE_LIVE=1"),
]

OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look something up.",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]
ANTHROPIC_TOOLS = [
    {
        "name": "lookup",
        "description": "Look something up.",
        "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
    }
]

OPENAI_BROKEN = {
    "orphan_after_trim": [ot("a"), ou("What did you find?")],
    "unanswered_call": [ou("look up x and y"), oa("a", "b"), ot("a"), ou("well?")],
    "user_between_call_and_result": [ou("look up x"), oa("a"), ou("hurry"), ot("a")],
}
ANTHROPIC_BROKEN = {
    "orphan_after_trim": [au(res("toolu_01")), aa("ok"), au("What did you find?")],
    "unanswered_call": [
        au("look up x and y"),
        aa(use("toolu_a"), use("toolu_b")),
        au(res("toolu_a")),
    ],
    "text_before_result": [au("look up x"), aa(use("toolu_a")), au("fyi", res("toolu_a"))],
}


def _openai(messages: list[dict[str, Any]]) -> None:
    openai = pytest.importorskip("openai")
    openai.OpenAI().chat.completions.create(
        model=os.environ.get("TURNSAFE_OPENAI_MODEL", "gpt-4o-mini"),
        messages=messages,
        tools=OPENAI_TOOLS,
        max_tokens=1,
    )


def _anthropic(messages: list[dict[str, Any]]) -> None:
    anthropic = pytest.importorskip("anthropic")
    anthropic.Anthropic().messages.create(
        model=os.environ.get("TURNSAFE_ANTHROPIC_MODEL", "claude-haiku-4-5"),
        messages=messages,
        tools=ANTHROPIC_TOOLS,
        max_tokens=1,
    )


@pytest.mark.parametrize("name", sorted(OPENAI_BROKEN))
def test_openai_accepts_repaired(name: str) -> None:
    openai = pytest.importorskip("openai")
    broken = OPENAI_BROKEN[name]
    with pytest.raises(openai.BadRequestError):
        _openai(broken)
    _openai(ts.repair(broken, "openai"))


@pytest.mark.parametrize("name", sorted(ANTHROPIC_BROKEN))
def test_anthropic_accepts_repaired(name: str) -> None:
    anthropic = pytest.importorskip("anthropic")
    broken = ANTHROPIC_BROKEN[name]
    with pytest.raises(anthropic.BadRequestError):
        _anthropic(broken)
    _anthropic(ts.repair(broken, "anthropic"))
