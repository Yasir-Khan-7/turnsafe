"""``fit()``: trim a history to a token budget without ever splitting a tool call
from its result.

The history is cut into turns. A turn starts at a user message that is not
just a carrier of tool results and runs until the next such message, so a
tool call and all of its results always land in the same turn. ``fit()``
drops whole turns, oldest first, and the result therefore passes
``validate()`` whenever the input did.

Dropping turns changes the conversation prefix, which invalidates replayed
Anthropic thinking blocks on models with prefix binding; see ``_thinking.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from . import _thinking
from ._providers import Adapter, get_adapter, normalize
from ._thinking import ThinkingPolicy
from ._types import ContextBudgetError, Message, Report
from .tokens import approx_tokens
from .validate import _violations

Counter = Callable[[Mapping[str, Any]], int]


@dataclass(frozen=True)
class FitResult:
    messages: list[Message]
    tokens: int
    """Estimated tokens of ``messages`` according to the counter used."""
    dropped_messages: int
    dropped_turns: int
    removed_thinking: int = 0
    """Assistant messages that had thinking blocks removed (Anthropic only)."""


def fit(
    messages: Iterable[Any],
    max_tokens: int,
    provider: str = "auto",
    *,
    counter: Counter = approx_tokens,
    keep_system: bool = True,
    keep_first_turn: bool = False,
    thinking: ThinkingPolicy = "keep",
) -> list[Message]:
    """Drop the oldest whole turns until the history fits in ``max_tokens``.

    Args:
        messages: A valid history. Run ``repair()`` first if it may not be.
        max_tokens: Budget for the messages, as measured by ``counter``.
            Leave room for the tool definitions, system prompt (Anthropic) and
            the model's reply yourself.
        provider: ``"openai"``, ``"anthropic"`` or ``"auto"``.
        counter: Tokens for one message. Defaults to an offline estimate.
        keep_system: Always keep leading ``system``/``developer`` messages (OpenAI;
            an Anthropic history must not start with a system message).
        keep_first_turn: Always keep the first turn started by a user message,
            which usually states the task.
        thinking: Anthropic only. ``"keep"`` (default) never touches thinking
            blocks. ``"strip_after_change"`` removes them from the kept messages
            when any turn is dropped, which Claude models with thinking-block
            prefix binding (Fable 5.1 and Opus 5.5 on newer accounts) need unless
            you set ``prefix_mismatch_behavior: "drop_block"``.

    Raises:
        InvalidHistoryError: the input breaks tool-call rules.
        ContextBudgetError: the messages that must be kept are already over budget.
    """
    return fit_detailed(
        messages,
        max_tokens,
        provider,
        counter=counter,
        keep_system=keep_system,
        keep_first_turn=keep_first_turn,
        thinking=thinking,
    ).messages


def fit_detailed(
    messages: Iterable[Any],
    max_tokens: int,
    provider: str = "auto",
    *,
    counter: Counter = approx_tokens,
    keep_system: bool = True,
    keep_first_turn: bool = False,
    thinking: ThinkingPolicy = "keep",
) -> FitResult:
    """Like ``fit()``, but also returns the token estimate and what was dropped."""
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0:
        raise ValueError(f"max_tokens must be a positive int, got {max_tokens!r}")
    _thinking.check_policy(thinking)
    msgs = normalize(messages)
    adapter = get_adapter(provider, msgs)
    Report(adapter.name, tuple(_violations(adapter.name, adapter, msgs))).raise_for_errors()

    def count(msg: Message) -> int:
        n = counter(msg)
        if not isinstance(n, int) or isinstance(n, bool) or n < 0:
            raise TypeError(f"counter must return a non-negative int, got {n!r}")
        return n

    pinned: list[Message] = []
    if keep_system:
        while len(pinned) < len(msgs) and msgs[len(pinned)].get("role") in ("system", "developer"):
            pinned.append(msgs[len(pinned)])
    turns = split_turns(adapter, msgs[len(pinned) :])
    pinned_cost = sum(count(m) for m in pinned)
    if not turns:
        if pinned_cost > max_tokens:
            raise ContextBudgetError(pinned_cost, max_tokens)
        return FitResult(pinned, pinned_cost, 0, 0)

    cost = [sum(count(m) for m in turn) for turn in turns]
    budget = max_tokens - pinned_cost
    if sum(cost) <= budget:
        return FitResult(
            pinned + [m for turn in turns for m in turn], pinned_cost + sum(cost), 0, 0
        )

    # keep_first_turn pins the first turn the user started (the task), not a
    # greeting or other messages that came before it.
    pin = None
    if keep_first_turn:
        pin = next((n for n, turn in enumerate(turns) if adapter.is_turn_start(turn[0])), None)
        if pin == len(turns) - 1:
            pin = None  # the latest turn is always kept anyway
    if pin is not None:
        budget -= cost[pin]

    keep_from = len(turns) - 1
    if cost[keep_from] > budget:
        raise ContextBudgetError(max_tokens - budget + cost[keep_from], max_tokens)
    used = cost[keep_from]
    lowest = pin + 1 if pin is not None else 0
    while keep_from - 1 >= lowest and used + cost[keep_from - 1] <= budget:
        keep_from -= 1
        used += cost[keep_from]

    kept_turns = ([turns[pin]] if pin is not None else []) + turns[keep_from:]
    out = pinned + [m for turn in kept_turns for m in turn]
    dropped = len(msgs) - len(out)
    touched: list[int] = []
    if adapter.name == "anthropic" and thinking == "strip_after_change" and dropped:
        out, touched = _thinking.strip_from(out, _thinking.first_change(msgs, out))
    return FitResult(
        messages=out,
        tokens=sum(count(m) for m in out),
        dropped_messages=len(msgs) - len(out),
        dropped_turns=len(turns) - len(kept_turns),
        removed_thinking=len(touched),
    )


def split_turns(adapter: Adapter, msgs: list[Message]) -> list[list[Message]]:
    """Split into turns. Anything before the first user message is its own turn."""
    turns: list[list[Message]] = []
    for msg in msgs:
        if not turns or adapter.is_turn_start(msg):
            turns.append([msg])
        else:
            turns[-1].append(msg)
    return turns
