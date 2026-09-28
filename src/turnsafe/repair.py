"""``repair()``: rebuild a history that the provider will accept.

Guarantees (checked by the property tests):

* The input is never mutated.
* A history that already validates comes back unchanged (``repair(x) == x``).
* The output passes ``validate()``, except for a ``system`` message as the
  very first Anthropic message, which only the caller can move.
* ``repair(repair(x)) == repair(x)``.
* Assistant messages are never rewritten, unless ``on_missing="strip"`` is
  chosen, a tool call id is repeated, or ``thinking="strip_after_change"``
  removes thinking blocks after the first change (see ``_thinking.py``).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

from . import _thinking
from ._match import Loc, Match, match
from ._providers import Adapter, get_adapter, normalize
from ._thinking import ThinkingPolicy
from ._types import Fix, Message

DEFAULT_PLACEHOLDER = (
    "Error: this tool call was not executed, or its result was lost from the conversation "
    "history. Do not assume it succeeded."
)

OnMissing = Literal["placeholder", "strip"]


def repair(
    messages: Iterable[Any],
    provider: str = "auto",
    *,
    on_missing: OnMissing = "placeholder",
    placeholder: str = DEFAULT_PLACEHOLDER,
    thinking: ThinkingPolicy = "keep",
) -> list[Message]:
    """Return a copy of ``messages`` that satisfies the provider's tool-call rules.

    * Results that are in the wrong place are moved next to their call.
    * Results that answer no call, and repeated results, are dropped.
    * Calls with no result get a placeholder error result
      (``on_missing="placeholder"``, the default) or are removed from the
      assistant message (``on_missing="strip"``).
    * Anthropic only: tool_result blocks are moved before other content,
      repeated tool_use ids are renamed, and empty text blocks and empty
      messages are removed.
    * Anthropic only, ``thinking="strip_after_change"``: thinking blocks after
      the first change are removed. Claude models with thinking-block prefix
      binding (Fable 5.1 and Opus 5.5 on newer accounts) need this unless you
      set ``prefix_mismatch_behavior: "drop_block"``. The default, ``"keep"``,
      never removes thinking.

    Use ``repair_detailed()`` to also get the list of changes.
    """
    return repair_detailed(
        messages, provider, on_missing=on_missing, placeholder=placeholder, thinking=thinking
    )[0]


def repair_detailed(
    messages: Iterable[Any],
    provider: str = "auto",
    *,
    on_missing: OnMissing = "placeholder",
    placeholder: str = DEFAULT_PLACEHOLDER,
    thinking: ThinkingPolicy = "keep",
) -> tuple[list[Message], list[Fix]]:
    """Like ``repair()``, but also returns a ``Fix`` for every change made.

    ``Fix.index`` always refers to a position in the input list.
    """
    if not isinstance(placeholder, str) or not placeholder.strip():
        raise ValueError("placeholder must be a non-empty string")
    if on_missing not in ("placeholder", "strip"):
        raise ValueError(f"on_missing must be 'placeholder' or 'strip', got {on_missing!r}")
    _thinking.check_policy(thinking)
    original = normalize(messages)
    adapter = get_adapter(provider, original)
    fixes: list[Fix] = []
    if adapter.name == "anthropic":
        msgs, orig = _drop_empty(original, fixes)
    else:
        msgs, orig = original, list(range(len(original)))

    out, src = _rebuild(adapter, msgs, orig, on_missing, placeholder, fixes)

    if adapter.name == "anthropic" and thinking == "strip_after_change":
        out, touched = _thinking.strip_after_change(original, out, src)
        for n in touched:
            fixes.append(
                Fix("removed_thinking", "thinking is only valid before the first change", src[n])
            )
    fixes.sort(key=lambda f: f.index)
    return out, fixes


def _rebuild(
    adapter: Adapter,
    msgs: list[Message],
    orig: list[int],
    on_missing: OnMissing,
    placeholder: str,
    fixes: list[Fix],
) -> tuple[list[Message], list[int]]:
    """The repaired messages, plus the input index each one came from."""
    m = match(adapter, msgs)
    items: dict[Loc, Any] = {
        (j, pos): item
        for j, msg in enumerate(msgs)
        for pos, (_, item) in enumerate(adapter.result_items(msg))
    }
    taken = {cid for cids in m.calls.values() for cid in cids}
    seen: set[str] = set()
    next_suffix: dict[str, int] = {}
    # Precomputed once so each turn's checks are O(1), not O(history).
    dup_calls: dict[int, list[str]] = {}
    for idx, cid in m.duplicate_calls:
        dup_calls.setdefault(idx, []).append(cid)
    dirty = {j for j, _ in m.orphans | m.duplicates | m.misplaced} | set(dup_calls)

    out: list[Message] = []
    src: list[int] = []

    def emit(msg: Message, j: int) -> None:
        out.append(msg)
        src.append(orig[j])

    absorbed: set[int] = set()
    for i, msg in enumerate(msgs):
        if i in absorbed:
            continue
        if i not in m.calls:
            if adapter.result_items(msg):
                # Its results were placed with their call or are reported below as dropped.
                kept = adapter.without_results(msg)
                if kept is not None:
                    emit(kept, i)
            else:
                emit(msg, i)
            continue

        run = m.runs[i]
        zone = adapter.zone(msgs, run[-1])
        renames = _renames(adapter, m.calls[i], seen, taken, next_suffix)
        if _is_clean(adapter, msgs, m, i, zone, dirty) and not renames:
            for j in [*run, *zone]:
                emit(msgs[j], j)
            absorbed.update(run[1:] + zone)
            continue
        absorbed.update(run[1:] + zone)

        turn: list[tuple[int, Message]] = [(k, msgs[k]) for k in run]
        dup_ids = list(dict.fromkeys(dup_calls.get(i, [])))
        if dup_ids:
            turn = _dedupe_calls(adapter, turn)
            for cid in dup_ids:
                fixes.append(
                    Fix("removed_duplicate_call", f"repeated tool call {cid!r}", orig[i], cid)
                )

        answered = m.answered.get(i, {})
        placed = sorted(loc for loc in answered.values() if loc not in m.misplaced)
        by_loc = {loc: cid for cid, loc in answered.items()}
        group = [(by_loc[loc], items[loc]) for loc in placed]
        for cid in m.calls[i]:
            loc = answered.get(cid)
            if loc is not None and loc in m.misplaced:
                group.append((cid, items[loc]))
                fixes.append(
                    Fix(
                        "moved_result",
                        f"moved result for {cid!r} from messages[{orig[loc[0]]}]",
                        orig[i],
                        cid,
                    )
                )

        missing = m.missing.get(i, [])
        if missing and on_missing == "placeholder":
            for cid in missing:
                group.append((cid, adapter.placeholder(cid, placeholder)))
                fixes.append(Fix("added_placeholder", f"no result for {cid!r}", orig[i], cid))
        elif missing:
            for cid in missing:
                fixes.append(Fix("removed_call", f"no result for {cid!r}", orig[i], cid))
            stripped: list[tuple[int, Message]] = []
            for k, msg_k in turn:
                if not set(missing) & set(adapter.call_ids(msg_k)):
                    stripped.append((k, msg_k))
                    continue
                kept_msg = adapter.strip_calls(msg_k, set(missing))
                if kept_msg is None:
                    fixes.append(
                        Fix("dropped_message", "assistant message had nothing left", orig[k])
                    )
                else:
                    stripped.append((k, kept_msg))
            turn = stripped

        result_items = [item for _, item in group]
        if renames:
            for old, new in renames.items():
                turn = [(k, adapter.rename_call(msg_k, old, new)) for k, msg_k in turn]
                fixes.append(
                    Fix(
                        "renamed_duplicate_call",
                        f"{old!r} was already used; now {new!r}",
                        orig[i],
                        old,
                    )
                )
            result_items = [
                adapter.rename_result(item, renames[cid]) if cid in renames else item
                for cid, item in group
            ]

        zone_msgs = [msgs[j] for j in zone]
        if adapter.name == "anthropic" and _results_not_first([msgs[j] for j in zone]):
            fixes.append(Fix("reordered_blocks", "moved tool_result blocks first", orig[zone[0]]))

        first_zone = zone[0] if zone else run[-1]
        if not turn:
            for z in zone_msgs:
                kept = adapter.without_results(z)
                if kept is not None:
                    emit(kept, first_zone)
            continue
        emitted = adapter.emit_group([msg_k for _, msg_k in turn], result_items, zone_msgs)
        for k, msg_k in turn:
            emit(msg_k, k)
        for z in emitted[len(turn) :]:
            emit(z, first_zone)

    for loc in sorted(m.duplicates):
        fixes.append(
            Fix(
                "dropped_duplicate_result", "call already answered", orig[loc[0]], m.result_ids[loc]
            )
        )
    for loc in sorted(m.orphans):
        fixes.append(
            Fix(
                "dropped_orphan_result",
                "answers no preceding call",
                orig[loc[0]],
                m.result_ids[loc],
            )
        )
    return out, src


def _is_clean(
    adapter: Adapter, msgs: list[Message], m: Match, i: int, zone: list[int], dirty: set[int]
) -> bool:
    """True when the call group at ``i`` and its zone need no change at all.

    ``dirty`` holds the turns with repeated calls and the messages holding
    orphaned, duplicate or misplaced results (a misplaced result is moved to
    its own call, so the message it sits in has to be rebuilt).
    """
    if m.missing.get(i) or i in dirty:
        return False
    if any(loc in m.misplaced for loc in m.answered.get(i, {}).values()):
        return False
    if any(j in dirty for j in zone):
        return False
    return not (adapter.name == "anthropic" and _results_not_first([msgs[j] for j in zone]))


def _renames(
    adapter: Adapter,
    ids: list[str],
    seen: set[str],
    taken: set[str],
    next_suffix: dict[str, int],
) -> dict[str, str]:
    """New ids for calls whose id an earlier assistant message already used.

    Anthropic rejects repeated tool_use ids anywhere in the history. OpenAI
    histories keep theirs: some OpenAI-compatible servers restart ids each turn.
    """
    renames: dict[str, str] = {}
    for cid in ids:
        if adapter.name == "anthropic" and cid in seen:
            n = next_suffix.get(cid, 2)
            while f"{cid}_{n}" in taken:
                n += 1
            next_suffix[cid] = n + 1
            renames[cid] = f"{cid}_{n}"
            taken.add(renames[cid])
    seen.update(ids)
    seen.update(renames.values())
    return renames


def _dedupe_calls(adapter: Adapter, turn: list[tuple[int, Message]]) -> list[tuple[int, Message]]:
    """Keep only the first call for each id across the assistant turn."""
    seen: set[str] = set()

    def first(cid: Any) -> bool:
        if cid in seen:
            return False
        seen.add(cid)
        return True

    out: list[tuple[int, Message]] = []
    for k, msg in turn:
        if adapter.name == "openai":
            calls = [c for c in msg.get("tool_calls") or [] if first(c.get("id"))]
            out.append(
                (
                    k,
                    {**msg, "tool_calls": calls}
                    if calls
                    else {kk: v for kk, v in msg.items() if kk != "tool_calls"},
                )
            )
        else:
            content = msg.get("content")
            if isinstance(content, list):
                msg = {
                    **msg,
                    "content": [
                        b for b in content if b.get("type") != "tool_use" or first(b.get("id"))
                    ],
                }
            out.append((k, msg))
    return out


def _drop_empty(msgs: list[Message], fixes: list[Fix]) -> tuple[list[Message], list[int]]:
    """Anthropic rejects blank text blocks and empty user messages.

    Empty assistant content and blank ``tool_result`` content are accepted, so
    they are left alone. Returns the cleaned list and, for each kept message,
    its index in ``msgs``.
    """
    kept: list[Message] = []
    orig: list[int] = []
    for i, msg in enumerate(msgs):
        content = msg.get("content")
        if isinstance(content, list) and any(_blank_text(b) for b in content):
            fixes.append(Fix("removed_empty_text", "removed blank text block(s)", i))
            content = [b for b in content if not _blank_text(b)]
            msg = {**msg, "content": content}
        if msg.get("role") == "user" and (
            content is None or content == [] or (isinstance(content, str) and not content.strip())
        ):
            fixes.append(Fix("dropped_message", "empty user message", i))
            continue
        kept.append(msg)
        orig.append(i)
    return kept, orig


def _blank_text(block: Any) -> bool:
    return (
        isinstance(block, dict)
        and block.get("type") == "text"
        and not str(block.get("text", "")).strip()
    )


def _results_not_first(zone_msgs: list[Message]) -> bool:
    """True if, once the zone's user messages are merged, text precedes a tool_result."""
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
