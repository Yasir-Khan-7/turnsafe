"""Pair every tool call with its tool result.

``validate`` reports from this and ``repair`` rebuilds from it, so the two can
never disagree about which result answers which call.

Matching rule: a call made by an assistant turn (one message, or for
Anthropic a run of consecutive assistant messages, which the API merges) is
answered by the first result with its id that comes after the turn and before
the next turn that calls the same id again. A result inside the provider's
allowed zone (see ``Adapter.zone``) is correctly placed; one elsewhere is
misplaced but recoverable. Further results for an answered call are
duplicates, and results that answer nothing are orphans.

Every result and call is visited a constant number of times, so matching is
linear in the size of the history.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from ._providers import Adapter
from ._types import Message

# A result item's location: (message index, position among that message's results).
Loc = tuple[int, int]


@dataclass
class Match:
    # assistant turn start -> call ids in order, duplicates removed
    calls: dict[int, list[str]] = field(default_factory=dict)
    # assistant turn start -> indices of the assistant messages in that turn
    runs: dict[int, list[int]] = field(default_factory=dict)
    # assistant index -> call id -> result location
    answered: dict[int, dict[str, Loc]] = field(default_factory=dict)
    # results that answer a call but sit outside the allowed zone
    misplaced: set[Loc] = field(default_factory=set)
    # assistant index -> call ids with no result anywhere in the window
    missing: dict[int, list[str]] = field(default_factory=dict)
    # (assistant index, call id) for ids repeated inside one message
    duplicate_calls: list[tuple[int, str]] = field(default_factory=list)
    # locations of second and later results for an already-answered call
    duplicates: set[Loc] = field(default_factory=set)
    # locations of results that answer nothing
    orphans: set[Loc] = field(default_factory=set)
    # result ids by location, for reporting
    result_ids: dict[Loc, str] = field(default_factory=dict)


def match(adapter: Adapter, msgs: Sequence[Message]) -> Match:
    m = Match()
    results_by_id: dict[str, list[Loc]] = {}
    for j, msg in enumerate(msgs):
        for pos, (rid, _) in enumerate(adapter.result_items(msg)):
            m.result_ids[(j, pos)] = rid
            results_by_id.setdefault(rid, []).append((j, pos))

    runs: list[list[int]] = []
    for i, msg in enumerate(msgs):
        if msg.get("role") != "assistant":
            continue
        if runs and runs[-1][-1] == i - 1 and adapter.merges_assistant_runs:
            runs[-1].append(i)
        else:
            runs.append([i])

    turn_ids: list[list[str]] = []
    calls_by_id: dict[str, list[int]] = {}
    for n, run in enumerate(runs):
        ids: list[str] = []
        seen: set[str] = set()
        for k in run:
            for cid in adapter.call_ids(msgs[k]):
                if cid in seen:
                    m.duplicate_calls.append((run[0], cid))
                else:
                    seen.add(cid)
                    ids.append(cid)
                    calls_by_id.setdefault(cid, []).append(n)
        turn_ids.append(ids)

    claimed: set[Loc] = set()
    result_ptr: dict[str, int] = {}
    call_ptr: dict[str, int] = {}
    for n, run in enumerate(runs):
        ids = turn_ids[n]
        if not ids:
            continue
        i, end = run[0], run[-1]
        m.calls[i] = ids
        m.runs[i] = run
        zone = set(adapter.zone(msgs, end))
        answered: dict[str, Loc] = {}
        for cid in ids:
            # Stop at the next turn that reuses this id: its results are its own.
            call_ptr[cid] = call_ptr.get(cid, 0) + 1
            later = calls_by_id[cid][call_ptr[cid] :]
            limit = runs[later[0]][0] if later else len(msgs)

            found = results_by_id.get(cid, [])
            k = result_ptr.get(cid, 0)
            while k < len(found) and found[k][0] <= end:
                k += 1  # before the call: left as orphans
            locs = []
            while k < len(found) and found[k][0] < limit:
                locs.append(found[k])
                k += 1
            result_ptr[cid] = k
            if not locs:
                m.missing.setdefault(i, []).append(cid)
                continue
            # Prefer a correctly placed result over a misplaced one.
            best = next((loc for loc in locs if loc[0] in zone), locs[0])
            answered[cid] = best
            if best[0] not in zone:
                m.misplaced.add(best)
            for loc in locs:
                claimed.add(loc)
                if loc != best:
                    m.duplicates.add(loc)
        m.answered[i] = answered

    m.orphans = {loc for loc in m.result_ids if loc not in claimed}
    return m
