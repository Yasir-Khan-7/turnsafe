"""Property tests: the guarantees turnsafe makes, checked on random broken histories.

Histories are generated valid, then damaged the ways real agents damage them:
naive trimming (dropping a prefix), losing messages (crashes, persistence
bugs), and reordering (interrupts, concurrent writes).
"""

from __future__ import annotations

import copy
from typing import Any

from hypothesis import assume, given, settings
from hypothesis import strategies as st

import turnsafe as ts
from tests.builders import aa, au, oa, osys, ot, ou, res, thinking, use
from turnsafe._thinking import THINKING_BLOCKS as THINKING
from turnsafe._thinking import first_change

Msg = dict[str, Any]


@st.composite
def openai_history(draw: st.DrawFn) -> list[Msg]:
    out: list[Msg] = [osys()] if draw(st.booleans()) else []
    n = 0
    for _ in range(draw(st.integers(1, 5))):
        out.append(ou(f"u{n}"))
        for _ in range(draw(st.integers(0, 3))):
            ids = [f"c{n}_{k}" for k in range(draw(st.integers(1, 3)))]
            n += 1
            out.append(oa(*ids))
            order = draw(st.permutations(ids))
            out.extend(ot(cid, f"r-{cid}") for cid in order)
        out.append(oa(text=f"a{n}"))
        n += 1
    return out


@st.composite
def anthropic_history(draw: st.DrawFn) -> list[Msg]:
    out: list[Msg] = []
    n = 0
    for _ in range(draw(st.integers(1, 5))):
        out.append(au(f"u{n}"))
        for _ in range(draw(st.integers(0, 3))):
            ids = [f"t{n}_{k}" for k in range(draw(st.integers(1, 3)))]
            n += 1
            head = [thinking(f"th{n}")] if draw(st.booleans()) else []
            blocks = [*head, *(use(cid) for cid in ids)]
            cut = draw(st.integers(0, len(blocks) - 1))
            if cut:
                # Consecutive assistant messages: legal, the API merges them.
                out.extend([aa(*blocks[:cut]), aa(*blocks[cut:])])
            else:
                out.append(aa(*blocks))
            results = [res(cid, f"r-{cid}") for cid in draw(st.permutations(ids))]
            tail = ["and then?"] if draw(st.booleans()) else []
            if draw(st.booleans()):
                out.append(au(*results, *tail))
            else:
                # One user message per result: legal, the API merges them.
                out.extend(au(r) for r in results)
                if tail:
                    out.append(au(*tail))
        out.append(aa(f"a{n}"))
        n += 1
    return out


@st.composite
def damaged(draw: st.DrawFn, base: st.SearchStrategy[list[Msg]]) -> list[Msg]:
    h = draw(base)
    for _ in range(draw(st.integers(0, 4))):
        if not h:
            break
        op = draw(st.sampled_from(["trim", "delete", "move", "dup"]))
        i = draw(st.integers(0, len(h) - 1))
        if op == "trim":
            h = h[i:]
        elif op == "delete":
            del h[i]
        elif op == "move":
            h.insert(draw(st.integers(0, len(h) - 1)), h.pop(i))
        else:
            h.insert(i, copy.deepcopy(h[i]))
    return h


any_openai = damaged(openai_history())
any_anthropic = damaged(anthropic_history())
PROVIDERS = {"openai": any_openai, "anthropic": any_anthropic}

# repair() leaves these to the caller; see docs/rules.md.
NOT_REPAIRABLE = {ts.SYSTEM_IN_MESSAGES}


def errors(h: list[Msg], provider: str) -> list[str]:
    return [v.code for v in ts.validate(h, provider).errors if v.code not in NOT_REPAIRABLE]


@settings(max_examples=400)
@given(
    st.sampled_from(["openai", "anthropic"]).flatmap(lambda p: st.tuples(st.just(p), PROVIDERS[p]))
)
def test_repair_output_always_validates(case: tuple[str, list[Msg]]) -> None:
    provider, h = case
    for policy in ("placeholder", "strip"):
        out = ts.repair(h, provider, on_missing=policy)  # type: ignore[arg-type]
        assert errors(out, provider) == [], (policy, h, out)


@settings(max_examples=300)
@given(
    st.sampled_from(["openai", "anthropic"]).flatmap(lambda p: st.tuples(st.just(p), PROVIDERS[p]))
)
def test_repair_is_idempotent_and_pure(case: tuple[str, list[Msg]]) -> None:
    provider, h = case
    before = copy.deepcopy(h)
    once = ts.repair(h, provider)
    assert h == before
    assert ts.repair(once, provider) == once


@settings(max_examples=200)
@given(st.one_of(openai_history(), anthropic_history()))
def test_valid_history_is_a_fixed_point(h: list[Msg]) -> None:
    assert ts.validate(h).violations == ()
    out, fixes = ts.repair_detailed(h)
    assert out == h and fixes == []


def without_thinking(msg: Msg) -> Msg:
    content = msg.get("content")
    if msg.get("role") != "assistant" or not isinstance(content, list):
        return msg
    return {**msg, "content": [b for b in content if b.get("type") not in THINKING]}


def has_thinking(msg: Msg) -> bool:
    content = msg.get("content")
    return isinstance(content, list) and any(b.get("type") in THINKING for b in content)


def unique_call_ids(h: list[Msg]) -> bool:
    ids = [
        b["id"]
        for m in h
        if m.get("role") == "assistant" and isinstance(m.get("content"), list)
        for b in m["content"]
        if b.get("type") == "tool_use"
    ]
    return len(ids) == len(set(ids))


@settings(max_examples=300)
@given(any_anthropic)
def test_repair_only_removes_thinking_and_only_after_the_first_change(h: list[Msg]) -> None:
    """Before the first change assistant messages are untouched, signatures included.
    After it, the only edit to an assistant message is dropping its thinking blocks,
    which Claude models with thinking-block prefix binding require."""
    assume(unique_call_ids(h))
    originals = [without_thinking(m) for m in h if m.get("role") == "assistant"]
    out = ts.repair(h, "anthropic", thinking="strip_after_change")
    start = first_change(h, out)
    for n, m in enumerate(out):
        if m.get("role") != "assistant":
            continue
        assert without_thinking(m) in originals
        if n > start:
            assert not has_thinking(m)


@settings(max_examples=300)
@given(any_anthropic)
def test_repair_by_default_never_edits_assistant_messages(h: list[Msg]) -> None:
    assume(unique_call_ids(h))
    originals = [m for m in h if m.get("role") == "assistant"]
    for m in ts.repair(h, "anthropic"):
        if m.get("role") == "assistant":
            assert m in originals


@settings(max_examples=300)
@given(st.one_of(openai_history(), anthropic_history()), st.integers(1, 60))
def test_fit_output_is_valid_and_within_budget(h: list[Msg], budget: int) -> None:
    def one(_: Any) -> int:
        return 1

    try:
        result = ts.fit_detailed(h, budget, counter=one)
    except ts.ContextBudgetError:
        return
    assert result.tokens == len(result.messages) <= budget
    assert ts.validate(result.messages).errors == ()
    # What is kept is the system prefix plus a contiguous tail of the history.
    body = [m for m in result.messages if m.get("role") != "system"]
    assert body == h[len(h) - len(body) :]


@settings(max_examples=200)
@given(anthropic_history(), st.integers(1, 60))
def test_fit_strip_after_change_removes_thinking_only_when_dropping(
    h: list[Msg], budget: int
) -> None:
    try:
        result = ts.fit_detailed(h, budget, counter=lambda _: 1, thinking="strip_after_change")
    except ts.ContextBudgetError:
        return
    if result.dropped_messages:
        # A kept tail of the input, minus thinking blocks; an assistant message
        # that held nothing but thinking disappears.
        stripped = [
            without_thinking(m)
            for m in h
            if not (m.get("role") == "assistant" and without_thinking(m).get("content") == [])
        ]
        assert result.messages == stripped[len(stripped) - len(result.messages) :]
        assert not any(has_thinking(m) for m in result.messages)
    else:
        assert result.messages == h
    assert ts.validate(result.messages).errors == ()


@settings(max_examples=200)
@given(
    st.sampled_from(["openai", "anthropic"]).flatmap(lambda p: st.tuples(st.just(p), PROVIDERS[p])),
    st.integers(1, 80),
)
def test_repair_then_fit_always_works(case: tuple[str, list[Msg]], budget: int) -> None:
    provider, h = case
    fixed = ts.repair(h, provider)
    if errors(fixed, provider):
        return
    try:
        out = ts.fit(fixed, budget, provider, counter=lambda _: 1)
    except ts.ContextBudgetError:
        return
    assert errors(out, provider) == []
