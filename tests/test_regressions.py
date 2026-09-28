"""Regressions for bugs found by adversarial testing before v0.1.0."""

import time

import pytest

import turnsafe as ts
from tests.builders import aa, au, oa, osys, ot, ou, res, use


@pytest.mark.parametrize("bad_id", [1, None, ""])
def test_unusable_openai_call_id_is_rejected(bad_id: object) -> None:
    msg = oa("x")
    msg["tool_calls"][0]["id"] = bad_id
    with pytest.raises(ValueError, match=r"messages\[1\]\.tool_calls\[0\]\.id"):
        ts.validate([ou(), msg])


def test_missing_openai_call_id_is_rejected() -> None:
    msg = oa("x")
    del msg["tool_calls"][0]["id"]
    with pytest.raises(ValueError, match="non-empty string"):
        ts.repair([ou(), msg])


@pytest.mark.parametrize("bad_id", [7, None])
def test_unusable_anthropic_ids_are_rejected(bad_id: object) -> None:
    block = use("x")
    block["id"] = bad_id
    with pytest.raises(ValueError, match=r"content\[0\]\.id"):
        ts.validate([au("q"), aa(block)])
    result = res("x")
    result["tool_use_id"] = bad_id
    with pytest.raises(ValueError, match="tool_use_id"):
        ts.validate([au("q"), aa(use("x")), au(result)])


def test_non_dict_block_error_names_the_block() -> None:
    with pytest.raises(TypeError, match="content block"):
        ts.validate([{"role": "user", "content": ["just a string"]}])


def test_messages_must_be_a_list() -> None:
    with pytest.raises(TypeError, match="list of message dicts"):
        ts.validate({"role": "user", "content": "hi"})  # type: ignore[arg-type]


def test_tuple_content_is_seen() -> None:
    msg = {"role": "assistant", "content": (use("a"),)}
    report = ts.validate([au("q"), msg])
    assert [v.code for v in report.errors] == [ts.UNANSWERED_TOOL_CALL]


def test_unknown_keys_on_result_only_user_message_survive() -> None:
    h = [au("hi"), aa(use("a"), use("b")), {**au(res("a", "x")), "meta": "KEEP"}]
    out = ts.repair(h)
    assert out[2]["meta"] == "KEEP"


def test_unknown_keys_on_every_merged_user_message_survive() -> None:
    h = [au("hi"), aa(use("a")), {**au("t"), "m1": 1}, {**au(res("a")), "m2": 2}]
    out = ts.repair(h)
    assert out[2]["m1"] == 1 and out[2]["m2"] == 2


def test_keep_first_turn_pins_the_task_not_a_greeting() -> None:
    h = [
        osys(),
        {"role": "assistant", "content": "hello, I am bot"},
        ou("TASK"),
        oa(text="a1"),
        ou("q2"),
        oa(text="a2"),
    ]
    out = ts.fit(h, 5, counter=lambda _: 1, keep_first_turn=True)
    assert out == [osys(), ou("TASK"), oa(text="a1"), ou("q2"), oa(text="a2")]


def test_real_result_after_intervening_assistant_is_moved_not_replaced() -> None:
    out, fixes = ts.repair_detailed([ou(), oa("a"), oa(text="hm"), ot("a", "REAL")])
    assert out == [ou(), oa("a"), ot("a", "REAL"), oa(text="hm")]
    assert [f.action for f in fixes] == ["moved_result"]


def test_tool_calls_none_does_not_confuse_detection() -> None:
    h = [au("q"), {"role": "assistant", "content": [use("a")], "tool_calls": None}]
    assert ts.detect_provider(h) == "anthropic"


@pytest.mark.parametrize("value", [None, "", "   "])
def test_placeholder_must_be_text(value: object) -> None:
    with pytest.raises(ValueError, match="placeholder"):
        ts.repair([ou()], placeholder=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [None, 0, -5, 1.5, True])
def test_max_tokens_must_be_positive_int(value: object) -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        ts.fit([ou()], value)  # type: ignore[arg-type]


def test_counter_must_return_int() -> None:
    with pytest.raises(TypeError, match="counter"):
        ts.fit([ou()], 10, counter=lambda _: None)  # type: ignore[arg-type,return-value]


def _timed(fn, *args, **kwargs) -> float:  # type: ignore[no-untyped-def]
    start = time.perf_counter()
    fn(*args, **kwargs)
    return time.perf_counter() - start


def test_many_orphans_is_linear() -> None:
    h = []
    for n in range(6_000):
        h += [au(f"q{n}"), aa(use(f"t{n}")), au(res(f"t{n}"), res(f"gone{n}")), aa("ok")]
    assert _timed(ts.repair, h) < 5


def test_reused_anthropic_id_every_turn_is_linear() -> None:
    h = []
    for n in range(6_000):
        h += [au(f"q{n}"), aa(use("toolu_0")), au(res("toolu_0")), aa("ok")]
    out = ts.repair(h)
    assert ts.validate(out).ok
    assert _timed(ts.repair, h) < 5


def test_many_parallel_calls_is_linear() -> None:
    ids = [f"c{n}" for n in range(10_000)]
    h = [ou(), oa(*ids), *(ot(cid) for cid in reversed(ids))]
    assert _timed(ts.validate, h) < 5
