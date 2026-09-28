from collections.abc import Mapping
from typing import Any

import pytest

import turnsafe as ts
from tests.builders import aa, au, oa, osys, ot, ou, res, thinking, use


def one_each(_: Mapping[str, Any]) -> int:
    return 1


def agent_history() -> list[dict[str, Any]]:
    # 3 turns: [u, a(tool), t, a] + [u, a(2 tools), t, t, a] + [u, a]
    return [
        osys(),
        ou("first task"),
        oa("a"),
        ot("a"),
        oa(text="did a"),
        ou("second"),
        oa("b", "c"),
        ot("b"),
        ot("c"),
        oa(text="did b, c"),
        ou("third"),
        oa(text="ok"),
    ]


def test_everything_fits() -> None:
    h = agent_history()
    result = ts.fit_detailed(h, 1000, counter=one_each)
    assert result.messages == h
    assert result.tokens == 12 and result.dropped_turns == 0


def test_drops_oldest_whole_turns() -> None:
    result = ts.fit_detailed(agent_history(), 8, counter=one_each)
    # system (1) + turn 2 (5) + turn 3 (2) = 8; turn 1 (4) is dropped whole.
    assert result.messages[0] == osys()
    assert result.messages[1] == ou("second")
    assert result.dropped_turns == 1 and result.dropped_messages == 4
    assert ts.validate(result.messages)


def test_never_splits_a_tool_group() -> None:
    # Budget of 5 cannot hold turn 2 (5) + turn 3 (2) + system (1): only turn 3 survives.
    out = ts.fit(agent_history(), 5, counter=one_each)
    assert out == [osys(), ou("third"), oa(text="ok")]


def test_keep_first_turn() -> None:
    out = ts.fit(agent_history(), 7, counter=one_each, keep_first_turn=True)
    assert out == [
        osys(),
        ou("first task"),
        oa("a"),
        ot("a"),
        oa(text="did a"),
        ou("third"),
        oa(text="ok"),
    ]
    assert ts.validate(out)


def test_keep_system_false_lets_system_be_dropped() -> None:
    out = ts.fit(agent_history(), 2, counter=one_each, keep_system=False)
    assert out == [ou("third"), oa(text="ok")]


def test_latest_turn_over_budget_raises() -> None:
    with pytest.raises(ts.ContextBudgetError) as exc:
        ts.fit(agent_history(), 2, counter=one_each)
    assert exc.value.needed == 3 and exc.value.max_tokens == 2


def test_invalid_input_raises_and_points_to_repair() -> None:
    with pytest.raises(ts.InvalidHistoryError, match="repair"):
        ts.fit([ot("a"), ou()], 100)


def test_anthropic_tool_result_messages_stay_with_their_call() -> None:
    h = [
        au("one"),
        aa(use("t1")),
        au(res("t1")),
        aa("r1"),
        au("two"),
        aa(use("t2")),
        au(res("t2"), "extra"),
        aa("r2"),
    ]
    out = ts.fit(h, 4, counter=one_each)
    assert out == h[4:]
    assert ts.validate(out).violations == ()


def test_leading_non_user_messages_form_their_own_turn() -> None:
    out = ts.fit([aa("greeting"), au("q"), aa("a")], 2, provider="anthropic", counter=one_each)
    assert out == [au("q"), aa("a")]


def test_only_system_messages() -> None:
    result = ts.fit_detailed([osys()], 10, counter=one_each)
    assert result.messages == [osys()] and result.tokens == 1


def test_bad_budget() -> None:
    with pytest.raises(ValueError):
        ts.fit([ou()], 0)


def test_default_counter_is_used() -> None:
    long = ou("x" * 4000)
    out = ts.fit([long, oa(text="ok"), ou("short"), oa(text="ok")], 200)
    assert out == [ou("short"), oa(text="ok")]


class TestApproxTokens:
    def test_scales_with_length(self) -> None:
        assert ts.approx_tokens(ou("x" * 400)) > ts.approx_tokens(ou("x" * 40))

    def test_minimum_one(self) -> None:
        assert ts.approx_tokens({}) >= 1

    def test_base64_image_counted_as_image_not_text(self) -> None:
        huge = "A" * 1_000_000
        anthropic_img = {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": huge},
                }
            ],
        }
        openai_img = {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + huge}}
            ],
        }
        assert ts.approx_tokens(anthropic_img) < 2000
        assert ts.approx_tokens(openai_img) < 2000


class TestThinking:
    def history(self) -> list[dict[str, Any]]:
        return [
            au("one"),
            aa(thinking("x"), use("t1")),
            au(res("t1")),
            aa("r1"),
            au("two"),
            aa(thinking("y"), use("t2")),
            au(res("t2")),
            aa("r2"),
        ]

    def test_stripped_from_kept_turns_after_a_drop(self) -> None:
        result = ts.fit_detailed(self.history(), 4, counter=one_each, thinking="strip_after_change")
        assert result.messages[1] == aa(use("t2"))
        assert result.removed_thinking == 1

    def test_untouched_when_nothing_is_dropped(self) -> None:
        h = self.history()
        result = ts.fit_detailed(h, 100, counter=one_each, thinking="strip_after_change")
        assert result.messages == h and result.removed_thinking == 0

    def test_kept_by_default(self) -> None:
        h = self.history()
        assert ts.fit(h, 4, counter=one_each) == h[4:]
