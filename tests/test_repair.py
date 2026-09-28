import copy

import pytest

import turnsafe as ts
from tests.builders import aa, au, oa, osys, ot, ou, res, thinking, use


def actions(fixes: list[ts.Fix]) -> list[str]:
    return [f.action for f in fixes]


class TestOpenAI:
    def test_valid_history_is_returned_unchanged(self) -> None:
        h = [osys(), ou(), oa("a", "b"), ot("b"), ot("a"), oa(text="done")]
        out, fixes = ts.repair_detailed(h)
        assert out == h and fixes == []

    def test_input_is_not_mutated(self) -> None:
        h = [ot("x"), ou(), oa("a"), ou("wait"), ot("a")]
        before = copy.deepcopy(h)
        ts.repair(h)
        assert h == before

    def test_drops_orphan_left_by_trimming(self) -> None:
        out, fixes = ts.repair_detailed([ot("a"), ou(), oa(text="ok")])
        assert out == [ou(), oa(text="ok")]
        assert actions(fixes) == ["dropped_orphan_result"]

    def test_moves_result_back_next_to_its_call(self) -> None:
        out, fixes = ts.repair_detailed([ou(), oa("a"), ou("wait"), ot("a", "A")])
        assert out == [ou(), oa("a"), ot("a", "A"), ou("wait")]
        assert fixes[0].action == "moved_result" and fixes[0].index == 1

    def test_placeholder_for_missing_result(self) -> None:
        out = ts.repair([ou(), oa("a", "b"), ot("a")])
        assert out[-1] == {"role": "tool", "tool_call_id": "b", "content": ts.DEFAULT_PLACEHOLDER}
        assert ts.validate(out)

    def test_custom_placeholder(self) -> None:
        out = ts.repair([ou(), oa("a")], placeholder="cancelled by user")
        assert out[-1]["content"] == "cancelled by user"

    def test_strip_removes_the_call(self) -> None:
        out = ts.repair([ou(), oa("a", "b"), ot("a")], on_missing="strip")
        assert [c["id"] for c in out[1]["tool_calls"]] == ["a"]
        assert ts.validate(out)

    def test_strip_drops_assistant_with_nothing_left(self) -> None:
        out, fixes = ts.repair_detailed([ou(), oa("a"), ou("next")], on_missing="strip")
        assert out == [ou(), ou("next")]
        assert actions(fixes) == ["removed_call", "dropped_message"]

    def test_strip_keeps_assistant_text(self) -> None:
        out = ts.repair([ou(), oa("a", text="checking"), ou("next")], on_missing="strip")
        assert out[1] == {"role": "assistant", "content": "checking"}

    def test_duplicate_result_dropped(self) -> None:
        out, fixes = ts.repair_detailed([ou(), oa("a"), ot("a", "1"), ot("a", "2")])
        assert out == [ou(), oa("a"), ot("a", "1")]
        assert actions(fixes) == ["dropped_duplicate_result"]

    def test_duplicate_call_id_removed(self) -> None:
        out = ts.repair([ou(), oa("a", "a"), ot("a")])
        assert len(out[1]["tool_calls"]) == 1
        assert ts.validate(out)

    def test_invalid_policy(self) -> None:
        with pytest.raises(ValueError, match="on_missing"):
            ts.repair([], on_missing="ignore")  # type: ignore[arg-type]


class TestAnthropic:
    def test_valid_history_is_returned_unchanged(self) -> None:
        h = [au("q"), aa(thinking(), use("t1")), au(res("t1"), "and?"), aa("done")]
        out, fixes = ts.repair_detailed(h)
        assert out == h and fixes == []

    def test_thinking_block_survives_placeholder_repair(self) -> None:
        asst = aa(thinking("secret plan"), use("t1"), use("t2"))
        out = ts.repair([au("q"), asst, au(res("t1"))])
        assert out[1] == asst  # byte-for-byte, signature included
        assert [b["tool_use_id"] for b in out[2]["content"]] == ["t1", "t2"]
        assert out[2]["content"][1]["is_error"] is True

    def test_results_merged_into_existing_user_message(self) -> None:
        out = ts.repair([au("q"), aa(use("t1")), au("keep this text")])
        assert out[2]["content"][0]["type"] == "tool_result"
        assert out[2]["content"][1] == {"type": "text", "text": "keep this text"}
        assert len(out) == 3

    def test_new_user_message_when_none_follows(self) -> None:
        out = ts.repair([au("q"), aa(use("t1"))])
        assert out[-1]["role"] == "user"
        assert out[-1]["content"][0]["tool_use_id"] == "t1"

    def test_reorders_text_before_results(self) -> None:
        out, fixes = ts.repair_detailed([au("q"), aa(use("t1")), au("note", res("t1"))])
        assert [b["type"] for b in out[2]["content"]] == ["tool_result", "text"]
        assert actions(fixes) == ["reordered_blocks"]

    def test_moves_misplaced_result(self) -> None:
        out = ts.repair([au("q"), aa(use("t1")), au("hold on"), au(res("t1"))])
        assert ts.validate(out)
        assert out[2]["content"][0]["tool_use_id"] == "t1"

    def test_orphan_only_message_is_dropped(self) -> None:
        out = ts.repair([au("q"), aa("hi"), au(res("zz"))])
        assert out == [au("q"), aa("hi")]

    def test_orphan_removed_but_text_kept(self) -> None:
        out = ts.repair([au(res("zz"), "question"), aa("hi")])
        assert out[0] == {"role": "user", "content": [{"type": "text", "text": "question"}]}

    def test_empty_user_messages_and_blank_blocks_removed(self) -> None:
        h = [au("q"), aa("a"), au(" "), aa({"type": "text", "text": ""}, use("t1")), au(res("t1"))]
        out, fixes = ts.repair_detailed(h, provider="anthropic")
        assert ts.validate(out, provider="anthropic").violations == ()
        assert {f.action for f in fixes} == {"dropped_message", "removed_empty_text"}
        assert {f.index for f in fixes} == {2, 3}  # indices point into the input

    def test_empty_assistant_content_left_alone(self) -> None:
        h = [au("q"), aa(""), au("x")]
        assert ts.repair(h, provider="anthropic") == h

    def test_strip_policy_drops_thinking_only_leftover(self) -> None:
        out = ts.repair([au("q"), aa(thinking(), use("t1")), au("next")], on_missing="strip")
        assert out == [au("q"), au("next")]

    def test_system_message_is_left_for_the_caller(self) -> None:
        h = [{"role": "system", "content": "s"}, au("q")]
        out = ts.repair(h, provider="anthropic")
        assert out == h
        assert not ts.validate(out, provider="anthropic")


class TestAnthropicVerifiedRules:
    """Behaviour driven by the verified rules in docs/rules.md."""

    def test_results_split_over_user_messages_pass_through(self) -> None:
        h = [au("q"), aa(use("t1"), use("t2")), au(res("t1")), au(res("t2")), aa("ok")]
        out, fixes = ts.repair_detailed(h)
        assert out == h and fixes == []

    def test_split_results_merged_when_one_is_missing(self) -> None:
        h = [au("q"), aa(use("t1"), use("t2"), use("t3")), au(res("t1")), au(res("t2"))]
        out = ts.repair(h)
        assert len(out) == 3
        assert [b["tool_use_id"] for b in out[2]["content"]] == ["t1", "t2", "t3"]

    def test_duplicate_id_in_later_message_is_renamed_with_its_result(self) -> None:
        h = [au("q"), aa(use("t1")), au(res("t1", "first")), aa(use("t1")), au(res("t1", "2nd"))]
        out, fixes = ts.repair_detailed(h)
        assert out[3]["content"][0]["id"] == "t1_2"
        assert out[4]["content"][0] == {
            "type": "tool_result",
            "tool_use_id": "t1_2",
            "content": "2nd",
        }
        assert "renamed_duplicate_call" in actions(fixes)
        assert ts.validate(out).violations == ()

    def test_thinking_kept_when_the_change_is_after_it(self) -> None:
        # An interrupted run: only the message after the last assistant changes.
        h = [au("q"), aa(thinking("a"), use("t1"), use("t2")), au(res("t1"))]
        out, fixes = ts.repair_detailed(h)
        assert out[1] == h[1]
        assert "removed_thinking" not in actions(fixes)

    def test_thinking_removed_after_an_earlier_change(self) -> None:
        h = [au(res("orphan"), "q"), aa(thinking("a"), use("t1")), au(res("t1")), aa("done")]
        out, fixes = ts.repair_detailed(h, thinking="strip_after_change")
        assert out[1] == aa(use("t1"))
        assert [f.index for f in fixes if f.action == "removed_thinking"] == [1]

    def test_thinking_kept_by_default(self) -> None:
        h = [au(res("orphan"), "q"), aa(thinking("a"), use("t1")), au(res("t1")), aa("done")]
        assert ts.repair(h)[1] == h[1]

    def test_thinking_before_the_changed_block_is_kept(self) -> None:
        # Renaming the repeated id changes the tool_use block, which comes after
        # the thinking block: nothing before the thinking changed.
        h = [au("q"), aa(use("A")), au(res("A")), aa(thinking("t"), use("A")), au(res("A"))]
        out = ts.repair(h, thinking="strip_after_change")
        assert out[3]["content"][0] == thinking("t")
        assert out[3]["content"][1]["id"] == "A_2"

    def test_invalid_thinking_policy(self) -> None:
        with pytest.raises(ValueError, match="thinking"):
            ts.repair([], thinking="drop")  # type: ignore[arg-type]

    def test_blank_text_inside_tool_result_is_left_alone(self) -> None:
        # Live-tested: blank tool_result content is accepted.
        blank = {
            "type": "tool_result",
            "tool_use_id": "t1",
            "content": [{"type": "text", "text": " "}],
        }
        h = [au("q"), aa(use("t1")), au(blank)]
        assert ts.repair(h) == h

    def test_calls_split_over_assistant_messages_keep_their_results(self) -> None:
        # Regression: the real result for A used to be dropped as an orphan.
        h = [au("q"), aa(use("A")), aa(use("B")), au(res("A", "real A"), res("B"))]
        out, fixes = ts.repair_detailed(h)
        assert out == h and fixes == []

    def test_merged_turn_missing_a_result(self) -> None:
        h = [au("q"), aa(use("A")), aa("thinking aloud"), aa(use("B")), au(res("A", "real A"))]
        out = ts.repair(h)
        assert out[:4] == h[:4]
        assert [b["tool_use_id"] for b in out[4]["content"]] == ["A", "B"]
        assert out[4]["content"][0]["content"] == "real A"

    def test_repeat_inside_merged_turn_is_removed(self) -> None:
        h = [au("q"), aa(use("A")), aa("x", use("A")), au(res("A"))]
        out = ts.repair(h)
        assert out[2] == {"role": "assistant", "content": [{"type": "text", "text": "x"}]}
        assert ts.validate(out).violations == ()
