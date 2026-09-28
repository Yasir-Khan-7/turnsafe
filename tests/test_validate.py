import pytest

import turnsafe as ts
from tests.builders import aa, au, oa, osys, ot, ou, res, thinking, use


def codes(report: ts.Report) -> list[str]:
    return [v.code for v in report.violations]


class TestOpenAI:
    def test_valid_history_with_parallel_calls(self) -> None:
        h = [osys(), ou(), oa("a", "b"), ot("b"), ot("a"), oa(text="done"), ou("thanks")]
        report = ts.validate(h)
        assert report.ok and bool(report)
        assert report.violations == ()
        assert report.explain() == "openai history is valid."

    def test_orphan_tool_message_after_trim(self) -> None:
        # The classic: a naive slice starts the window on a tool message.
        report = ts.validate([ot("a"), ou(), oa(text="ok")])
        assert codes(report) == [ts.ORPHAN_TOOL_RESULT]
        assert report.violations[0].index == 0
        assert report.violations[0].tool_call_id == "a"

    def test_unanswered_call(self) -> None:
        report = ts.validate([ou(), oa("a", "b"), ot("a")])
        assert codes(report) == [ts.UNANSWERED_TOOL_CALL]
        assert report.violations[0].tool_call_id == "b"
        assert report.violations[0].index == 1

    def test_user_message_between_call_and_result_is_misplaced(self) -> None:
        report = ts.validate([ou(), oa("a"), ou("wait"), ot("a")])
        assert codes(report) == [ts.MISPLACED_TOOL_RESULT]
        assert report.violations[0].index == 3

    def test_result_after_next_assistant_is_misplaced(self) -> None:
        # The real result exists, so repair() can move it back instead of
        # inventing a "not executed" placeholder.
        report = ts.validate([ou(), oa("a"), oa(text="hm"), ot("a")])
        assert codes(report) == [ts.MISPLACED_TOOL_RESULT]

    def test_reused_id_limits_the_search(self) -> None:
        # call_0 is reused by the next turn, so the later result belongs to it.
        report = ts.validate([ou(), oa("call_0"), ou(), oa("call_0"), ot("call_0")])
        assert codes(report) == [ts.UNANSWERED_TOOL_CALL]
        assert report.violations[0].index == 1

    def test_duplicate_result(self) -> None:
        report = ts.validate([ou(), oa("a"), ot("a"), ot("a")])
        assert codes(report) == [ts.DUPLICATE_TOOL_RESULT]

    def test_duplicate_call_id(self) -> None:
        report = ts.validate([ou(), oa("a", "a"), ot("a")])
        assert codes(report) == [ts.DUPLICATE_TOOL_CALL_ID]

    def test_ids_may_repeat_across_turns(self) -> None:
        # Some providers restart ids like "call_0" every turn.
        assert ts.validate([ou(), oa("call_0"), ot("call_0"), ou(), oa("call_0"), ot("call_0")])

    def test_system_messages_anywhere_are_fine(self) -> None:
        assert ts.validate([ou(), osys(), oa(text="x")], provider="openai")

    def test_raise_for_errors(self) -> None:
        with pytest.raises(ts.InvalidHistoryError) as exc:
            ts.validate([ot("a")]).raise_for_errors()
        assert "repair()" in str(exc.value)
        assert exc.value.report.errors[0].code == ts.ORPHAN_TOOL_RESULT


class TestAnthropic:
    def test_valid_history_with_thinking(self) -> None:
        h = [
            au("q"),
            aa(thinking(), use("t1"), use("t2")),
            au(res("t2"), res("t1"), "more?"),
            aa("done"),
        ]
        assert ts.validate(h).violations == ()

    def test_unanswered_tool_use(self) -> None:
        report = ts.validate([au("q"), aa(use("t1")), au("no results here")])
        assert codes(report) == [ts.UNANSWERED_TOOL_CALL]

    def test_results_split_over_consecutive_user_messages_are_valid(self) -> None:
        # The API merges consecutive user messages before checking.
        h = [au("q"), aa(use("t1"), use("t2")), au(res("t1")), au(res("t2")), aa("ok")]
        assert ts.validate(h).violations == ()

    def test_text_then_result_in_consecutive_user_messages(self) -> None:
        # Merged, this puts text before the tool_result.
        report = ts.validate([au("q"), aa(use("t1")), au("hold on"), au(res("t1"))])
        assert codes(report) == [ts.TOOL_RESULT_NOT_FIRST]
        assert report.violations[0].index == 2

    def test_system_between_call_and_result_is_misplaced(self) -> None:
        h = [au("q"), aa(use("t1")), {"role": "system", "content": "s"}, au(res("t1"))]
        assert ts.MISPLACED_TOOL_RESULT in codes(ts.validate(h, provider="anthropic"))

    def test_duplicate_tool_use_id_across_messages(self) -> None:
        h = [au("q"), aa(use("t1")), au(res("t1")), aa(use("t1")), au(res("t1"))]
        report = ts.validate(h)
        assert codes(report) == [ts.DUPLICATE_TOOL_CALL_ID]
        assert report.violations[0].index == 3

    def test_blank_text_inside_tool_result_is_accepted(self) -> None:
        blank = {
            "type": "tool_result",
            "tool_use_id": "t1",
            "content": [{"type": "text", "text": " "}],
        }
        report = ts.validate([au("q"), aa(use("t1")), au(blank)])
        assert report.violations == ()

    def test_orphan_result(self) -> None:
        report = ts.validate([au(res("x")), aa("hi")])
        assert codes(report) == [ts.ORPHAN_TOOL_RESULT]

    def test_text_before_tool_result(self) -> None:
        report = ts.validate([au("q"), aa(use("t1")), au("note", res("t1"))])
        assert codes(report) == [ts.TOOL_RESULT_NOT_FIRST]

    def test_system_message_first_is_an_error(self) -> None:
        report = ts.validate([{"role": "system", "content": "s"}, au("q")], provider="anthropic")
        assert [v.code for v in report.errors] == [ts.SYSTEM_IN_MESSAGES]

    def test_mid_conversation_system_message_is_a_warning(self) -> None:
        # Supported on Opus 4.8+, Opus 5, Fable and Mythos; rejected by older models.
        h = [au("q"), aa("a"), {"role": "system", "content": "s"}, au("next")]
        report = ts.validate(h, provider="anthropic")
        assert report.ok
        assert [v.code for v in report.warnings] == [ts.SYSTEM_IN_MESSAGES]

    def test_empty_user_message(self) -> None:
        report = ts.validate([au("q"), aa("a"), au(" ")], provider="anthropic")
        assert codes(report) == [ts.EMPTY_CONTENT]

    def test_empty_assistant_content_is_accepted(self) -> None:
        # Live-tested: "" and [] assistant content are accepted in any position.
        h = [au("q"), aa(""), au("x"), {"role": "assistant", "content": []}, au("y")]
        assert ts.validate(h, provider="anthropic").violations == ()

    def test_whitespace_text_block_is_empty(self) -> None:
        report = ts.validate(
            [au("q"), aa({"type": "text", "text": "  "}), au("x")], provider="anthropic"
        )
        assert codes(report) == [ts.EMPTY_CONTENT]

    def test_empty_final_assistant_is_prefill(self) -> None:
        assert ts.validate([au("q"), aa("")], provider="anthropic").violations == ()

    def test_first_message_not_user_is_a_warning(self) -> None:
        report = ts.validate([aa("hi"), au("q")], provider="anthropic")
        assert report.ok
        assert [v.code for v in report.warnings] == [ts.FIRST_MESSAGE_NOT_USER]

    def test_server_tool_blocks_need_no_client_result(self) -> None:
        h = [
            au("q"),
            aa(
                {"type": "server_tool_use", "id": "srv", "name": "web_search", "input": {}},
                {"type": "web_search_tool_result", "tool_use_id": "srv", "content": []},
                "answer",
            ),
            au("thanks"),
        ]
        assert ts.validate(h).violations == ()


class TestConsecutiveAssistantMessages:
    """The Anthropic API merges consecutive assistant messages into one turn."""

    def test_calls_split_over_assistant_messages_are_valid(self) -> None:
        h = [au("q"), aa(use("A")), aa(use("B")), au(res("A"), res("B")), aa("ok")]
        assert ts.validate(h).violations == ()

    def test_text_message_between_call_and_results_is_valid(self) -> None:
        h = [au("q"), aa(thinking(), use("A")), aa("checking"), au(res("A")), aa("ok")]
        assert ts.validate(h).violations == ()

    def test_repeat_inside_a_merged_turn(self) -> None:
        h = [au("q"), aa(use("A")), aa(use("A")), au(res("A"))]
        assert codes(ts.validate(h)) == [ts.DUPLICATE_TOOL_CALL_ID]

    def test_openai_does_not_merge(self) -> None:
        report = ts.validate([ou(), oa("a"), oa(text="hm"), ot("a")])
        assert codes(report) == [ts.MISPLACED_TOOL_RESULT]


class TestDetection:
    def test_detects_openai(self) -> None:
        assert ts.detect_provider([ou(), oa("a"), ot("a")]) == "openai"

    def test_detects_anthropic(self) -> None:
        assert ts.detect_provider([au("q"), aa(use("a"))]) == "anthropic"

    def test_anthropic_with_stray_system_is_still_anthropic(self) -> None:
        assert ts.detect_provider([{"role": "system", "content": "s"}, aa(use("a"))]) == "anthropic"

    def test_mixed_formats_raise(self) -> None:
        with pytest.raises(ts.ProviderDetectionError):
            ts.detect_provider([ot("a"), aa(use("b"))])

    def test_unknown_provider(self) -> None:
        with pytest.raises(ValueError, match="Unknown provider"):
            ts.validate([], provider="gemini")


def test_accepts_sdk_objects_with_model_dump() -> None:
    class FakeMessage:
        def __init__(self, data: dict) -> None:
            self.data = data

        def model_dump(self, exclude_none: bool = False) -> dict:
            return {k: v for k, v in self.data.items() if not (exclude_none and v is None)}

    report = ts.validate([ou(), FakeMessage(oa("a")), ot("a")])
    assert report.ok


def test_rejects_non_messages() -> None:
    with pytest.raises(TypeError, match="message dict"):
        ts.validate(["not a message"])
