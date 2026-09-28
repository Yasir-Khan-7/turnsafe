# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-09-28

### Added

- `validate()`: checks an OpenAI Chat Completions or Anthropic Messages history and returns a
  `Report` of typed `Violation`s. The report says what the API would reject, where, and for
  which tool call.
- `repair()` and `repair_detailed()`: rebuild a valid history. They move misplaced results,
  drop orphaned and duplicate results, and add placeholder results for unanswered calls (or
  remove those calls when `on_missing="strip"`). They also fix the Anthropic block-order and
  empty-content rules.
- `fit()` and `fit_detailed()`: trim a history to a token budget by dropping whole turns, so a
  tool call is never split from its results. Options: `keep_system`, `keep_first_turn` and a
  custom `counter`.
- `thinking=` on `repair()` and `fit()`. The default `"keep"` never touches Anthropic thinking
  blocks. `"strip_after_change"` removes them from the first changed block onward, for Claude
  models with thinking-block prefix binding (Fable 5.1, Opus 5.5).
- Anthropic rules verified against the docs on 2026-09-28:
  - Consecutive user messages, and consecutive assistant messages, are each treated as one turn.
  - `tool_use` ids must be unique across the history.
  - Blank `tool_result` content and empty assistant content are accepted.
  - A mid-conversation `system` message is a warning; it's an error only as the first message.
- A result that ended up after an unrelated assistant message is moved back to its call instead
  of being replaced by a placeholder. The search stops at the next call that reuses the same id.
- Tool call and result ids must be non-empty strings. Anything else raises a `ValueError` that
  names the position, instead of silently escaping the rules.
- Validation, repair and trimming are linear in the size of the history, including for
  histories with many orphaned results, reused ids or thousands of parallel calls.
- `approx_tokens()`: an offline token estimate that counts images as images.
- `detect_provider()` and `provider="auto"`.
- Accepts SDK message objects that have `model_dump()`.
- Property-based tests for the invariants listed in `AGENT.md`, and opt-in live contract tests
  against both APIs.
