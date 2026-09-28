<div align="center">

# turnsafe

**Never send a broken chat history to an LLM again.**

Validate, repair and trim OpenAI and Anthropic message lists, without ever separating a tool
call from its result.

[![CI](https://github.com/Yasir-Khan-7/turnsafe/actions/workflows/ci.yml/badge.svg)](https://github.com/Yasir-Khan-7/turnsafe/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/turnsafe.svg)](https://pypi.org/project/turnsafe/)
[![Python](https://img.shields.io/pypi/pyversions/turnsafe.svg)](https://pypi.org/project/turnsafe/)
[![Typed](https://img.shields.io/badge/typing-strict-blue.svg)](https://mypy.readthedocs.io/)
[![Dependencies: 0](https://img.shields.io/badge/dependencies-0-brightgreen.svg)](https://github.com/Yasir-Khan-7/turnsafe/blob/main/pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/Yasir-Khan-7/turnsafe/blob/main/LICENSE)

</div>

---

## Did one of these errors bring you here?

```text
anthropic.BadRequestError: messages.4: `tool_use` ids were found without `tool_result` blocks
immediately after: toolu_01A... Each `tool_use` block must have a corresponding `tool_result`
block in the next message.
```

```text
openai.BadRequestError: Invalid parameter: messages with role 'tool' must be a response to a
preceeding message with 'tool_calls'.
```

```text
openai.BadRequestError: An assistant message with 'tool_calls' must be followed by tool
messages responding to each 'tool_call_id'. The following tool_call_ids did not have
response messages: call_abc123
```

Each one means **a tool call got separated from its result**. Common causes:

- You trimmed the history to fit the context window (`messages[-20:]`) and the cut landed in
  the middle of a tool call.
- You saved the conversation and reloaded it, and something got lost or reordered.
- The agent crashed or was cancelled while a tool was running.
- The user sent a message while tools were still running.

After that, **every later request in the conversation fails** until someone fixes the history.

`turnsafe` fixes the history before you send it. It works on the plain `messages` list you
already pass to the OpenAI and Anthropic SDKs. It has no dependencies and needs no framework or
network access.

## Contents

- [Install](#install)
- [Quickstart](#quickstart)
- [Try it without an API key](#try-it-without-an-api-key)
- [How it works](#how-it-works)
- [API](#api)
- [Anthropic thinking blocks](#anthropic-thinking-blocks)
- [How it compares](#how-it-compares)
- [FAQ](#faq)
- [Status and roadmap](#status-and-roadmap)
- [Contributing](#contributing)

## Install

```bash
pip install turnsafe
```

It needs Python 3.10 or newer and has zero runtime dependencies. On PyPI:
[pypi.org/project/turnsafe](https://pypi.org/project/turnsafe/).

## Quickstart

Add one line before each API call in your agent loop:

```python
import anthropic
import turnsafe

client = anthropic.Anthropic()

while True:
    request = turnsafe.fit(turnsafe.repair(messages), max_tokens=150_000)
    response = client.messages.create(
        model="claude-sonnet-5", max_tokens=1024, tools=tools, messages=request
    )
    ...
```

- `repair()` heals a broken history.
- `fit()` trims whole turns so the history fits your budget.

They work the same way with `openai.OpenAI().chat.completions.create(...)`. The provider is
detected from the message format, or you can pass `provider="openai"` or `"anthropic"`.

For complete runnable agents that save and reload their history, see
[`examples/anthropic_agent.py`](https://github.com/Yasir-Khan-7/turnsafe/blob/main/examples/anthropic_agent.py)
and [`examples/openai_agent.py`](https://github.com/Yasir-Khan-7/turnsafe/blob/main/examples/openai_agent.py).

## Try it without an API key

```bash
git clone https://github.com/Yasir-Khan-7/turnsafe && cd turnsafe
pip install -e . && python examples/offline_demo.py
```

```text
=== Anthropic: agent crashed between two parallel tool calls ===
anthropic history has 1 error(s), 0 warning(s):
  [error] messages[1] unanswered_tool_call: tool call 'toolu_2' has no tool result in the next user message
  fix: messages[1] added_placeholder: no result for 'toolu_2'
  after repair: anthropic history is valid.

=== OpenAI: a user message landed between a call and its result ===
openai history has 1 error(s), 0 warning(s):
  [error] messages[3] misplaced_tool_result: result for tool call 'call_7' must be in the tool messages directly after the call
  fix: messages[1] moved_result: moved result for 'call_7' from messages[3]
  after repair: openai history is valid.

=== fit() to 1,000 tokens ===
kept 24 of 80 messages (~977 tokens), dropped 14 whole turns; valid: True
```

## How it works

```text
user ─┐
      │  turn 1          fit() drops whole turns, oldest first.
asst  │  (tool call) ─┐  A tool call and its results are always in the
tool  │  (result)  ◄──┘  same turn, so they leave together or stay together.
asst ─┘
user ─┐
      │  turn 2          repair() pairs every call with its result and fixes
asst  │  (tool call) ─┐  anything the API would reject: missing results,
user  │  "hurry!"     │  orphaned results, misplaced results, duplicates.
tool  │  (result)  ◄──┘  ← misplaced: repair() moves it back next to its call
asst ─┘
```

Every rule turnsafe checks comes from the provider's documentation, or from the exact API error
text when the docs are silent. All sources are listed in
[`docs/rules.md`](https://github.com/Yasir-Khan-7/turnsafe/blob/main/docs/rules.md).

The core promises are checked by property-based tests that run thousands of randomly damaged
histories on every change:

- `repair()` output always passes `validate()`.
- A valid history comes back unchanged.
- `repair()` never mutates its input.
- `fit()` never splits a tool call from its result.

## API

### `validate(messages, provider="auto") -> Report`

`validate()` tells you what the API would reject, without changing anything.

```python
report = turnsafe.validate(messages)
if not report:                  # falsy when there are errors
    print(report.explain())
report.raise_for_errors()       # or fail fast, for example in tests or CI
```

Each `Violation` has:
- a stable `code`, such as `"orphan_tool_result"`
- the message `index`
- the `tool_call_id`
- a `severity`, either `"error"` or `"warning"`

### `repair(messages, provider="auto", *, on_missing="placeholder", placeholder=..., thinking="keep")`

`repair()` returns a fixed copy of the history. Use `repair_detailed()` to also get a `Fix` for
every change, which is useful for logging.

```python
messages, fixes = turnsafe.repair_detailed(messages)
for fix in fixes:
    log.warning("turnsafe: %s", fix)   # messages[1] added_placeholder: no result for 'toolu_2'
```

| Problem | What `repair()` does |
|---|---|
| A tool result whose call is gone, for example trimmed away | Drops it and keeps any other content in the same message. |
| A tool call with no result, for example after an interrupted run | Adds an error result saying the tool didn't run. With `on_missing="strip"` it removes the call instead. |
| A result in the wrong place, for example a user message got in between | Moves it back right after its call. |
| A second result for the same call | Keeps the first. |
| Anthropic: text before the `tool_result` blocks | Moves the results first. |
| Anthropic: the same `tool_use` id used twice in the history | Renames the later call and its result. |
| Anthropic: an empty user message, or a blank text block | Removes it. |

`repair()` also guarantees:
- It never mutates your list.
- A valid history comes back unchanged.
- The output passes `validate()`. The one exception is an Anthropic `system` message placed
  first, which you have to move to the `system` parameter yourself.
- By default it never rewrites an assistant message, so Anthropic thinking blocks and their
  signatures survive.

### `fit(messages, max_tokens, provider="auto", *, counter=approx_tokens, keep_system=True, keep_first_turn=False, thinking="keep")`

`fit()` drops the oldest whole turns until the history fits in `max_tokens`.

```python
result = turnsafe.fit_detailed(messages, max_tokens=100_000, keep_first_turn=True)
print(result.tokens, result.dropped_turns)
```

- `keep_system` keeps leading system and developer messages.
- `keep_first_turn` keeps the first turn, which is usually the original task.
- If even the latest turn doesn't fit, `fit()` raises `ContextBudgetError` instead of returning
  a broken history.
- If the input breaks the rules, it raises `InvalidHistoryError`, so call `repair()` first.
- `max_tokens` covers the messages only. Leave room for tools, the system prompt and the reply.
- The default counter is an offline estimate: about 4 characters per token, with images counted
  by size rather than by their base64 length. Pass `counter=` for exact numbers, for example
  with `tiktoken` or Anthropic's token-counting endpoint.

### Rules checked

| Code | Providers | Meaning |
|---|---|---|
| `unanswered_tool_call` | both | A tool call has no result |
| `orphan_tool_result` | both | A result answers no call in the preceding assistant turn |
| `misplaced_tool_result` | both | The result exists but isn't directly after its call |
| `duplicate_tool_result` | both | A call has more than one result |
| `duplicate_tool_call_id` | both | A tool call id is repeated. For Anthropic, this means anywhere in the history. |
| `tool_result_not_first` | Anthropic | Text comes before the `tool_result` blocks in the user turn |
| `system_in_messages` | Anthropic | An error when first. Elsewhere a warning, because only Opus 4.8+, Opus 5, Fable and Mythos accept it. |
| `empty_content` | Anthropic | An empty user message, or a blank text block outside a `tool_result` |
| `first_message_not_user` | Anthropic | Warning: Bedrock and some deployments require a user message first |

The Anthropic API merges consecutive messages with the same role, and turnsafe does the same.
Results split over several user messages in a row, or calls split over several assistant
messages, are valid.

## Anthropic thinking blocks

By default, turnsafe never touches `thinking` or `redacted_thinking` blocks.

**On Claude Fable 5.1 and Opus 5.5**, a replayed thinking block may be valid only while every
message before it is unchanged. This is enforced by default for accounts created on or after
2026-08-31. Trimming the front of the history then makes the API reject later thinking blocks
with ``Invalid `signature` in `thinking` block``. There are two fixes:

- Set `thinking.block_binding.prefix_mismatch_behavior` to `"drop_block"` on the request. That's
  the field the API error names, and the API then drops the stale blocks itself.
- Or pass `thinking="strip_after_change"` to `fit()` and `repair()`. turnsafe then removes
  thinking blocks after the first change, which is the documented recovery.

Don't strip on other models. They expect the latest turn's thinking to be replayed. See the
[details](https://github.com/Yasir-Khan-7/turnsafe/blob/main/docs/rules.md#anthropic-thinking-blocks).

## How it compares

| | turnsafe | litellm `modify_params` | LangChain `trim_messages` | pydantic-ai history repair |
|---|---|---|---|---|
| Works on plain SDK dicts, with no framework | ✅ | ❌ inside `litellm.completion()` | ❌ LangChain messages | ❌ pydantic-ai messages |
| OpenAI **and** Anthropic formats | ✅ | Anthropic path only | ✅ | ✅ |
| Reports problems without changing anything | ✅ `validate()` | ❌ | ❌ | ❌ |
| Repairs broken pairs | ✅ | ✅ | ❌ | ✅ |
| Trims to a token budget | ✅ | ❌ | ✅ | ❌ |
| Trimming can't orphan a tool call | ✅ property-tested | – | ❌ [#33245](https://github.com/langchain-ai/langchain/issues/33245) | – |
| Tells you what it changed | ✅ | ❌ | ❌ | ❌ |

If your framework already handles this for you, keep using it. `turnsafe` is for agent loops
written directly on the provider SDKs, for histories you store and reload yourself, and for
anyone who wants to check a history before sending it. For how the problem was chosen and what
we learned from existing tools, see
[`docs/research.md`](https://github.com/Yasir-Khan-7/turnsafe/blob/main/docs/research.md).

## FAQ

**Does it call any API or send my data anywhere?**
No. It's pure Python working on your list in memory.

**Can I pass SDK objects instead of dicts?**
Yes. Messages and content blocks that have `model_dump()` are accepted, for example
`response.choices[0].message` or `response.content`. The output is always plain dicts.

**What if a tool call id isn't a string?**
turnsafe raises a `ValueError` naming the position, for example
`messages[3].tool_calls[0].id must be a non-empty string`. A call without a usable id can't be
matched to its result, so it isn't silently ignored.

**What does the placeholder result say?**
It says the tool call wasn't executed or its result was lost, and that the model shouldn't
assume it succeeded. Change it with `placeholder="..."`.

**My history ends with an assistant tool call because I'm about to run the tools. Is that
an error?**
`validate()` reports it, because the API would reject it if you sent it. Don't `repair()`
a history while you're about to add the real results yourself.

**Should I use this with server-side compaction?**
Yes, they work together. Anthropic context editing and OpenAI Responses compaction make the
context smaller, and turnsafe makes sure what you send is valid.

## Status and roadmap

- **v0.1:** OpenAI Chat Completions and Anthropic Messages.
- **Planned:**
  - OpenAI Responses API items
  - Gemini `contents`
  - an optional LangChain adapter
- Legacy OpenAI `function_call` messages aren't checked.

The version follows [Semantic Versioning](https://semver.org/). Violation codes and fix actions
are public API. Changes are listed in the
[CHANGELOG](https://github.com/Yasir-Khan-7/turnsafe/blob/main/CHANGELOG.md).

## Contributing

Issues and pull requests are welcome. See
[CONTRIBUTING.md](https://github.com/Yasir-Khan-7/turnsafe/blob/main/CONTRIBUTING.md) to get
set up. If you work with an AI coding agent, point it at
[AGENT.md](https://github.com/Yasir-Khan-7/turnsafe/blob/main/AGENT.md). To report a security
issue, see [SECURITY.md](https://github.com/Yasir-Khan-7/turnsafe/blob/main/SECURITY.md).

Found a provider rule turnsafe gets wrong? Open an issue with the exact API error text. That's
the most valuable report you can send.

## License

[MIT](https://github.com/Yasir-Khan-7/turnsafe/blob/main/LICENSE) © Yasir Khan
