# Provider rules turnsafe checks

This page lists every rule `turnsafe` enforces. For each rule you'll find the API error it
prevents, where the rule comes from, and what `repair()` does about it. The rules were
verified against the official docs and real API error reports on 2026-09-28.

**Matching error text in your own code:** Anthropic's documentation says the trailing sentence
of some errors can change. Match only on the stable start of each message, shown below.
OpenAI spells "preceeding" with a double e; OpenAI-compatible servers often spell it
"preceding", so match both.

## How calls are matched to results

- **What answers a call.** A call is answered by the first result with its id that comes after
  the call. The search stops at the next assistant turn that calls the same id again, because
  some OpenAI-compatible servers restart ids such as `call_0` every turn.
- **Misplaced results.** A result found later than the provider allows is reported as
  `misplaced_tool_result`, and `repair()` moves it back next to its call. The real result is
  never replaced by a placeholder.
- **Ids must be strings.** Every tool call and result id must be a non-empty string. Otherwise
  turnsafe raises a `ValueError` naming the position (for example
  `messages[3].tool_calls[0].id`), because a call without a usable id can't be checked.

## OpenAI Chat Completions

| Code | Rule | Error you'd get | `repair()` |
|---|---|---|---|
| `unanswered_tool_call` | Every id in an assistant message's `tool_calls` must be answered by a `role: "tool"` message, and those answers must come before any other role. | `An assistant message with 'tool_calls' must be followed by tool messages responding to each 'tool_call_id'. The following tool_call_ids did not have response messages: …` | Adds a placeholder error result, or removes the call with `on_missing="strip"` |
| `orphan_tool_result` | A `tool` message must answer a call in the assistant message just before its run of tool messages. | `Invalid parameter: messages with role 'tool' must be a response to a preceeding message with 'tool_calls'.`, or `Invalid parameter: 'tool_call_id' of '…' not found in 'tool_calls' of previous message.` | Drops it |
| `misplaced_tool_result` | The answer exists but another message (for example a user message) sits between it and its call. | Same as `unanswered_tool_call` | Moves it back next to its call |
| `duplicate_tool_result` | A call is answered twice. | `Invalid parameter: Duplicate value for 'tool_call_id' of '…' in messages[i] and messages[j]` | Keeps the first answer |
| `duplicate_tool_call_id` | One assistant message repeats an id. | Unverified: no error report found. Checked defensively. | Keeps the first call |

Not flagged:
- **Tool answers in a different order from the calls.** No docs or error reports show this is
  rejected, since answers are matched by id.
- **`content: null` on an assistant message that has `tool_calls`.** The API spec types allow
  it: "Required unless `tool_calls` or `function_call` is specified."
- **The same id reused in a later turn.** Some OpenAI-compatible servers restart ids at
  `call_0` every turn.
- **A `system` or `developer` message between two tool answers.** It's unclear whether the API
  accepts this. turnsafe treats it strictly: an answer after the interruption is reported as
  misplaced.

Sources: [openai-python assistant message type](https://github.com/openai/openai-python/blob/main/src/openai/types/chat/chat_completion_assistant_message_param.py),
and error reports that quote the exact wording:
- [warp #14129](https://github.com/warpdotdev/warp/issues/14129)
- [OpenManus #300](https://github.com/FoundationAgents/OpenManus/issues/300)
- [CopilotKit #2933](https://github.com/CopilotKit/CopilotKit/issues/2933)
- [search2ai #30](https://github.com/fatwang2/search2ai/issues/30)

## Anthropic Messages

| Code | Rule | Error you'd get | `repair()` |
|---|---|---|---|
| `unanswered_tool_call` | Every `tool_use` block needs a `tool_result` in the user turn that follows it. | ``messages.N: `tool_use` ids were found without `tool_result` blocks immediately after: …`` | Adds a placeholder `tool_result` with `is_error: true`, or removes the call with `on_missing="strip"` |
| `orphan_tool_result` | A `tool_result` must answer a `tool_use` in the previous assistant message. | ``messages.N.content.M: unexpected `tool_use_id` found in `tool_result` blocks: …`` | Drops the block and keeps any other content |
| `misplaced_tool_result` | Something other than a user message, such as a `system` message, sits between the assistant turn and its results. | Same as `unanswered_tool_call` | Moves the result into the user turn right after the call |
| `tool_result_not_first` | In the user turn that carries results, every `tool_result` must come before any text. | 400. Real reports show the `unanswered_tool_call` wording. | Moves results first |
| `duplicate_tool_call_id` | A `tool_use` id appears more than once anywhere in the history. | `messages.N.content.M: tool_use ids must be unique` | Renames the later call and its result to `<id>_2` |
| `duplicate_tool_result` | A call is answered twice. | (Same family as `orphan_tool_result`) | Keeps the first result |
| `empty_content` | A user message must not be empty or whitespace-only, and a text block must not be blank. Accepted, per a live test on 2026-09-05 ([experiential #804](https://github.com/experientiallabs/experiential/pull/804), a single source): empty or blank `tool_result` content, and `""` or `[]` assistant content in any position. | `text content blocks must be non-empty`, `text content blocks must contain non-whitespace text`, `user messages must have non-empty content` | Removes blank text blocks, then drops user messages left empty |
| `system_in_messages` | **Error** when the first message has `role: "system"`: no model accepts that. **Warning** elsewhere: mid-conversation system messages work on Claude Opus 4.8+, Opus 5/5.5, Fable 5/5.1 and Mythos 5/5.1, and are rejected by Sonnet 5 and older models. | `Unexpected role "system". The Messages API accepts a top-level system parameter…` | Not repaired. Move it to the top-level `system` parameter yourself. |
| `first_message_not_user` | Warning. The first-party API currently accepts a conversation that starts with an assistant message. Bedrock and some deployments still reject it. | `messages: first message must use the user role` | Not changed. When `fit()` drops turns, the kept part starts at a user turn. With `keep_first_turn=True` the pinned first turn stays as it was. |

**The API merges consecutive messages with the same role** into one turn before it checks the
rules ("Consecutive `user` or `assistant` turns in your request will be combined into a single
turn"). turnsafe does the same:
- Results spread over several user messages in a row are valid.
- Tool calls spread over several assistant messages in a row are one turn, answered by the user
  turn that follows.
- The merge happens first, so `[user: text] [user: tool_result]` becomes "text before result"
  and is rejected.

**Server tools need no client result.** This covers `server_tool_use` blocks with their
`web_search_tool_result`, `mcp_tool_use` and other result blocks, which the model produces
inside the same assistant message. turnsafe never splits an assistant message, so these
pairs always stay together.

Sources:
- [Handling tool calls](https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls)
- [Messages API](https://platform.claude.com/docs/en/api/messages)
- [Mid-conversation system messages](https://platform.claude.com/docs/en/build-with-claude/mid-conversation-system-messages)
- [API errors](https://platform.claude.com/docs/en/api/errors)
- Error reports quoting the exact text: [vercel/ai #8516](https://github.com/vercel/ai/issues/8516), [craft #14](https://github.com/craft-build/craft/issues/14), [PI-Desktop #718](https://github.com/vastsa/PI-Desktop/issues/718), [llm-anthropic #37](https://github.com/simonw/llm-anthropic/issues/37)

## Anthropic thinking blocks

This is the rule most likely to bite a trimming library.

- **Replay the latest assistant message's thinking unchanged.** Its `thinking` and
  `redacted_thinking` blocks must be sent back exactly as the model produced them.
  Otherwise: ``` `thinking` or `redacted_thinking` blocks in the latest assistant message cannot be modified```.
- **Prefix binding (Claude Fable 5.1 and Opus 5.5).** On these models a replayed thinking block
  is valid only while `system`, `tools` and *every earlier message* are unchanged. It is
  enforced by default for accounts created on or after 2026-08-31. Trimming from the front,
  dropping or editing an earlier message all invalidate that block and every later one:
  ``Invalid `signature` in `thinking` block. The block is bound to a different conversation.``
  The documented fix is to remove the thinking blocks from the first changed position onward.
  Putting a removed block back is never allowed.

Other models behave differently. Opus 4.5+ and Sonnet 4.6+ keep prior thinking but don't bind
it to the prefix. Older models and Haiku drop old thinking themselves. The docs don't agree on
whether removing *all* of the latest turn's thinking is allowed, and in manual extended thinking
mode a tool-loop turn without it is rejected:
``Expected `thinking` or `redacted_thinking`, but found `tool_use`.``

What turnsafe does:

- By default (`thinking="keep"`), turnsafe never touches thinking blocks.
- **With `thinking="strip_after_change"`**, for Fable 5.1 or Opus 5.5 with prefix binding:
  - `repair()` finds the first block it had to change. It removes the thinking blocks that come
    after that point and keeps those before it.
  - The most common repair, answering the calls of an interrupted last turn, changes only the
    message *after* the last assistant message. In that case no thinking is removed.
  - `fit()` removes thinking blocks from every kept message once it drops any turn. If nothing
    is dropped, nothing changes.
- Instead of stripping, you can set
  `thinking.block_binding.prefix_mismatch_behavior: "drop_block"` on the request, and the API
  drops the stale blocks itself.

Sources:
- [Extended thinking](https://platform.claude.com/docs/en/build-with-claude/thinking)
- [Preserved thinking](https://platform.claude.com/docs/en/build-with-claude/preserved-thinking)
- [Thinking troubleshooting](https://platform.claude.com/docs/en/build-with-claude/thinking-troubleshooting)

## Rules turnsafe does not check (v0.1)

- **Prefill.** Claude 4.6 and newer models reject a conversation that ends with an assistant
  message (`This model does not support assistant message prefill`). A stored history
  normally ends with the assistant's reply, and you append the next user message before
  sending, so turnsafe doesn't warn about it.
- **Where a mid-conversation `system` message may go.** It must follow a user turn, and be
  followed by an assistant turn or be last. turnsafe only checks that it doesn't split a tool
  call from its result.
- **A pending `server_tool_use`.** When an assistant turn ends on a `server_tool_use` that has
  no result yet, the next user message may contain only `tool_result` blocks.
- **OpenAI Responses API items and Gemini `contents`.** Both are planned.
