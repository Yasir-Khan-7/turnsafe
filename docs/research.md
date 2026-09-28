# Why turnsafe exists: research record

This page records how we chose to build `turnsafe`, what existing tools already do, and why the problem is still open. All research was done on 2026-09-28.

Download counts are last-month figures from the [pypistats.org API](https://pypistats.org/api/). Stars come from the GitHub API, and release dates from the PyPI JSON API. When a number could not be verified, we left it out.

## The problem

Agent code trims, summarises, saves, reloads and replays conversations. Sooner or later a history gets cut between an assistant turn that calls tools and the messages that carry those tools' results. The provider then rejects the request with a 400. The same broken history is sent again on the next turn, so the conversation stays broken until someone repairs it.

- Anthropic: `tool_use ids were found without tool_result blocks immediately after`
- OpenAI: `messages with role 'tool' must be a response to a preceeding message with 'tool_calls'` ("preceeding" is how the API spells it)

`turnsafe` v0.1 works on plain provider message dicts in two formats: OpenAI Chat Completions and Anthropic Messages. It offers three functions:

- **`validate()`** reports every rule the history breaks.
- **`repair()`** fixes the history using an explicit policy.
- **`fit()`** trims to a token budget, but only between whole turns. It never splits a tool call from its result.

Block types it doesn't recognise, such as thinking blocks and server-tool blocks, pass through untouched.

## Method

1. **Three researchers each took one area and five problems.** The areas were:
   - (A) LLM and agent tooling
   - (B) data engineering
   - (C) general developer workflow

   For each problem they collected:
   - evidence of pain (issues, forum threads, blog posts)
   - the existing packages, with downloads, stars and last release
   - where those packages fall short
   - whether a v0.1 could be built well in one or two days
   - whether a name was free on PyPI
2. **Scoring rubric: four criteria, 1 to 5 each, 20 points in total.**
   - **Pain:** how often it hurts, and how badly.
   - **Gap:** how much better a new package could do than what exists.
   - **Feasibility:** can a high-quality v0.1 be built in one or two days, pure Python, with a small dependency footprint, and fully testable offline.
   - **Reach:** how many Python developers would benefit.
3. **Cross-critique.** Each researcher attacked all three finalists, including their own, and searched live for competitors the original reports had missed. Each then rescored all three.

## All 15 problems, as first scored

| Area | Problem | P | G | F | R | Total | Main incumbents |
|---|---|---|---|---|---|---|---|
| A | **Validate, repair and trim chat history without splitting tool calls from their results** | 5 | 4 | 5 | 4 | **18** | LangChain `trim_messages`, pydantic-ai, openai-messages-token-helper, tokentrim (last release 2023) |
| A | Repair and partially parse broken JSON from LLMs | 4 | 1 | 4 | 4 | 13 | json-repair (28M/mo), partial-json-parser (5.9M/mo), jiter |
| A | Convert one schema into each provider's rules (OpenAI strict, Gemini) | 4 | 3 | 3 | 3 | 13 | pydantic-ai's `JsonSchemaTransformer`; the openai SDK's private `to_strict_json_schema` |
| A | Record and replay LLM HTTP calls in tests | 4 | 2 | 3 | 3 | 12 | vcrpy (23M/mo), pytest-recording, cassetter, pytest-llm-vcr |
| A | Token and cost accounting | 3 | 1 | 3 | 4 | 11 | genai-prices (23M/mo), litellm, tokencost |
| B | Find the real table in messy spreadsheets | 5 | 4 | 4 | 4 | **17** | messytables (last release 2017), d6tstack; pandas has no detection |
| B | Schema drift and data contracts in CI | 4 | 2 | 5 | 3 | 14 | pandera, great-expectations, datacontract-cli, dlt |
| B | Detect PII at the column level | 4 | 3 | 4 | 3 | 14 | presidio, scrubadub, dataprofiler |
| B | Sniff CSV dialect and encoding | 3 | 2 | 5 | 3 | 13 | clevercsv, charset-normalizer, DuckDB `sniff_csv` |
| B | Diff DataFrames | 3 | 1 | 5 | 3 | 12 | datacompy (3.5M/mo), daff |
| C | Verify webhook signatures from many providers | 4 | 3 | 5 | 3 | **15** | standardwebhooks (4.4M/mo), vendor SDKs |
| C | Rate limiting that reads the server's rate-limit headers | 4 | 2 | 4 | 4 | 14 | pyrate-limiter, aiolimiter, tenacity, stamina |
| C | Redact secrets and PII from logs | 4 | 2 | 5 | 2 | 13 | loggingredactor, maskerlogger, scrubadub |
| C | Graceful shutdown for workers | 3 | 3 | 3 | 3 | 12 | aiorun; frameworks (uvicorn, Celery) handle their own |
| C | Idempotency keys for HTTP APIs | 3 | 1 | 3 | 3 | 10 | asgi-idempotency-header, django-idempotency-key, fastapi-idempotency |

P = pain, G = gap, F = feasibility, R = reach. The finalists, in bold, were the top pick in each area.

## Evidence for the problem

- **LangChain.** `trim_messages` returns a history with a tool message whose assistant tool-call message was trimmed away ([langchain #33245](https://github.com/langchain-ai/langchain/issues/33245), open). See also [#29637](https://github.com/langchain-ai/langchain/issues/29637).
- **litellm.** Its own `trim_messages()` broke tool-call pairs repeatedly ([#4931](https://github.com/BerriAI/litellm/issues/4931), [#6505](https://github.com/BerriAI/litellm/issues/6505), [#8261](https://github.com/BerriAI/litellm/issues/8261), [#10696](https://github.com/BerriAI/litellm/issues/10696)), and [PR #11517](https://github.com/BerriAI/litellm/pull/11517) addressed it.
- **Naive slicing plus parallel tool calls** can make the kept window start on a tool result with no call ([lif-core #1326](https://github.com/LIF-Initiative/lif-core/issues/1326)).
- **Trimming at 50 messages split tool-call groups**, and the provider then rejected every later turn ([openhuman #6721](https://github.com/tinyhumansai/openhuman/issues/6721), marked P1).
- **Frameworks report the same error**, whether the cause is trimming, saved state or interrupted runs:
  - [llama_index #13715](https://github.com/run-llama/llama_index/issues/13715)
  - [semantic-kernel #4679](https://github.com/microsoft/semantic-kernel/issues/4679)
  - [autogen #2869](https://github.com/microsoft/autogen/discussions/2869)
  - [openai-agents-python #873](https://github.com/openai/openai-agents-python/issues/873)
  - [claude-code #7380](https://github.com/anthropics/claude-code/issues/7380)
- **Projects keep writing the same fix by hand:**
  - [alexkroman/agent #1527](https://github.com/alexkroman/agent/pull/1527)
  - [spora-core #273](https://github.com/spora-ai/spora-core/pull/273)
  - [agent-os #3233](https://github.com/use-agent-os/agent-os/issues/3233), a bug in a hand-written repair

**How much this demand is really worth is open to debate.** Searching GitHub for the Anthropic error string alone returns between about 900 and 1,600 issues, depending on the query. Most are closed, and many were fixed by the project itself in 30 to 80 lines. Some are automated noise. So we treat the count as evidence that the error is common, not as the size of our market.

## What we learned from existing tools

| Tool | What it does well | Lesson for turnsafe |
|---|---|---|
| [json-repair](https://github.com/mangiucugna/json_repair) (28M/mo) | One obvious entry point that mirrors `json.loads`. Narrow scope, very actively maintained. | Keep the API to a few verbs. Aim to become the default for one problem, not a framework. |
| LangChain `trim_messages` | Many options (`strategy`, `start_on`, `end_on`, `include_system`, ...). | Options don't guarantee a valid result: it can still return an orphaned tool message ([#33245](https://github.com/langchain-ai/langchain/issues/33245)) and has hit IndexErrors ([#26895](https://github.com/langchain-ai/langchain/issues/26895), [#30376](https://github.com/langchain-ai/langchain/issues/30376)). turnsafe makes validity a guarantee: `fit()` output must pass `validate()`. |
| litellm [message sanitization](https://docs.litellm.ai/docs/completion/message_sanitization) (`modify_params=True`) | Handles the three core fixes: add a placeholder result for an unanswered call, drop results with no call, fix empty content. | It is a global opt-in flag, applied only on litellm's Anthropic path, and hidden inside `completion()`. turnsafe makes repair a plain function you call explicitly and can inspect. |
| litellm `trim_messages` | Trims to model limits. | Four pairing bugs show that tool groups must be treated as one unit. Never pick individual messages out of a group. |
| pydantic-ai repair processor ([#4728](https://github.com/pydantic/pydantic-ai/issues/4728)) | Plugs into the framework's history-processing hook. Handles unmatched calls and orphaned results. | This validated the repair policies we adopted. It only works on pydantic-ai message types, so framework-free users still have nothing. |
| LangChain v1 [SummarizationMiddleware](https://reference.langchain.com/python/langchain/agents/middleware/summarization/SummarizationMiddleware) | Keeps assistant and tool message pairs together when summarising. | Pair-awareness is becoming table stakes, but ordering bugs remain ([langchainjs #9272](https://github.com/langchain-ai/langchainjs/issues/9272)). Test provider ordering rules too, not just pairing. |
| [cassetter](https://github.com/Kludex/cassetter) | "Safe by default". | The default policy must never produce a request the provider will reject. |
| [datacompy](https://github.com/capitalone/datacompy) | Returns a report that explains every difference. | `repair()` and `fit()` report what they changed and why, not just a new list. |
| [standardwebhooks](https://pypi.org/project/standardwebhooks/) | One spec absorbs many providers. | Convert each format to one internal model of turn groups, with a thin adapter per provider. |
| openai `to_strict_json_schema` | Correct, but private ([pydantic #10852](https://github.com/pydantic/pydantic/issues/10852)). | People want the fixes buried inside SDKs as a public, stable function. |
| tokentrim (last release 2023) | Simple. | It cut characters out of the middle of messages. turnsafe's `fit()` keeps or drops whole turns and, by default, never edits message content. |

## Finalists and the strongest case against each

**A: `turnsafe` (chat history validate, repair and fit)**
- **Frameworks are fixing this themselves.** litellm, pydantic-ai, LangChain v1 and the OpenAI Agents SDK now each handle it, and frameworks tend to copy a 50-line fix into their own code rather than add a dependency. The real audience is people who call the provider SDKs directly with their own agent loop.
- **Providers now do server-side context management.** Anthropic has [context editing and compaction](https://platform.claude.com/docs/en/build-with-claude/context-editing), and OpenAI has [Responses compaction](https://developers.openai.com/api/docs/guides/compaction). That shrinks the case for trimming on the client.
- **The message formats keep changing.** Thinking blocks with signatures, reasoning items and server-tool blocks all have their own rules. A wrong rule swaps one 400 for another.
- **Other people are building the same thing.** At least two clones exist on GitHub but not yet on PyPI: `akashvashisht/toolpair` (created 2026-09-27) and `MukundaKatta/agent-message-trim`.

**B: `headerhunt` (find the real table in a messy spreadsheet)**
- **A heuristic detector is only as good as its test files.** One or two days is enough for a demo, not a reliable tool, and a wrong guess silently corrupts data.
- **Pain is occasional.** Many people hard-code `skiprows` once, or ask an LLM to.
- **Adjacent tools exist:** [unstructured](https://github.com/Unstructured-IO/unstructured) `partition_xlsx(find_subtable=True)` and [eparse](https://github.com/ChrisPappalardo/eparse). Two tiny direct entrants appeared in September 2026: criba and xlsxflat.

**C: `hooksig` (verify webhook signatures from many providers)**
- **Per-user pain is shallow.** Most apps verify one or two providers with the vendor SDK they already have installed.
- **It is authentication code.** Teams won't readily trust a new package for it, and one bug is a security vulnerability.
- **Standard Webhooks is absorbing new providers.** At least four clones already exist on GitHub.

### Rescoring by all three critics (out of 20)

| Finalist | Original | Critic A | Critic B | Critic C | Mean |
|---|---|---|---|---|---|
| A: turnsafe | 18 | 15 | 16 | 16 | **15.7** |
| B: headerhunt | 17 | 15 | 15 | 15 | 15.0 |
| C: hooksig | 15 | 13 | 11 | 12 | 12.0 |

## Decision

**We are building A as `turnsafe`**, with a narrower scope. The reasons:

1. **All three critics ranked it first, including the researchers who proposed B and C.**
2. **The pain is concrete and easy to search for.** Developers paste the exact 400 error strings into search, which gives the documentation a natural path to be found.
3. **Correctness can be proven offline.** The key rule ("whatever the history and budget, `fit()` output passes `validate()`, and no tool group is ever split") is checked by property-based tests on thousands of randomly damaged histories. B's quality depends on real-world files we don't have.
4. **There is a clear standalone gap.** Every existing fix is tied to a framework, limited to one provider, or both.

Scope decisions from the critique round:
- Lead with `validate()` and `repair()`. These are the parts server-side compaction doesn't replace.
- Build `fit()` as a trimmer that only cuts between whole turns. Document provider-native compaction as the complementary option.
- Support OpenAI Chat Completions and Anthropic Messages only. The Responses API and Gemini wait for a later release.
- We picked the name `turnsafe` over `toolpair` to avoid confusion with an existing GitHub repo.

## How turnsafe differs

| | turnsafe | litellm `modify_params` | LangChain `trim_messages` | pydantic-ai repair processor |
|---|---|---|---|---|
| Needs a framework or client | No, plain dicts and zero dependencies | Yes, runs inside `litellm.completion()` | Yes, LangChain message objects | Yes, pydantic-ai message types |
| Providers | OpenAI Chat Completions and Anthropic Messages | Anthropic path only ([docs](https://docs.litellm.ai/docs/completion/message_sanitization)) | Any, through LangChain messages | Any pydantic-ai model |
| Reports rule violations without changing anything | Yes, `validate()` returns typed violations | No | No | No |
| Repairs orphaned calls and results | Yes, with an explicit policy | Yes, automatic | No | Yes |
| Trims to a token budget | Yes, `fit()` | No (litellm's separate `trim_messages` does) | Yes | No (not part of this processor) |
| Never splits a tool-call group | Guaranteed by design and by tests | n/a | Not guaranteed ([#33245](https://github.com/langchain-ai/langchain/issues/33245)) | n/a |
| Unknown blocks (thinking, server tools) | Passed through untouched with their turn | Not documented | Not documented | Not documented |
| Reports what it changed | Yes | No | No | Not documented |

If you already use one of these frameworks and its fix works for you, keep it. `turnsafe` is for code that talks to the provider SDKs directly, and for anyone who wants to check a history before sending it.

## Known risks and how v0.1 addresses them

| Risk | How v0.1 addresses it |
|---|---|
| **A wrong provider rule produces histories that pass `validate()` but still get a 400.** This is the hardest risk. | Only two formats. Each rule is a small named check, listed with its source in [rules.md](rules.md). Tests cover parallel calls, thinking blocks with tools, interrupted runs and naive trimming. An optional live test, off by default, sends repaired and trimmed histories to both real APIs. |
| **Invalidating thinking signatures.** Rules verification found that on Claude Fable 5.1 and Opus 5.5, a thinking block is bound to everything before it, so any trim invalidates later thinking. | `repair()` never edits an assistant message before the first change it makes. Thinking blocks are kept by default. `thinking="strip_after_change"` opts in to the documented recovery for prefix-bound models, removing only blocks after the first changed block. Property tests check both modes. Unknown block types pass through untouched. See [rules.md](rules.md#anthropic-thinking-blocks). |
| **Offline token counts are approximate**, since Anthropic has no local tokenizer. | The token counter is pluggable. The default estimate is conservative, and `fit()` reports its estimate instead of claiming to be exact. |
| **Framework fixes and server-side compaction shrink the audience.** | Position `turnsafe` as the framework-free option. Lead with `validate()`, which none of the others offer, and document how it complements compaction. |
| **Someone else publishes a similar package first.** | Ship a small, well-tested v0.1 quickly. Compete on correctness, the guaranteed invariant, and change reports, not on the number of features. |
| **Provider formats change.** | Each provider rule is a separate check, so adding or updating a format is additive. New formats ship in minor releases, backed by fixtures. |
