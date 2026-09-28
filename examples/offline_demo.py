"""See turnsafe work without an API key: python examples/offline_demo.py

Three things that break real agents, and what turnsafe does about each.
"""

import turnsafe


def show(title: str, messages: list[dict]) -> None:
    print(f"\n=== {title} ===")
    print(turnsafe.validate(messages).explain())
    fixed, fixes = turnsafe.repair_detailed(messages)
    for fix in fixes:
        print(f"  fix: {fix}")
    print(f"  after repair: {turnsafe.validate(fixed).explain()}")


def call(call_id: str) -> dict:
    return {"id": call_id, "type": "function", "function": {"name": "search", "arguments": "{}"}}


# 1. Naive trimming: history[-4:] started on a tool result whose call was cut off.
show(
    "OpenAI: history[-4:] cut a tool call in half",
    [
        {"role": "tool", "tool_call_id": "call_1", "content": "3 results"},
        {"role": "assistant", "content": "I found 3 results."},
        {"role": "user", "content": "Summarise the first one."},
        {"role": "assistant", "content": "Sure..."},
    ],
)

# 2. An interrupted run: the process died while tool 2 of 2 was running.
show(
    "Anthropic: agent crashed between two parallel tool calls",
    [
        {"role": "user", "content": "Weather in Paris and Rome?"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "weather",
                    "input": {"city": "Paris"},
                },
                {"type": "tool_use", "id": "toolu_2", "name": "weather", "input": {"city": "Rome"}},
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "18C"}],
        },
    ],
)

# 3. The user typed while tools were running, and the message landed in between.
show(
    "OpenAI: a user message landed between a call and its result",
    [
        {"role": "user", "content": "Look up order 42."},
        {"role": "assistant", "content": None, "tool_calls": [call("call_7")]},
        {"role": "user", "content": "Also check order 43!"},
        {"role": "tool", "tool_call_id": "call_7", "content": "shipped"},
    ],
)

# Trimming that never splits a tool call from its result.
history = []
for n in range(20):
    history += [
        {"role": "user", "content": f"step {n} " + "details " * 40},
        {"role": "assistant", "content": None, "tool_calls": [call(f"call_{n}")]},
        {"role": "tool", "tool_call_id": f"call_{n}", "content": "ok " * 30},
        {"role": "assistant", "content": f"done with step {n}"},
    ]
result = turnsafe.fit_detailed(history, max_tokens=1_000, keep_first_turn=True)
print("\n=== fit() to 1,000 tokens ===")
print(
    f"kept {len(result.messages)} of {len(history)} messages (~{result.tokens} tokens), "
    f"dropped {result.dropped_turns} whole turns; valid: {bool(turnsafe.validate(result.messages))}"
)
