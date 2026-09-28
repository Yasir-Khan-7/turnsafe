"""A minimal Claude tool-use loop that stores its history and never sends a broken one.

pip install turnsafe anthropic
ANTHROPIC_API_KEY=... python examples/anthropic_agent.py
"""

import json
import pathlib

import anthropic

import turnsafe

MODEL = "claude-haiku-4-5"
HISTORY = pathlib.Path("history.json")
TOOLS = [
    {
        "name": "get_time",
        "description": "Current time in a city.",
        "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
    }
]


def run_tool(name: str, args: dict) -> str:
    return f"12:00 in {args.get('city', 'somewhere')}"


client = anthropic.Anthropic()
# A history saved by an earlier run may be broken, for example if that run crashed mid-tool.
messages = json.loads(HISTORY.read_text()) if HISTORY.exists() else []
messages.append({"role": "user", "content": "What time is it in Tokyo and Lima?"})

while True:
    # repair() heals the stored history; fit() trims whole turns to stay in budget.
    request = turnsafe.fit(turnsafe.repair(messages), max_tokens=150_000, keep_first_turn=True)
    response = client.messages.create(model=MODEL, max_tokens=1024, tools=TOOLS, messages=request)
    messages.append({"role": "assistant", "content": [b.model_dump() for b in response.content]})
    HISTORY.write_text(json.dumps(messages))
    if response.stop_reason != "tool_use":
        break
    results = [
        {"type": "tool_result", "tool_use_id": b.id, "content": run_tool(b.name, b.input)}
        for b in response.content
        if b.type == "tool_use"
    ]
    messages.append({"role": "user", "content": results})
    HISTORY.write_text(json.dumps(messages))

print(response.content[-1].text)
