"""A minimal OpenAI tool-calling loop that stores its history and never sends a broken one.

pip install turnsafe openai
OPENAI_API_KEY=... python examples/openai_agent.py
"""

import json
import pathlib

import openai

import turnsafe

MODEL = "gpt-4o-mini"
HISTORY = pathlib.Path("history.json")
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Current time in a city.",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
    }
]


def run_tool(name: str, args: dict) -> str:
    return f"12:00 in {args.get('city', 'somewhere')}"


client = openai.OpenAI()
# A history saved by an earlier run may be broken, for example if that run crashed mid-tool.
messages = json.loads(HISTORY.read_text()) if HISTORY.exists() else []
messages.append({"role": "user", "content": "What time is it in Tokyo and Lima?"})

while True:
    # repair() heals the stored history; fit() trims whole turns to stay in budget.
    request = turnsafe.fit(turnsafe.repair(messages), max_tokens=100_000, keep_first_turn=True)
    response = client.chat.completions.create(model=MODEL, tools=TOOLS, messages=request)
    message = response.choices[0].message
    messages.append(message.model_dump(exclude_none=True))
    HISTORY.write_text(json.dumps(messages))
    if not message.tool_calls:
        break
    for tool_call in message.tool_calls:
        args = json.loads(tool_call.function.arguments or "{}")
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": run_tool(tool_call.function.name, args),
            }
        )
    HISTORY.write_text(json.dumps(messages))

print(message.content)
