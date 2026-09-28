# Contributing to turnsafe

Thanks for helping. This guide covers setting up, what a good change looks like, and how to
report a provider rule turnsafe gets wrong.

## Set up

```bash
git clone https://github.com/Yasir-Khan-7/turnsafe && cd turnsafe
uv venv && uv pip install -e '.[dev]'     # or: python -m venv .venv && pip install -e '.[dev]'
pytest -q                                 # fully offline, about 3 seconds
```

## Before you open a pull request

```bash
pytest -q
mypy
ruff check src tests && ruff format src tests
```

CI runs all of these on Python 3.10 to 3.14.

## What a good change looks like

- **Tests.** Every behaviour change comes with a test. If it touches `repair()` or `fit()`, also
  check that the property tests in `tests/test_properties.py` still hold, or extend their
  generators.
- **Cited rules.** Every provider rule needs a source. Add it to `docs/rules.md` with the
  official docs link, and where possible the exact API error text. If you can't source a rule,
  make it a warning, not an error.
- **Format code lives in one place.** Format-specific code goes in an `Adapter` in
  `src/turnsafe/_providers.py`, not in `if provider == ...` branches elsewhere.
- **Stable names.** Public names, violation codes and `Fix.action` strings are stable API. Note
  any change to them in `CHANGELOG.md`.

The invariants the code must keep are listed in [AGENT.md](AGENT.md#invariants-never-break-these).

## Reporting a wrong rule

The most useful issue you can open includes:
1. The exact API error text.
2. The model and platform: first-party API, Bedrock, Vertex or a gateway.
3. The smallest `messages` list that reproduces it. Redact the content, since only the
   structure matters.

## Live tests

`tests/test_live.py` sends real requests to check that repaired histories are accepted. It
costs a few tokens and is skipped by default:

```bash
TURNSAFE_LIVE=1 OPENAI_API_KEY=... ANTHROPIC_API_KEY=... pytest tests/test_live.py
```
