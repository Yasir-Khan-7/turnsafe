# AGENT.md

A guide for AI coding agents and human contributors working on `turnsafe`.

## What this project is

`turnsafe` is a zero-dependency Python library that validates, repairs and trims LLM chat
histories. Its purpose is to make sure a tool call is never separated from its result. It works on
plain provider message dicts: OpenAI Chat Completions and Anthropic Messages in v0.1.

Why it exists, and what was rejected in its favour, is in [docs/research.md](docs/research.md).
Every provider rule is listed with its source in [docs/rules.md](docs/rules.md).

## Commands

```bash
uv venv && uv pip install -e '.[dev]'   # set up
pytest -q                                # all offline tests, including property tests (~2 s)
mypy                                     # strict, src only
ruff check src tests && ruff format src tests
TURNSAFE_LIVE=1 pytest tests/test_live.py   # real API contract tests; needs API keys
python -m build && twine check dist/*       # build and check the release artefacts
```

A change is done only when `pytest`, `mypy` and `ruff check` all pass.

## Layout

| File | Responsibility |
|---|---|
| `src/turnsafe/_providers.py` | **All** format-specific knowledge. It has one `Adapter` per provider: where calls and results live, the allowed "zone" for results, what starts a turn, and how to build a placeholder. It also contains `normalize()` (deep copy, SDK objects to dicts) and `detect()`. |
| `src/turnsafe/_match.py` | Pairs every call with its result. `validate` and `repair` both use it, so they cannot disagree. |
| `src/turnsafe/validate.py` | Turns a `Match` plus provider-only rules into `Violation`s. |
| `src/turnsafe/repair.py` | Rebuilds the history from a `Match`. |
| `src/turnsafe/fit.py` | Splits the history into turns and drops whole turns, oldest first. |
| `src/turnsafe/_thinking.py` | Opt-in removal of Anthropic thinking blocks after the first change (prefix binding). |
| `src/turnsafe/tokens.py` | Offline token estimate. |
| `src/turnsafe/_types.py` | Public dataclasses, violation codes and exceptions. |
| `tests/builders.py` | Short constructors (`ou`, `oa`, `ot`, `au`, `aa`, `use`, `res`, `thinking`). Use them in tests. |
| `tests/test_properties.py` | Hypothesis tests of the invariants below. |

## Invariants: never break these

The property tests check all of them. If a change needs one relaxed, stop and ask a
maintainer first.

1. **Pure.** No public function mutates its input. `normalize()` deep-copies first.
2. **Repaired output validates.** `validate(repair(x))` has no errors except
   `system_in_messages` on the first message, which only the caller can move.
3. **Fixed point.** If `validate(x)` has no violations, then `repair(x) == x` and there are no fixes.
4. **Idempotent.** `repair(repair(x)) == repair(x)`.
5. **Assistant messages are sacred.** With the default `thinking="keep"`, `repair()` never
   edits an assistant message, except to remove a repeated tool call id or under
   `on_missing="strip"`. With `thinking="strip_after_change"`, the only extra edit is removing
   thinking blocks that come after the first changed block, because on models with prefix
   binding those blocks are invalid. Read the thinking section of `docs/rules.md` before
   touching this.
6. **Fitted output validates, and is a suffix.** `fit()` output is valid, and apart from pinned
   system messages and an optional pinned first turn, it is a contiguous tail of the input
   (with `thinking="strip_after_change"`, thinking blocks are removed once anything was dropped).
   It never splits a turn.
7. **Unknown content passes through.** Blocks and keys that turnsafe doesn't understand are
   copied unchanged, never dropped.
8. **No runtime dependencies.** Only the standard library. SDKs may be used in tests only.
9. **Linear time.** Matching, validation, repair and fit are linear in the history size.
   `tests/test_regressions.py` has timing guards for the pathological cases (many orphans,
   reused ids, thousands of parallel calls). Don't add per-turn scans over the whole history.

## Rules for changing provider behaviour

- **Every rule needs a source.** Add a row to `docs/rules.md` with the official doc URL and, where
  possible, the exact API error string. Don't encode a rule you can't source. If a rule is only
  sometimes enforced, make it a `warning`.
- **Put the rule in the right place.** Format knowledge goes in an `Adapter` method. A
  provider-only check that isn't about pairing goes in `validate.py` (see
  `_anthropic_rules`). Don't add `if provider == ...` branches to `_match.py` or `fit.py`.
- **Add a live case.** When you add or change a rule, add a case to `tests/test_live.py` that
  the real API rejects before repair and accepts after.
- **Adding a provider** (Responses, Gemini) means writing a new `Adapter` subclass, adding it
  to `ADAPTERS`, adding detection markers, adding a strategy in `test_properties.py`, and adding
  a section in `docs/rules.md`.

## Style

- Match the existing code: small functions, type hints everywhere (`mypy --strict`), docstrings
  on public functions, and comments only where the reason isn't obvious.
- Public API changes must update `__all__`, `README.md` and `CHANGELOG.md`.
- Violation codes and `Fix.action` strings are public API. Don't rename them outside a
  major release.
- Error and violation messages say what is wrong and where. `InvalidHistoryError` points the
  user to `repair()`.

## Releasing

1. Bump `version` in `pyproject.toml` and `__version__` in `src/turnsafe/__init__.py`.
2. Move the "Unreleased" entries in `CHANGELOG.md` under the new version.
3. Tag `vX.Y.Z` and push the tag. `.github/workflows/release.yml` builds the package and
   publishes it to PyPI through trusted publishing. Never publish from a laptop with a
   long-lived token.
