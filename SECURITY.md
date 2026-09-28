# Security policy

## Supported versions

The latest minor release receives fixes.

## Reporting a vulnerability

Please don't open a public issue. Use GitHub's
[private vulnerability reporting](https://github.com/Yasir-Khan-7/turnsafe/security/advisories/new)
instead. Expect a first response within a few days.

## Scope

`turnsafe` works on data in memory only. It makes no network calls, runs no code from
messages, and has no runtime dependencies. The issues most relevant to it are:

- **Losing or changing message content** in a way the documentation doesn't describe. For
  example, a tool result attached to the wrong call, so the model is told the wrong outcome.
- **Denial of service through crafted histories.** For example, input that makes `validate()`,
  `repair()` or `fit()` take super-linear time.
