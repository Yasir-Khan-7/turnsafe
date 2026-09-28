"""Public result types and exceptions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Message = dict[str, Any]
Provider = Literal["openai", "anthropic"]
Severity = Literal["error", "warning"]

# Violation codes. Each one maps to a request the provider rejects (errors)
# or one that some deployments reject (warnings). See docs/rules.md.
UNANSWERED_TOOL_CALL = "unanswered_tool_call"
MISPLACED_TOOL_RESULT = "misplaced_tool_result"
ORPHAN_TOOL_RESULT = "orphan_tool_result"
DUPLICATE_TOOL_RESULT = "duplicate_tool_result"
DUPLICATE_TOOL_CALL_ID = "duplicate_tool_call_id"
TOOL_RESULT_NOT_FIRST = "tool_result_not_first"
SYSTEM_IN_MESSAGES = "system_in_messages"
EMPTY_CONTENT = "empty_content"
FIRST_MESSAGE_NOT_USER = "first_message_not_user"


@dataclass(frozen=True)
class Violation:
    """One rule the history breaks.

    ``index`` is the position of the offending message in the list that was
    validated. ``tool_call_id`` is set for rules about a specific tool call.
    """

    code: str
    message: str
    index: int
    severity: Severity = "error"
    tool_call_id: str | None = None

    def __str__(self) -> str:
        return f"messages[{self.index}] {self.code}: {self.message}"


@dataclass(frozen=True)
class Fix:
    """One change ``repair()`` made. ``index`` refers to the input list."""

    action: str
    detail: str
    index: int
    tool_call_id: str | None = None

    def __str__(self) -> str:
        return f"messages[{self.index}] {self.action}: {self.detail}"


@dataclass(frozen=True)
class Report:
    """The result of ``validate()``. Truthy when the history has no errors."""

    provider: Provider
    violations: tuple[Violation, ...] = field(default_factory=tuple)

    @property
    def errors(self) -> tuple[Violation, ...]:
        return tuple(v for v in self.violations if v.severity == "error")

    @property
    def warnings(self) -> tuple[Violation, ...]:
        return tuple(v for v in self.violations if v.severity == "warning")

    @property
    def ok(self) -> bool:
        return not self.errors

    def __bool__(self) -> bool:
        return self.ok

    def explain(self) -> str:
        if not self.violations:
            return f"{self.provider} history is valid."
        lines = [
            f"{self.provider} history has {len(self.errors)} error(s), "
            f"{len(self.warnings)} warning(s):"
        ]
        lines += [f"  [{v.severity}] {v}" for v in self.violations]
        return "\n".join(lines)

    def raise_for_errors(self) -> None:
        """Raise ``InvalidHistoryError`` if there is at least one error."""
        if self.errors:
            raise InvalidHistoryError(self)


class TurnsafeError(Exception):
    """Base class for turnsafe exceptions."""


class InvalidHistoryError(TurnsafeError, ValueError):
    """The history breaks provider rules. ``.report`` has the details."""

    def __init__(self, report: Report) -> None:
        self.report = report
        super().__init__(report.explain() + "\nCall turnsafe.repair() to fix it.")


class ContextBudgetError(TurnsafeError, ValueError):
    """Even the smallest history ``fit()`` may return is over budget."""

    def __init__(self, needed: int, max_tokens: int) -> None:
        self.needed = needed
        self.max_tokens = max_tokens
        super().__init__(
            f"The pinned messages and the latest turn need ~{needed} tokens, "
            f"over the budget of {max_tokens}. Shorten the latest turn or raise max_tokens."
        )


class ProviderDetectionError(TurnsafeError, ValueError):
    """``provider="auto"`` could not tell which message format this is."""
