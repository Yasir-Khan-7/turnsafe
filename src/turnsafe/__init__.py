"""turnsafe: validate, repair and trim LLM chat history without orphaning tool calls.

    import turnsafe

    report = turnsafe.validate(messages)           # what the API would reject, and why
    messages = turnsafe.repair(messages)           # fix orphaned / missing / misplaced results
    messages = turnsafe.fit(messages, 100_000)     # trim whole turns to a token budget

Works on plain OpenAI Chat Completions and Anthropic Messages dicts. No
dependencies, no framework, no network.
"""

from ._providers import detect as detect_provider
from ._types import (
    DUPLICATE_TOOL_CALL_ID,
    DUPLICATE_TOOL_RESULT,
    EMPTY_CONTENT,
    FIRST_MESSAGE_NOT_USER,
    MISPLACED_TOOL_RESULT,
    ORPHAN_TOOL_RESULT,
    SYSTEM_IN_MESSAGES,
    TOOL_RESULT_NOT_FIRST,
    UNANSWERED_TOOL_CALL,
    ContextBudgetError,
    Fix,
    InvalidHistoryError,
    ProviderDetectionError,
    Report,
    TurnsafeError,
    Violation,
)
from .fit import FitResult, fit, fit_detailed
from .repair import DEFAULT_PLACEHOLDER, repair, repair_detailed
from .tokens import approx_tokens
from .validate import validate

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_PLACEHOLDER",
    "DUPLICATE_TOOL_CALL_ID",
    "DUPLICATE_TOOL_RESULT",
    "EMPTY_CONTENT",
    "FIRST_MESSAGE_NOT_USER",
    "MISPLACED_TOOL_RESULT",
    "ORPHAN_TOOL_RESULT",
    "SYSTEM_IN_MESSAGES",
    "TOOL_RESULT_NOT_FIRST",
    "UNANSWERED_TOOL_CALL",
    "ContextBudgetError",
    "FitResult",
    "Fix",
    "InvalidHistoryError",
    "ProviderDetectionError",
    "Report",
    "TurnsafeError",
    "Violation",
    "__version__",
    "approx_tokens",
    "detect_provider",
    "fit",
    "fit_detailed",
    "repair",
    "repair_detailed",
    "validate",
]
