"""Exception hierarchy.

Every error Holdout raises on purpose derives from :class:`HoldoutError`, so the CLI can
tell a refused or invalid operation (exit with a clear message) from a bug (traceback).
"""


class HoldoutError(Exception):
    """Base class for all deliberate Holdout failures."""


class DataValidationError(HoldoutError):
    """The input data cannot be used as given (missing target, unusable columns, ...)."""


class SplitError(HoldoutError):
    """A split specification is inconsistent with the data it is applied to."""


class SealError(HoldoutError):
    """Base class for holdout-seal violations."""


class HoldoutAlreadyUsedError(SealError):
    """The holdout of this run was already evaluated and ``force`` was not given."""


class HoldoutTamperedError(SealError):
    """The sealed holdout file no longer matches the sha256 recorded at sealing time."""


class HoldoutInvalidError(SealError):
    """The holdout was found contaminated (shares rows, groups or time with development
    data); it can never be evaluated, not even with ``force`` or an accepted risk."""


class PlanValidationError(HoldoutError):
    """A model plan failed data-aware validation; ``issues`` holds the typed reasons."""

    def __init__(self, message: str, issues: list[object]) -> None:
        super().__init__(message)
        self.issues = issues


class EvaluationError(HoldoutError):
    """Cross-validation or holdout scoring failed for a reason the user can act on."""


class LLMError(HoldoutError):
    """Base class for language-model failures (never silently replaced by a default)."""


class LLMConfigError(LLMError):
    """The LLM cannot be used as configured: no API key, an unknown or unpriced model."""


class LLMRequestError(LLMError):
    """The API rejected a request with a non-transient error (400, 401, 403, ...)."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class LLMUnavailableError(LLMError):
    """The service stayed unavailable (429, 5xx, timeouts) after every retry.

    The agent run is checkpointed and can be resumed; ``node`` is filled in by the graph.
    """

    def __init__(self, message: str, *, status: int | None = None, retries: int = 0) -> None:
        super().__init__(message)
        self.status = status
        self.retries = retries
        self.node: str | None = None


class LLMOutputError(LLMError):
    """The model's answer failed schema validation even after one repair attempt."""


class BudgetExceededError(LLMError):
    """A call would exceed the per-run or project spend cap."""


class AgentError(HoldoutError):
    """The agent stopped: a loop guard tripped, a node timed out, or a plan was rejected."""


class ExportError(HoldoutError):
    """An artifact (pipeline script, manifest, report) could not be produced."""
