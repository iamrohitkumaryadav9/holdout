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


class ExportError(HoldoutError):
    """An artifact (pipeline script, manifest, report) could not be produced."""
