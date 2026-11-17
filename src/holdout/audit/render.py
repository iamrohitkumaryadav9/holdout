"""Plain-text rendering of fixes, shared by the CLI and report.md."""

from holdout.audit.types import (
    ChangeMetric,
    ChangeSplit,
    CleanFormats,
    Deduplicate,
    DropColumns,
    Fix,
    NoAction,
    NormaliseCategories,
    ParseNumeric,
    RemoveCausalClaim,
    ReplaceSentinel,
)


def fix_text(fix: Fix) -> str:
    match fix:
        case DropColumns(columns=columns):
            return "drop " + ", ".join(repr(c) for c in columns)
        case ChangeSplit(instruction=instruction):
            return instruction
        case Deduplicate(scope="before_split"):
            return "deduplicate the data before splitting, then start a new run"
        case Deduplicate():
            return "deduplicate development rows"
        case CleanFormats(column=column, action=ParseNumeric(strip_tokens=tokens, percent=pct)):
            stripped = ", ".join(repr(t) for t in tokens)
            return f"parse {column!r} as numbers (strip {stripped}{'; percent' if pct else ''})"
        case CleanFormats(column=column, action=NormaliseCategories(mapping=mapping)):
            return f"normalise {len(mapping)} spelling variant(s) in {column!r}"
        case CleanFormats(column=column, action=ReplaceSentinel(value=value)):
            return f"treat {value:g} in {column!r} as missing"
        case ChangeMetric(metric=metric):
            return f"use {metric.value}"
        case RemoveCausalClaim():
            return "remove the causal claim(s)"
        case NoAction(reason=reason):
            return reason
    raise AssertionError(f"unrendered fix {fix!r}")
