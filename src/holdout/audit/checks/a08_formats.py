"""A8 format issues: numbers stored as text, category spelling variants, sentinels.

These do not leak, but they silently degrade a pipeline: a numeric column read as text is
target-encoded as thousands of categories, "Mumbai" and " mumbai" become two levels, and
-999 is treated as a real (extreme) value. Each flag carries the exact cleaning action.
"""

import re

import numpy as np
import pandas as pd

from holdout.audit.checks import CheckOutcome
from holdout.audit.context import AuditContext
from holdout.audit.types import (
    AuditFlag,
    Category,
    CheckId,
    CleanFormats,
    FormatAction,
    FormatEvidence,
    FormatIssue,
    NormaliseCategories,
    ParseNumeric,
    ReplaceSentinel,
    Severity,
)

#: Currency tokens and unit suffixes removed before trying to parse a number.
_CURRENCY = ["rs.", "rs", "inr", "usd", "eur", "gbp", "$", "€", "£", "₹"]
_UNITS = ["kg", "km", "cm", "mm", "lbs", "lb", "g", "m", "%"]
_NUMBER = re.compile(r"^[-+]?\d+(\.\d+)?$")
_N_EXAMPLES = 5


def run(ctx: AuditContext) -> CheckOutcome:
    t = ctx.thresholds.a08_formats
    flags: list[AuditFlag] = []
    for column in ctx.features:
        values = ctx.dev[column]
        non_missing = values.dropna()
        if non_missing.empty:
            continue
        if pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values):
            flags.extend(_sentinels(ctx, column, non_missing.astype("float64")))
            continue
        text = non_missing.astype(str)
        numeric = _numeric_strings(text, t.numeric_parse_min_share)
        if numeric is not None:
            action, n_affected, examples = numeric
            flags.append(
                _flag(
                    column,
                    "numeric_strings",
                    n_affected,
                    len(values),
                    examples,
                    f"{column!r} holds numbers stored as text (e.g. {examples[0]!r}); the "
                    "pipeline treats them as categories.",
                    action,
                )
            )
            continue
        variants = _category_variants(text)
        if variants:
            mapping, n_affected = variants
            examples = sorted(mapping)[:_N_EXAMPLES]
            flags.append(
                _flag(
                    column,
                    "category_variants",
                    n_affected,
                    len(values),
                    examples,
                    f"{column!r} has spelling variants of the same category "
                    f"({', '.join(repr(e) for e in examples[:3])}); they are counted separately.",
                    NormaliseCategories(mapping=mapping),
                )
            )
    return CheckOutcome(flags=flags, detail=f"{len(flags)} format issue(s)")


def _clean_number(value: str) -> tuple[str, list[str], bool]:
    """Strip currency, units, separators; return (candidate, removed tokens, percent)."""
    text = value.strip().lower()
    removed: list[str] = []
    for token in _CURRENCY:
        if text.startswith(token):
            text = text[len(token) :].strip()
            removed.append(token)
            break
    percent = text.endswith("%")
    for unit in _UNITS:
        if text.endswith(unit):
            text = text[: -len(unit)].strip()
            removed.append(unit)
            break
    if "," in text:
        text = text.replace(",", "")
        removed.append(",")
    return text.replace(" ", ""), removed, percent


def _numeric_strings(
    text: pd.Series, min_share: float
) -> tuple[ParseNumeric, int, list[str]] | None:
    unique = text.unique()
    parsed = [_clean_number(v) for v in unique]
    ok = [
        (v, removed, pct)
        for v, (num, removed, pct) in zip(unique, parsed, strict=True)
        if _NUMBER.match(num)
    ]
    if not ok:
        return None
    counts = text.value_counts()
    parse_share = sum(int(counts[v]) for v, _, _ in ok) / len(text)
    needed_cleaning = [v for v, removed, _ in ok if removed]
    if parse_share < min_share or not needed_cleaning:
        return None
    tokens = sorted({token for _, removed, _ in ok for token in removed})
    percent = any(pct for _, _, pct in ok)
    n_affected = sum(int(counts[v]) for v in needed_cleaning)
    return (
        ParseNumeric(strip_tokens=tokens, percent=percent),
        n_affected,
        needed_cleaning[:_N_EXAMPLES],
    )


def _category_variants(text: pd.Series) -> tuple[dict[str, str], int] | None:
    counts = text.value_counts()
    folded = pd.Series({v: " ".join(v.split()).lower() for v in counts.index})
    mapping: dict[str, str] = {}
    n_affected = 0
    for _, group in folded.groupby(folded):
        raw = list(group.index)
        if len(raw) < 2:
            continue
        canonical = max(raw, key=lambda v: (int(counts[v]), v))
        for variant in raw:
            if variant != canonical:
                mapping[variant] = canonical
                n_affected += int(counts[variant])
    return (mapping, n_affected) if mapping else None


def _sentinels(ctx: AuditContext, column: str, values: pd.Series) -> list[AuditFlag]:
    t = ctx.thresholds.a08_formats
    flags: list[AuditFlag] = []
    for sentinel in t.sentinel_values:
        hits = values == sentinel
        share = float(hits.mean())
        if share < t.sentinel_min_share:
            continue
        others = values[~hits]
        if len(others) < 10:
            continue
        q25, median, q75 = np.percentile(others, [25, 50, 75])
        iqr = q75 - q25
        if iqr <= 0 or abs(sentinel - median) / iqr < t.sentinel_min_iqr_distance:
            continue
        flags.append(
            _flag(
                column,
                "sentinel",
                int(hits.sum()),
                len(ctx.dev),
                [str(sentinel)],
                f"{column!r} uses {sentinel:g} in {share:.1%} of rows, far outside its other "
                f"values (median {median:g}); it is probably a missing-value code.",
                ReplaceSentinel(value=sentinel),
            )
        )
    return flags


def _flag(
    column: str,
    issue: FormatIssue,
    n_affected: int,
    n_rows: int,
    examples: list[str],
    message: str,
    action: FormatAction,
) -> AuditFlag:
    return AuditFlag(
        check=CheckId.A8,
        category=Category.FORMAT_ISSUE,
        severity=Severity.WARNING,
        columns=[column],
        evidence=FormatEvidence(
            issue=issue,
            n_affected_rows=n_affected,
            share=n_affected / n_rows if n_rows else 0.0,
            examples=examples,
        ),
        message=message,
        fix=CleanFormats(column=column, action=action),
    )
