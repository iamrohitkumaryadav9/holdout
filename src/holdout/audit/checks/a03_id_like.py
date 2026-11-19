"""A3 ID-like column with signal: identifiers should not predict the outcome.

A column is ID-like when it is near-unique (or the profile already classified it as an
identifier). Supporting signals: an integer sequence without gaps, a uniform code shape
("AB-1234"), correlation with row order. An identifier that carries signal in A1 was
issued in an order related to the outcome (or encodes it), so a model can exploit it.

v1.1 (post-benchmark, with thresholds_v11): near-uniqueness alone is not enough; a column is
ID-like only with a pattern as well - an integer sequence, a uniform code, a monotonic row
order, or a key-like name - or when the profile classified it as an identifier. Messages
claim an outcome-related ordering only when the row-order correlation shows it.

Severity: signal and used by the plan -> critical; signal but dropped by the plan ->
warning (the pipeline is safe, but the data's ordering encodes the target); no signal ->
info (the default plan already drops profile-detected identifiers).
"""

import re
from collections.abc import Sequence

import numpy as np
import pandas as pd

from holdout.audit.checks import CheckOutcome
from holdout.audit.checks.a01_single_feature import SingleFeatureResult
from holdout.audit.context import AuditContext
from holdout.audit.types import (
    AuditFlag,
    Category,
    CheckId,
    DropColumns,
    IdLikeEvidence,
    Severity,
)
from holdout.core import config
from holdout.core.schemas import ColumnKind

_LETTER = re.compile(r"[A-Za-z]")
_DIGIT = re.compile(r"[0-9]")


def run(ctx: AuditContext, single: SingleFeatureResult | None) -> tuple[CheckOutcome, set[str]]:
    """Return the outcome and the set of ID-like columns (for A1 corroboration)."""
    t = ctx.thresholds.a03_id_like
    in_file_order = ctx.dev.sort_index()
    flags: list[AuditFlag] = []
    id_like: set[str] = set()
    for column in ctx.features:
        values = in_file_order[column]
        non_missing = values.dropna()
        if len(non_missing) < config.MIN_VALUES_FOR_RATIO_RULES:
            continue
        unique_ratio = non_missing.nunique() / len(non_missing)
        profile_id = _profile_kind(ctx, column) is ColumnKind.ID_LIKE
        # Continuous measurements are naturally near-unique; only integer-valued or
        # string columns can be identifiers by structure.
        near_unique = unique_ratio >= t.near_unique_ratio and _id_capable(non_missing)
        if not (near_unique or profile_id):
            continue
        signals = (["near_unique"] if near_unique else []) + (
            ["profile_id_like"] if profile_id else []
        )
        # Development rows miss the holdout's share of a sequence, so allow the gaps.
        span_ratio = t.integer_sequence_max_span_ratio / (1.0 - ctx.spec.test_size)
        if _integer_sequence(non_missing, span_ratio):
            signals.append("integer_sequence")
        if _uniform_code(non_missing, t.uniform_code_min_share):
            signals.append("uniform_code")
        spearman = _row_order_spearman(values)
        if spearman is not None and abs(spearman) >= t.row_order_min_abs_spearman:
            signals.append("row_order_correlation")
        v11 = ctx.thresholds_v11
        if v11 is not None:
            # v1.1 (post-benchmark): near-uniqueness alone is not an identifier. Integer
            # measurements (page counts, pixel coordinates) are naturally near-unique; an
            # identifier also shows a pattern (error analysis, section 3).
            if key_like_name(column, v11.a03_id_like.key_name_tokens):
                signals.append("key_like_name")
            if not (profile_id or set(signals) & _PATTERNS):
                continue
        id_like.add(column)
        score = single.scores.get(column) if single is not None else None
        has_signal = score is not None and score.normalized_score >= t.signal_min_normalized_score
        used = column in ctx.used_features
        if has_signal and used:
            severity = Severity.CRITICAL
            message = (
                f"{column!r} looks like an identifier ({', '.join(signals)}) yet predicts the "
                "target, and the plan uses it: the IDs were issued in an outcome-related order."
            )
        elif has_signal:
            severity = Severity.WARNING
            message = (
                f"{column!r} looks like an identifier and predicts the target; the plan drops "
                "it, but the data's ordering encodes the outcome."
            )
            if v11 is not None and "row_order_correlation" not in signals:
                # v1.1: claim an ordering only when the row-order correlation shows one.
                message = (
                    f"{column!r} looks like an identifier ({', '.join(signals)}) and predicts "
                    "the target; the plan drops it, so the pipeline does not use it."
                )
        else:
            severity = Severity.INFO
            message = f"{column!r} looks like an identifier ({', '.join(signals)}) with no signal."
        flags.append(
            AuditFlag(
                check=CheckId.A3,
                category=Category.ID_SIGNAL,
                severity=severity,
                columns=[column],
                evidence=IdLikeEvidence(
                    signals=signals,
                    unique_ratio=float(unique_ratio),
                    row_order_spearman=spearman,
                    single_feature_relative_lift=None if score is None else score.relative_lift,
                    single_feature_normalized_score=(
                        None if score is None else score.normalized_score
                    ),
                    used_by_plan=used,
                ),
                message=message,
                fix=DropColumns(columns=[column], reason="identifier; must not carry signal"),
            )
        )
    return CheckOutcome(flags=flags, detail=f"{len(id_like)} ID-like column(s)"), id_like


#: v1.1 identifier patterns: at least one is required besides near-uniqueness.
_PATTERNS = frozenset(
    {"integer_sequence", "uniform_code", "row_order_correlation", "key_like_name"}
)
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_TOKEN_SPLIT = re.compile(r"[_\-.\s]+")


def name_tokens(name: str) -> list[str]:
    """Lower-case name tokens, split on separators and camelCase ("patientNbr" -> patient,
    nbr)."""
    parts = _TOKEN_SPLIT.split(_CAMEL.sub("_", str(name)))
    return [p.lower() for p in parts if p]


def key_like_name(name: str, tokens: Sequence[str]) -> bool:
    """The name ends in a key token ("record_number", "encounter_id", "patientNbr") or is
    one ("id"). A key token elsewhere does not count: "number_vmail_messages" is a count."""
    parts = name_tokens(name)
    wanted = {t.lower() for t in tokens}
    return bool(parts) and parts[-1] in wanted


def _profile_kind(ctx: AuditContext, column: str) -> ColumnKind | None:
    try:
        return ctx.profile.column(column).kind
    except KeyError:
        return None


def _id_capable(values: pd.Series) -> bool:
    """Strings, or numbers that are all integers (not continuous measurements)."""
    if pd.api.types.is_bool_dtype(values):
        return False
    if not pd.api.types.is_numeric_dtype(values):
        return True
    return bool(np.all(np.mod(values.to_numpy(dtype="float64"), 1.0) == 0.0))


def _integer_sequence(values: pd.Series, max_span_ratio: float) -> bool:
    if not pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
        return False
    numbers = values.to_numpy(dtype="float64")
    if not np.all(np.mod(numbers, 1.0) == 0.0):
        return False
    span = numbers.max() - numbers.min() + 1
    return bool(span / max(1, len(np.unique(numbers))) <= max_span_ratio)


def _uniform_code(values: pd.Series, min_share: float) -> bool:
    if pd.api.types.is_numeric_dtype(values):
        return False
    shapes = values.astype(str).map(lambda v: _DIGIT.sub("9", _LETTER.sub("A", v)))
    return bool(shapes.value_counts(normalize=True).iloc[0] >= min_share)


def _row_order_spearman(values: pd.Series) -> float | None:
    if not pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
        return None
    numbers = values.astype("float64")
    mask = numbers.notna().to_numpy()
    if mask.sum() < 3 or numbers[mask].nunique() < 2:
        return None
    position = pd.Series(np.arange(len(values)), dtype="float64")[mask]
    return float(
        numbers[mask]
        .reset_index(drop=True)
        .corr(position.reset_index(drop=True), method="spearman")
    )
