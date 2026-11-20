"""Entity structure detected from the data's shape alone (no target needed).

A column looks like an entity key (customer, patient, device) when its values repeat over
several rows, its cardinality is moderate, and other columns are constant within each of
its values (age of a patient, plan of a customer). Used twice: before sealing, to warn
that a row-level split may put one entity on both sides; and by the group-leakage audit
(A5), to choose which columns to test with GroupKFold.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Other columns inspected per candidate when measuring entity-constant features.
_MAX_OTHER_COLUMNS = 200


@dataclass(frozen=True)
class EntityCandidate:
    column: str
    n_groups: int
    mean_rows_per_group: float
    unique_ratio: float
    constant_feature_share: float


def entity_candidates(
    frame: pd.DataFrame,
    exclude: list[str],
    *,
    min_mean_rows_per_value: float,
    min_groups: int,
    max_unique_ratio: float,
    max_candidates: int,
) -> list[EntityCandidate]:
    """Rank columns that look like entity keys, best first (at most ``max_candidates``).

    Eligible: not boolean, not continuous floats, no missing values, at least
    ``min_groups`` distinct values, at most ``max_unique_ratio`` distinct/rows, and at
    least ``min_mean_rows_per_value`` rows per value. Ranked by the share of other
    columns that are constant within every value, then by number of distinct values.
    """
    n_rows = len(frame)
    if n_rows == 0:
        return []
    excluded = set(exclude)
    eligible: list[tuple[str, int]] = []
    for column in frame.columns:
        name = str(column)
        values = frame[column]
        if name in excluded or values.isna().any() or pd.api.types.is_bool_dtype(values):
            continue
        if pd.api.types.is_float_dtype(values) and not _integral(values):
            continue
        n_unique = int(values.nunique())
        if (
            n_unique >= min_groups
            and n_unique / n_rows <= max_unique_ratio
            and n_rows / n_unique >= min_mean_rows_per_value
        ):
            eligible.append((name, n_unique))
    candidates = [
        EntityCandidate(
            column=name,
            n_groups=n_unique,
            mean_rows_per_group=n_rows / n_unique,
            unique_ratio=n_unique / n_rows,
            constant_feature_share=_constant_share(frame, name, excluded),
        )
        for name, n_unique in eligible
    ]
    candidates.sort(key=lambda c: (-c.constant_feature_share, -c.n_groups, c.column))
    return candidates[:max_candidates]


@dataclass(frozen=True)
class EntityKeyRule:
    """v1.1: what makes a structural candidate a plausible entity key (thresholds_v11)."""

    key_name_tokens: tuple[str, ...]
    min_constant_features: int


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


def plausible_entity_key(
    frame: pd.DataFrame, column: str, exclude: Sequence[str], rule: EntityKeyRule
) -> bool:
    """v1.1, the single rule for the split proposer and A5: a string or categorical column,
    or a numeric one with a key-like name, with entity-constant features. A repeated count
    or measurement is not an entity, whatever else happens to be constant with it."""
    values = frame[column]
    key_like = (
        not pd.api.types.is_numeric_dtype(values)
        or pd.api.types.is_bool_dtype(values)
        or key_like_name(column, rule.key_name_tokens)
    )
    return key_like and constant_feature_count(frame, column, list(exclude)) >= (
        rule.min_constant_features
    )


def entity_keys(
    frame: pd.DataFrame,
    exclude: list[str],
    rule: EntityKeyRule,
    *,
    min_mean_rows_per_value: float,
    min_groups: int,
    max_unique_ratio: float,
    max_candidates: int,
) -> list[EntityCandidate]:
    """v1.1: structural candidates that pass :func:`plausible_entity_key`, best first."""
    found = entity_candidates(
        frame,
        exclude,
        min_mean_rows_per_value=min_mean_rows_per_value,
        min_groups=min_groups,
        max_unique_ratio=max_unique_ratio,
        max_candidates=frame.shape[1],
    )
    return [c for c in found if plausible_entity_key(frame, c.column, exclude, rule)][
        :max_candidates
    ]


def _integral(values: pd.Series) -> bool:
    finite = values.to_numpy(dtype="float64")
    return bool(np.all(np.mod(finite, 1.0) == 0.0))


def _constant_share(frame: pd.DataFrame, column: str, excluded: set[str]) -> float:
    constant = _constant_columns(frame, column, excluded)
    return float(constant.mean()) if len(constant) else 0.0


def constant_feature_count(frame: pd.DataFrame, column: str, exclude: list[str]) -> int:
    """How many other (varying) columns are constant within every value of ``column``: the
    evidence that ``column`` keys an entity whose attributes repeat with it."""
    return int(_constant_columns(frame, column, set(exclude)).sum())


def _constant_columns(frame: pd.DataFrame, column: str, excluded: set[str]) -> pd.Series:
    others = [str(c) for c in frame.columns if str(c) != column and str(c) not in excluded]
    others = others[:_MAX_OTHER_COLUMNS]
    if not others:
        return pd.Series(dtype=bool)
    per_group = frame.groupby(column, sort=False)[others].nunique(dropna=False)
    return (per_group.max(axis=0) <= 1) & (frame[others].nunique(dropna=False) > 1)
