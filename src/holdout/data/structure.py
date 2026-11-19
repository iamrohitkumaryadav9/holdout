"""Entity structure detected from the data's shape alone (no target needed).

A column looks like an entity key (customer, patient, device) when its values repeat over
several rows, its cardinality is moderate, and other columns are constant within each of
its values (age of a patient, plan of a customer). Used twice: before sealing, to warn
that a row-level split may put one entity on both sides; and by the group-leakage audit
(A5), to choose which columns to test with GroupKFold.
"""

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
