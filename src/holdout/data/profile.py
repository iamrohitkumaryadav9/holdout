"""Target-blind schema profiling.

The profile is computed on the full dataset *before* the holdout is sealed, so it must not
contain any statistic that relates a feature to the target: such a statistic computed over
holdout rows would leak them into every later decision. Per-column statistics are
computed on each column alone; the only target statistics are its own distribution (task
hint and class frequencies), which are needed to choose a stratified split.

The test ``tests/property/test_profile_invariance.py`` enforces this: shuffling the target
column's values (which destroys every feature-target relation but keeps the target's own
distribution) must leave the profile unchanged.
"""

import re

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from holdout.core import config
from holdout.core.errors import DataValidationError
from holdout.core.portable import time_order_key
from holdout.core.schemas import ColumnKind, Task
from holdout.data.structure import entity_candidates as entity_structure_candidates

_ID_NAME = re.compile(r"(^id$|[_\-\s.]id$|^id[_\-\s.]|uuid|guid)", re.IGNORECASE)
_CAMEL_ID = re.compile(r"[a-z](ID|Id)$")
_NUMERIC_TEXT = re.compile(r"\s*[-+]?\d+(\.\d+)?\s*")
_BOOLEAN_TOKENS = frozenset({"true", "false", "yes", "no", "y", "n", "t", "f"})


class ColumnProfile(BaseModel):
    """Single-column facts. Every field is a function of this column alone."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    kind: ColumnKind
    dtype: str
    missing_rate: float = Field(ge=0.0, le=1.0)
    n_unique: int = Field(ge=0)
    unique_ratio: float = Field(ge=0.0, le=1.0, description="n_unique / non-missing count")
    examples: list[str] = Field(description="Most frequent values, ties broken by text order")


class StructuralEntity(BaseModel):
    """A column whose shape suggests it identifies repeated entities."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    column: str
    n_groups: int
    mean_rows_per_group: float
    constant_feature_share: float


class DatasetProfile(BaseModel):
    """Dataset-level facts used to propose a split and validate a plan."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    n_rows: int = Field(ge=2)
    n_cols: int = Field(ge=2)
    target: str
    task_hint: Task
    columns: list[ColumnProfile]
    target_class_counts: dict[str, int] | None = Field(
        description="Classification only; allowed because stratification needs it"
    )
    target_class_frequencies: dict[str, float] | None
    datetime_candidates: list[str]
    monotonic_candidates: list[str]
    entity_candidates: list[str]
    structural_entities: list[StructuralEntity] = Field(
        description="entity-like columns found from structure alone (repeats + constant features)"
    )
    looks_time_ordered: bool
    time_order_signals: list[str]

    def column(self, name: str) -> ColumnProfile:
        """Return the profile of ``name``; raise ``KeyError`` naming the known columns."""
        for column in self.columns:
            if column.name == name:
                return column
        raise KeyError(f"no column {name!r}; known columns: {[c.name for c in self.columns]}")

    @property
    def feature_columns(self) -> list[ColumnProfile]:
        return [c for c in self.columns if c.name != self.target]

    @property
    def n_classes(self) -> int | None:
        counts = self.target_class_counts
        return None if counts is None else len(counts)

    @property
    def minority_class_fraction(self) -> float | None:
        frequencies = self.target_class_frequencies
        return None if not frequencies else min(frequencies.values())


def profile_schema(frame: pd.DataFrame, target: str, task: Task | None = None) -> DatasetProfile:
    """Profile ``frame`` without computing any feature-target statistic.

    Args:
        frame: the full dataset (target included).
        target: name of the target column.
        task: explicit task; when given, class counts are computed for classification even
            if the hint says regression. ``task_hint`` itself is always the inferred value.

    Raises:
        DataValidationError: if the target is missing or constant, or there are no features.
    """
    if target not in frame.columns:
        raise DataValidationError(
            f"target column {target!r} not found; columns are {list(frame.columns)}"
        )
    if frame.shape[1] < 2:
        raise DataValidationError("the dataset has no feature columns besides the target")
    if frame.columns.duplicated().any():
        duplicated = sorted(set(frame.columns[frame.columns.duplicated()]))
        raise DataValidationError(f"duplicate column names: {duplicated}")
    target_values = frame[target]
    if target_values.nunique(dropna=True) < 2:
        raise DataValidationError(f"target {target!r} has fewer than 2 distinct values")

    columns = [_profile_column(str(name), frame[name]) for name in frame.columns]
    features = [c for c in columns if c.name != target]
    task_hint = _task_hint(target_values)
    effective_task = task if task is not None else task_hint
    counts: dict[str, int] | None = None
    frequencies: dict[str, float] | None = None
    if effective_task is Task.CLASSIFICATION:
        value_counts = target_values.astype(str).value_counts()
        counts = {str(k): int(v) for k, v in sorted(value_counts.items())}
        total = sum(counts.values())
        frequencies = {k: v / total for k, v in counts.items()}

    datetime_candidates = [c.name for c in features if c.kind is ColumnKind.DATETIME]
    monotonic_candidates = [c.name for c in features if _is_monotonic_index(frame[c.name], c)]
    entity_candidates = [
        c.name
        for c in features
        if c.kind is ColumnKind.ID_LIKE
        and c.missing_rate == 0.0
        and c.unique_ratio <= config.ENTITY_MAX_UNIQUE_RATIO
        and c.n_unique >= 2 * config.N_FOLDS
    ]
    structural = [
        StructuralEntity(
            column=c.column,
            n_groups=c.n_groups,
            mean_rows_per_group=c.mean_rows_per_group,
            constant_feature_share=c.constant_feature_share,
        )
        for c in entity_structure_candidates(
            frame,
            exclude=[target],
            min_mean_rows_per_value=config.STRUCTURAL_ENTITY_MIN_MEAN_ROWS,
            min_groups=config.STRUCTURAL_ENTITY_MIN_GROUPS,
            max_unique_ratio=config.ENTITY_MAX_UNIQUE_RATIO,
            max_candidates=config.STRUCTURAL_ENTITY_MAX_CANDIDATES,
        )
        # Datetime columns repeat too (several rows per day) but are time keys, handled by
        # the time-order logic; name-detected entity ids are already proposed for grouping.
        if c.constant_feature_share > 0.0
        and c.column not in entity_candidates
        and c.column not in datetime_candidates
    ]
    signals = [f"{name!r} parses as datetime" for name in datetime_candidates] + [
        f"{name!r} is a monotonic, near-unique numeric index" for name in monotonic_candidates
    ]
    return DatasetProfile(
        n_rows=len(frame),
        n_cols=frame.shape[1],
        target=target,
        task_hint=task_hint,
        columns=columns,
        target_class_counts=counts,
        target_class_frequencies=frequencies,
        datetime_candidates=datetime_candidates,
        monotonic_candidates=monotonic_candidates,
        entity_candidates=entity_candidates,
        structural_entities=structural,
        looks_time_ordered=bool(signals),
        time_order_signals=signals,
    )


def _profile_column(name: str, values: pd.Series) -> ColumnProfile:
    non_missing = values.dropna()
    n_unique = int(non_missing.nunique())
    unique_ratio = n_unique / len(non_missing) if len(non_missing) else 0.0
    return ColumnProfile(
        name=name,
        kind=_infer_kind(name, values, non_missing, n_unique, unique_ratio),
        dtype=str(values.dtype),
        missing_rate=float(values.isna().mean()),
        n_unique=n_unique,
        unique_ratio=unique_ratio,
        examples=_examples(non_missing),
    )


def _examples(non_missing: pd.Series) -> list[str]:
    """Most frequent values; order-independent so the profile is row-order invariant."""
    counts = non_missing.astype(str).value_counts()
    ranked = sorted(counts.items(), key=lambda item: (-int(item[1]), str(item[0])))
    return [
        str(value)[: config.EXAMPLE_MAX_CHARS] for value, _ in ranked[: config.N_EXAMPLE_VALUES]
    ]


def _looks_like_id_name(name: str) -> bool:
    return bool(_ID_NAME.search(name) or _CAMEL_ID.search(name))


def _infer_kind(
    name: str, values: pd.Series, non_missing: pd.Series, n_unique: int, unique_ratio: float
) -> ColumnKind:
    if pd.api.types.is_bool_dtype(values):
        return ColumnKind.BOOLEAN
    if pd.api.types.is_datetime64_any_dtype(values):
        return ColumnKind.DATETIME
    enough_values = len(non_missing) >= config.MIN_VALUES_FOR_RATIO_RULES
    if pd.api.types.is_numeric_dtype(values):
        if n_unique == 2 and set(non_missing.unique().tolist()) <= {0, 1}:
            return ColumnKind.BOOLEAN
        if pd.api.types.is_integer_dtype(values) and (
            _looks_like_id_name(name)
            or (
                enough_values
                and unique_ratio >= config.ID_UNIQUE_RATIO
                and non_missing.is_monotonic_increasing
            )
        ):
            return ColumnKind.ID_LIKE
        return ColumnKind.NUMERIC
    text = non_missing.astype(str)
    if n_unique == 2 and set(text.str.strip().str.lower().unique()) <= _BOOLEAN_TOKENS:
        return ColumnKind.BOOLEAN
    if _looks_like_id_name(name):
        return ColumnKind.ID_LIKE
    if _looks_like_datetime(text):
        return ColumnKind.DATETIME
    if (
        unique_ratio >= config.TEXT_MIN_UNIQUE_RATIO
        and text.str.len().mean() >= config.TEXT_MIN_MEAN_LENGTH
    ):
        return ColumnKind.TEXT
    if enough_values and unique_ratio >= config.ID_UNIQUE_RATIO:
        return ColumnKind.ID_LIKE
    return ColumnKind.CATEGORICAL


def _looks_like_datetime(text: pd.Series) -> bool:
    if text.empty:
        return False
    if len(text) > config.DATETIME_SAMPLE_SIZE:
        positions = np.linspace(0, len(text) - 1, config.DATETIME_SAMPLE_SIZE).astype(int)
        text = text.iloc[positions]
    if text.str.fullmatch(_NUMERIC_TEXT).mean() > 0.5:
        return False  # numeric codes ("2024", "17") parse as dates but are not timestamps
    parsed = time_order_key(text.reset_index(drop=True))
    return bool(parsed.notna().mean() >= config.DATETIME_MIN_PARSE_RATE)


def _is_monotonic_index(values: pd.Series, column: ColumnProfile) -> bool:
    if column.kind not in (ColumnKind.NUMERIC, ColumnKind.ID_LIKE):
        return False
    if not pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
        return False
    return (
        column.missing_rate == 0.0
        and column.unique_ratio >= config.MONOTONIC_MIN_UNIQUE_RATIO
        and len(values) >= config.MIN_VALUES_FOR_RATIO_RULES
        and (values.is_monotonic_increasing or values.is_monotonic_decreasing)
    )


def _task_hint(target: pd.Series) -> Task:
    non_missing = target.dropna()
    if pd.api.types.is_bool_dtype(target) or not pd.api.types.is_numeric_dtype(target):
        return Task.CLASSIFICATION
    integral = bool(np.all(np.mod(non_missing.to_numpy(dtype="float64"), 1.0) == 0.0))
    if integral and non_missing.nunique() <= config.CLASSIFICATION_MAX_CLASSES:
        return Task.CLASSIFICATION
    return Task.REGRESSION
