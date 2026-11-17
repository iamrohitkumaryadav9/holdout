"""SplitSpec validation, proposal rules and apply_split edge cases."""

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from holdout.core.errors import SplitError
from holdout.core.schemas import DataType, SplitStrategy, Task
from holdout.core.seeds import derive_seed
from holdout.data.profile import profile_schema
from holdout.data.split import TIME_ORDER_WARNING, SplitSpec, apply_split, propose_split


@pytest.mark.parametrize("test_size", [0.05, 0.09, 0.41, 0.9])
def test_test_size_is_bounded(test_size: float) -> None:
    with pytest.raises(ValidationError, match="test_size"):
        SplitSpec(strategy=SplitStrategy.RANDOM, test_size=test_size, seed=1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"strategy": SplitStrategy.GROUP},
        {"strategy": SplitStrategy.TIME},
        {"strategy": SplitStrategy.RANDOM, "group_col": "g"},
        {"strategy": SplitStrategy.STRATIFIED, "time_col": "t"},
        {"strategy": SplitStrategy.GROUP, "group_col": "g", "time_col": "t"},
    ],
)
def test_strategy_requires_exactly_its_column(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValidationError, match="required for a"):
        SplitSpec(seed=1, **kwargs)  # type: ignore[arg-type]


def test_unknown_fields_rejected() -> None:
    with pytest.raises(ValidationError, match="extra"):
        SplitSpec(strategy=SplitStrategy.RANDOM, seed=1, shuffle=False)  # type: ignore[call-arg]


def test_to_config_derives_the_split_seed() -> None:
    spec = SplitSpec(strategy=SplitStrategy.RANDOM, seed=42)
    assert spec.to_config()["seed"] == derive_seed(42, "split")
    assert spec.fingerprint() == SplitSpec(strategy=SplitStrategy.RANDOM, seed=42).fingerprint()
    assert spec.fingerprint() != SplitSpec(strategy=SplitStrategy.RANDOM, seed=43).fingerprint()


# ------------------------------------------------------------------ proposal rules


def test_classification_defaults_to_stratified(churn_df: pd.DataFrame) -> None:
    profile = profile_schema(churn_df, "churned")
    proposal = propose_split(profile, Task.CLASSIFICATION, DataType.TABULAR, seed=1)
    assert proposal.spec.strategy is SplitStrategy.STRATIFIED
    assert proposal.warnings == []
    assert any("stratify" in r for r in proposal.reasons)


def test_regression_without_structure_is_random(churn_df: pd.DataFrame) -> None:
    profile = profile_schema(churn_df, "monthly_spend")
    proposal = propose_split(profile, Task.REGRESSION, DataType.TABULAR, seed=1)
    assert proposal.spec.strategy is SplitStrategy.RANDOM


def test_entity_column_suggests_group_split(grouped_df: pd.DataFrame) -> None:
    profile = profile_schema(grouped_df, "label")
    proposal = propose_split(profile, Task.CLASSIFICATION, DataType.TABULAR, seed=1)
    assert proposal.spec.strategy is SplitStrategy.GROUP
    assert proposal.spec.group_col == "customer_id"
    assert any("entity" in r for r in proposal.reasons)


def test_time_ordered_tabular_data_gets_the_exact_warning(sales_df: pd.DataFrame) -> None:
    profile = profile_schema(sales_df, "units_sold")
    proposal = propose_split(profile, Task.REGRESSION, DataType.TABULAR, seed=1)
    assert proposal.spec.strategy is SplitStrategy.RANDOM
    assert TIME_ORDER_WARNING in proposal.warnings
    assert TIME_ORDER_WARNING == (
        "This looks time-ordered; a random split may leak the future. Switch to a time-based split?"
    )
    assert proposal.alternatives == [
        SplitSpec(strategy=SplitStrategy.TIME, time_col="date", seed=1)
    ]


def test_declared_timeseries_uses_time_split(sales_df: pd.DataFrame) -> None:
    profile = profile_schema(sales_df, "units_sold")
    proposal = propose_split(profile, Task.REGRESSION, DataType.TIMESERIES, seed=1)
    assert proposal.spec == SplitSpec(strategy=SplitStrategy.TIME, time_col="date", seed=1)
    assert proposal.warnings == []


def test_declared_timeseries_without_time_column_warns(churn_df: pd.DataFrame) -> None:
    profile = profile_schema(churn_df, "churned")
    proposal = propose_split(profile, Task.CLASSIFICATION, DataType.TIMESERIES, seed=1)
    assert proposal.spec.strategy is SplitStrategy.STRATIFIED
    assert any("--split time:COLUMN" in w for w in proposal.warnings)


def test_monotonic_index_is_offered_as_time_alternative() -> None:
    frame = pd.DataFrame({"t": np.arange(100.0), "x": np.arange(100) % 3, "y": np.arange(100) % 2})
    profile = profile_schema(frame, "y")
    proposal = propose_split(profile, Task.CLASSIFICATION, DataType.TABULAR, seed=1)
    assert TIME_ORDER_WARNING in proposal.warnings
    assert proposal.alternatives[0].time_col == "t"


def test_rare_classes_warn() -> None:
    frame = pd.DataFrame({"x": np.arange(60) % 7, "y": ["a"] * 55 + ["b"] * 5})
    profile = profile_schema(frame, "y")
    proposal = propose_split(profile, Task.CLASSIFICATION, DataType.TABULAR, seed=1)
    assert any("fewer than 10 rows" in w for w in proposal.warnings)


# ------------------------------------------------------------------ apply_split


def test_apply_split_random_partitions_rows(churn_df: pd.DataFrame) -> None:
    spec = SplitSpec(strategy=SplitStrategy.RANDOM, seed=5)
    dev, holdout = apply_split(churn_df, spec, "churned")
    assert len(holdout) == 80
    assert sorted(dev.index.tolist() + holdout.index.tolist()) == churn_df.index.tolist()
    assert dev.index.is_monotonic_increasing


def test_apply_split_stratified_preserves_class_balance(churn_df: pd.DataFrame) -> None:
    spec = SplitSpec(strategy=SplitStrategy.STRATIFIED, seed=5)
    _, holdout = apply_split(churn_df, spec, "churned")
    assert holdout["churned"].mean() == pytest.approx(churn_df["churned"].mean(), abs=0.01)


def test_apply_split_time_holds_out_the_latest_rows(sales_df: pd.DataFrame) -> None:
    shuffled = sales_df.sample(frac=1.0, random_state=0)
    spec = SplitSpec(strategy=SplitStrategy.TIME, time_col="date", seed=5)
    dev, holdout = apply_split(shuffled, spec, "units_sold")
    assert dev["date"].max() <= holdout["date"].min()
    assert dev["date"].is_monotonic_increasing


@pytest.mark.parametrize(
    ("spec", "frame", "message"),
    [
        (
            SplitSpec(strategy=SplitStrategy.GROUP, group_col="nope", seed=1),
            pd.DataFrame({"x": range(20), "y": [0, 1] * 10}),
            "group_col 'nope' not found",
        ),
        (
            SplitSpec(strategy=SplitStrategy.TIME, time_col="y", seed=1),
            pd.DataFrame({"x": range(20), "y": [0, 1] * 10}),
            "cannot be the target",
        ),
        (
            SplitSpec(strategy=SplitStrategy.TIME, time_col="when", seed=1),
            pd.DataFrame({"when": ["2024-01-01", "soon"] * 10, "y": [0, 1] * 10}),
            "do not parse",
        ),
        (
            SplitSpec(strategy=SplitStrategy.TIME, time_col="flag", seed=1),
            pd.DataFrame({"flag": [True, False] * 10, "y": [0, 1] * 10}),
            "boolean",
        ),
        (
            SplitSpec(strategy=SplitStrategy.GROUP, group_col="g", seed=1),
            pd.DataFrame({"g": ["a", None] * 10, "y": [0, 1] * 10}),
            "missing value",
        ),
        (
            SplitSpec(strategy=SplitStrategy.GROUP, group_col="g", seed=1),
            pd.DataFrame({"g": ["a", "b", "c"] * 10, "y": [0, 1] * 15}),
            "has 3 groups",
        ),
        (
            SplitSpec(strategy=SplitStrategy.STRATIFIED, seed=1),
            pd.DataFrame({"x": range(20), "y": [0] * 19 + [1]}),
            "at least 2 rows per class",
        ),
        (
            SplitSpec(strategy=SplitStrategy.RANDOM, seed=1),
            pd.DataFrame({"x": range(8), "y": [0, 1] * 4}),
            "development rows",
        ),
        (
            SplitSpec(strategy=SplitStrategy.GROUP, group_col="g", seed=1),
            pd.DataFrame({"g": list("abcdef") * 4, "y": [0, 1] * 12}),
            "groups left for development",
        ),
    ],
)
def test_apply_split_errors_are_precise(spec: SplitSpec, frame: pd.DataFrame, message: str) -> None:
    with pytest.raises(SplitError, match=message):
        apply_split(frame, spec, "y")


def test_apply_split_missing_target() -> None:
    with pytest.raises(SplitError, match="target column 'y' not found"):
        apply_split(pd.DataFrame({"x": range(20)}), SplitSpec(strategy="random", seed=1), "y")  # type: ignore[arg-type]
