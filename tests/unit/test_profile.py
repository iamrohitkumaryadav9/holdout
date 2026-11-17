"""Column-kind inference, task hint, time-order signals and entity detection."""

import numpy as np
import pandas as pd
import pytest

from holdout.core.errors import DataValidationError
from holdout.core.schemas import ColumnKind, Task
from holdout.data.profile import profile_schema


def _kinds(frame: pd.DataFrame, target: str) -> dict[str, ColumnKind]:
    return {c.name: c.kind for c in profile_schema(frame, target).columns}


def test_kinds_are_inferred_per_column() -> None:
    n = 200
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "amount": rng.normal(size=n),
            "color": rng.choice(["red", "blue"], n).tolist(),
            "flag": rng.choice(["yes", "no"], n),
            "is_new": rng.integers(0, 2, n),
            "real_bool": rng.integers(0, 2, n).astype(bool),
            "created": pd.date_range("2024-01-01", periods=n, freq="h").astype(str),
            "stamp": pd.date_range("2024-01-01", periods=n, freq="D"),
            "user_id": rng.integers(0, 20, n),
            "CustomerID": [f"k{i % 7}" for i in range(n)],
            "token": [f"tok-{i:05d}-{rng.integers(1e9)}" for i in range(n)],
            "note": [f"customer wrote a long free text comment number {i}" for i in range(n)],
            "zip_code": rng.choice(["1011", "2022", "3033"], n),
            "y": rng.integers(0, 2, n),
        }
    )
    kinds = _kinds(frame, "y")
    assert kinds["amount"] is ColumnKind.NUMERIC
    assert kinds["color"] is ColumnKind.CATEGORICAL
    assert kinds["flag"] is ColumnKind.BOOLEAN
    assert kinds["is_new"] is ColumnKind.BOOLEAN
    assert kinds["real_bool"] is ColumnKind.BOOLEAN
    assert kinds["created"] is ColumnKind.DATETIME
    assert kinds["stamp"] is ColumnKind.DATETIME
    assert kinds["user_id"] is ColumnKind.ID_LIKE
    assert kinds["CustomerID"] is ColumnKind.ID_LIKE
    assert kinds["token"] is ColumnKind.ID_LIKE
    assert kinds["note"] is ColumnKind.TEXT
    assert kinds["zip_code"] is ColumnKind.CATEGORICAL  # numeric codes are not dates


def test_monotonic_integer_counter_is_an_id_and_a_time_signal() -> None:
    frame = pd.DataFrame({"row": np.arange(100), "x": np.random.default_rng(0).normal(size=100)})
    frame["y"] = np.tile([0, 1], 50)
    profile = profile_schema(frame, "y")
    assert profile.column("row").kind is ColumnKind.ID_LIKE
    assert profile.monotonic_candidates == ["row"]
    assert profile.looks_time_ordered
    assert "monotonic" in profile.time_order_signals[0]


def test_column_statistics(churn_df: pd.DataFrame) -> None:
    profile = profile_schema(churn_df, "churned")
    age = profile.column("age")
    assert 0.0 < age.missing_rate < 0.1
    assert len(age.examples) == 5
    assert profile.n_rows == len(churn_df)
    assert profile.n_cols == churn_df.shape[1]
    assert [c.name for c in profile.feature_columns] == [
        c for c in churn_df.columns if c != "churned"
    ]
    with pytest.raises(KeyError, match="no column 'nope'"):
        profile.column("nope")


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([0, 1] * 50, Task.CLASSIFICATION),
        (["a", "b", "c", "d"] * 25, Task.CLASSIFICATION),
        (list(range(5)) * 20, Task.CLASSIFICATION),
        (list(np.linspace(0, 1, 100)), Task.REGRESSION),
        (list(range(100)), Task.REGRESSION),
    ],
)
def test_task_hint(values: list[object], expected: Task) -> None:
    frame = pd.DataFrame({"x": np.arange(len(values)) % 7, "y": values})
    assert profile_schema(frame, "y").task_hint is expected


def test_class_frequencies_only_for_classification(churn_df: pd.DataFrame) -> None:
    profile = profile_schema(churn_df, "churned")
    assert profile.target_class_counts is not None
    assert sum(profile.target_class_counts.values()) == len(churn_df)
    assert profile.n_classes == 2
    positive_rate = churn_df["churned"].mean()
    assert profile.minority_class_fraction == pytest.approx(min(positive_rate, 1 - positive_rate))
    regression = profile_schema(churn_df, "monthly_spend")
    assert regression.target_class_frequencies is None
    assert regression.n_classes is None
    assert regression.minority_class_fraction is None


def test_explicit_task_overrides_class_counting(sales_df: pd.DataFrame) -> None:
    sales_df["promo_flag"] = sales_df["promo"]
    forced = profile_schema(sales_df, "units_sold", task=Task.CLASSIFICATION)
    assert forced.task_hint is Task.REGRESSION
    assert forced.target_class_counts is not None


def test_entity_candidates(grouped_df: pd.DataFrame) -> None:
    profile = profile_schema(grouped_df, "label")
    assert profile.entity_candidates == ["customer_id"]


@pytest.mark.parametrize(
    ("frame", "target", "message"),
    [
        (pd.DataFrame({"x": [1, 2]}), "y", "target column 'y' not found"),
        (pd.DataFrame({"y": [1, 2]}), "y", "no feature columns"),
        (pd.DataFrame({"x": [1, 2], "y": [1, 1]}), "y", "fewer than 2 distinct"),
    ],
)
def test_invalid_inputs(frame: pd.DataFrame, target: str, message: str) -> None:
    with pytest.raises(DataValidationError, match=message):
        profile_schema(frame, target)


def test_duplicate_columns_rejected() -> None:
    frame = pd.DataFrame([[1, 2, 0], [3, 4, 1]], columns=["a", "a", "y"])
    with pytest.raises(DataValidationError, match="duplicate column names"):
        profile_schema(frame, "y")
