"""The portable kernel: import discipline, time keys, encoders' input, scoring, splitters."""

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
)

from holdout.core import portable

ALLOWED_THIRD_PARTY = {"numpy", "pandas", "sklearn", "lightgbm"}


def test_portable_imports_only_stdlib_and_the_modelling_stack() -> None:
    tree = ast.parse(Path(portable.__file__).read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
    disallowed = {m for m in modules if m not in sys.stdlib_module_names} - ALLOWED_THIRD_PARTY
    assert disallowed == set(), f"portable.py must stay embeddable; found {disallowed}"


def test_time_order_key_handles_numeric_datetime_and_strings() -> None:
    numeric = portable.time_order_key(pd.Series([3, 1, 2]))
    assert numeric.tolist() == [3.0, 1.0, 2.0]
    naive = portable.time_order_key(pd.Series(pd.to_datetime(["2024-01-02", "2024-01-01"])))
    assert str(naive.dt.tz) == "UTC"
    aware = portable.time_order_key(
        pd.Series(pd.to_datetime(["2024-01-02"]).tz_localize("Europe/Amsterdam"))
    )
    assert str(aware.dt.tz) == "UTC"
    iso = portable.time_order_key(pd.Series(["2024-01-01", "2024-02-01", None]))
    assert iso.notna().sum() == 2
    mixed = portable.time_order_key(pd.Series(["2024-01-01", "Feb 3 2024", "garbage"]))
    assert mixed.notna().sum() == 2


def test_time_order_key_rejects_booleans() -> None:
    with pytest.raises(ValueError, match="boolean"):
        portable.time_order_key(pd.Series([True, False], name="flag"))


def test_to_string_categories_unifies_types_and_keeps_missing() -> None:
    frame = pd.DataFrame({"c": [1, "1", None, 2.5]})
    out = portable.to_string_categories(frame)
    assert out["c"].tolist()[:2] == ["1", "1"]
    assert pd.isna(out["c"].iloc[2])
    assert out["c"].iloc[3] == "2.5"


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [
        ("stratified", StratifiedKFold),
        ("group", GroupKFold),
        ("time", TimeSeriesSplit),
        ("random", KFold),
    ],
)
def test_make_cv_matches_strategy(strategy: str, expected: type) -> None:
    splitter = portable.make_cv(strategy, 5, 123)
    assert isinstance(splitter, expected)
    assert splitter.get_n_splits() == 5


def test_unknown_names_are_rejected_loudly() -> None:
    with pytest.raises(ValueError, match="unknown split strategy"):
        portable.make_cv("bogus", 5, 1)
    with pytest.raises(ValueError, match="unknown metric"):
        portable.score("bogus", DummyClassifier().fit([[0], [1]], [0, 1]), pd.DataFrame(), None)  # type: ignore[arg-type]


@pytest.fixture
def binary_fit() -> tuple[LogisticRegression, pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"x": rng.normal(size=200)})
    y = pd.Series(np.where(X["x"] + rng.normal(scale=0.5, size=200) > 0, "yes", "no"))
    return LogisticRegression().fit(X, y), X, y


@pytest.mark.parametrize("metric", ["roc_auc", "pr_auc", "f1", "balanced_accuracy", "accuracy"])
def test_classification_metrics_use_classes_1_as_positive(
    binary_fit: tuple[LogisticRegression, pd.DataFrame, pd.Series], metric: str
) -> None:
    model, X, y = binary_fit
    value = portable.score(metric, model, X, y)
    assert 0.5 < value <= 1.0  # informative model on string labels


def test_multiclass_roc_is_ovr_and_pr_auc_is_refused() -> None:
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"x": rng.normal(size=300)})
    y = pd.Series(np.digitize(X["x"], [-0.5, 0.5]))
    model = LogisticRegression().fit(X, y)
    assert portable.score("roc_auc", model, X, y) > 0.8
    assert 0.0 <= portable.score("f1", model, X, y) <= 1.0
    with pytest.raises(ValueError, match="binary targets only"):
        portable.score("pr_auc", model, X, y)


def test_regression_metrics_have_natural_sign() -> None:
    from sklearn.linear_model import LinearRegression

    X = pd.DataFrame({"x": np.arange(10.0)})
    y = pd.Series(2 * np.arange(10.0) + 1)
    model = LinearRegression().fit(X, y)
    assert portable.score("rmse", model, X, y) == pytest.approx(0.0, abs=1e-9)
    assert portable.score("mae", model, X, y) == pytest.approx(0.0, abs=1e-9)
    assert portable.score("r2", model, X, y) == pytest.approx(1.0)


def test_load_csv_drops_missing_targets_and_reindexes(tmp_path: Path) -> None:
    path = tmp_path / "d.csv"
    pd.DataFrame({"x": [1, 2, 3], "y": [0, None, 1]}).to_csv(path, index=False)
    frame, dropped = portable.load_csv(path, "y")
    assert dropped == 1
    assert frame.index.tolist() == [0, 1]
    with pytest.raises(KeyError, match="target column 'nope' not found"):
        portable.load_csv(path, "nope")


def test_file_sha256_changes_with_content(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    path.write_bytes(b"a")
    first = portable.file_sha256(path)
    path.write_bytes(b"b")
    assert portable.file_sha256(path) != first
