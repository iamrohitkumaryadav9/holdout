"""Leakage by construction: a spy records exactly which rows every fit sees.

A test-only transformer is prepended to the compiled pipeline. It reads a hidden row-id
column (which the plan drops, so the model never sees it) and records the ids at every
``fit``. The test then proves, for every split strategy and for plans that exercise every
learned step:

* each CV fit sees exactly that fold's training rows, never its validation rows;
* no fit (CV or final) ever sees a holdout row;
* the holdout evaluation itself performs no fit at all;
* inside the pipeline, the model receives exactly as many rows as the spy recorded.
"""

from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline

from holdout.core import portable
from holdout.core.schemas import SplitStrategy
from holdout.core.state import RunState
from holdout.data.examples import make_classification_example
from holdout.data.profile import DatasetProfile, profile_schema
from holdout.data.seal import SealedHoldout
from holdout.data.split import SplitSpec
from holdout.plan import compiler
from holdout.plan.metrics import get_metric
from holdout.plan.schema import ModelPlan
from tests.helpers import plan_dict

ROW_ID = "__row_id__"

#: Fit records live outside the estimators: sklearn's clone() deep-copies parameters,
#: so a list passed as a parameter would be copied away from the test.
FIT_LOG: dict[str, list[frozenset[int]]] = defaultdict(list)
MODEL_INPUT_ROWS: dict[str, list[int]] = defaultdict(list)


class RowIdSpy(TransformerMixin, BaseEstimator):  # type: ignore[misc]
    """Records the row ids of every fit; passes data through unchanged."""

    def __init__(self, key: str = "") -> None:
        self.key = key

    def fit(self, X: pd.DataFrame, y: Any = None) -> "RowIdSpy":
        FIT_LOG[self.key].append(frozenset(int(i) for i in X[ROW_ID]))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X


class RowCountSpy(TransformerMixin, BaseEstimator):  # type: ignore[misc]
    """Sits right before the model and records how many rows the model is fitted on."""

    def __init__(self, key: str = "") -> None:
        self.key = key

    def fit(self, X: Any, y: Any = None) -> "RowCountSpy":
        MODEL_INPUT_ROWS[self.key].append(int(np.asarray(X).shape[0]))
        return self

    def transform(self, X: Any) -> Any:
        return X


def _spied(key: str, original: Callable[..., Pipeline]) -> Callable[..., Pipeline]:
    def compile_with_spies(plan: ModelPlan, profile: DatasetProfile, *, run_seed: int) -> Pipeline:
        inner = original(plan, profile, run_seed=run_seed)
        inner.steps.insert(len(inner.steps) - 1, ("count_spy", RowCountSpy(key)))
        return Pipeline([("row_spy", RowIdSpy(key)), ("plan", inner)])

    return compile_with_spies


def _dataset() -> pd.DataFrame:
    frame = make_classification_example(n_rows=500, seed=11)
    frame["customer"] = [f"cust_{i % 60:02d}" for i in range(len(frame))]
    frame["when"] = pd.date_range("2023-01-01", periods=len(frame), freq="D").strftime("%Y-%m-%d")
    frame = frame.sample(frac=1.0, random_state=0).reset_index(drop=True)
    frame[ROW_ID] = frame.index.to_numpy()
    return frame


SPECS = {
    "stratified": SplitSpec(strategy=SplitStrategy.STRATIFIED, seed=5),
    "random": SplitSpec(strategy=SplitStrategy.RANDOM, seed=5),
    "group": SplitSpec(strategy=SplitStrategy.GROUP, group_col="customer", seed=5),
    "time": SplitSpec(strategy=SplitStrategy.TIME, time_col="when", seed=5),
}

DROPS = [
    {"column": ROW_ID, "reason": "hidden row id for the spy"},
    {"column": "customer", "reason": "entity id"},
    {"column": "when", "reason": "raw timestamp"},
]

PLANS = {
    "target_encoding_kbest": plan_dict(
        drop_columns=DROPS,
        numeric={"impute": "mean", "scale": "standard"},
        categorical={"impute": "constant", "encode": "target_cv"},
        feature_selection={"method": "k_best", "k": 5},
        model={"name": "logreg"},
    ),
    "onehot_model_based_lightgbm": plan_dict(
        drop_columns=DROPS,
        categorical={"encode": "onehot"},
        feature_selection={"method": "model_based"},
        model={"name": "lightgbm", "n_estimators": 20},
    ),
}


@pytest.mark.parametrize("plan_name", sorted(PLANS))
@pytest.mark.parametrize("split_name", sorted(SPECS))
def test_every_fit_sees_only_its_training_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, split_name: str, plan_name: str
) -> None:
    key = f"{split_name}/{plan_name}"
    FIT_LOG.pop(key, None)
    MODEL_INPUT_ROWS.pop(key, None)
    monkeypatch.setattr(compiler, "compile_plan", _spied(key, compiler.compile_plan))

    frame = _dataset()
    spec = SPECS[split_name]
    profile = profile_schema(frame, "churned")
    plan = ModelPlan.model_validate(PLANS[plan_name])
    state = RunState.create(tmp_path, run_seed=5)
    seal, dev = SealedHoldout.seal(frame, "churned", spec, state, data_sha256="sha")
    # The test (not the code under test) peeks at the sealed file to know the holdout ids.
    holdout_ids = frozenset(int(i) for i in pd.read_parquet(state.holdout_path)[ROW_ID])

    X, y, groups = portable.cv_inputs(dev, "churned", spec.to_config())
    expected_folds = [
        (
            frozenset(int(i) for i in X[ROW_ID].iloc[train]),
            frozenset(int(i) for i in X[ROW_ID].iloc[valid]),
        )
        for train, valid in compiler.cv_from_split(spec).split(X, y, groups)
    ]

    compiler.evaluate_cv(plan, dev, "churned", spec, profile)
    cv_fits = list(FIT_LOG[key])
    assert len(cv_fits) == len(expected_folds) == 5
    for seen, (train_ids, valid_ids) in zip(cv_fits, expected_folds, strict=True):
        assert seen == train_ids, "a fit saw rows outside its fold's training indices"
        assert seen.isdisjoint(valid_ids), "a fit saw validation rows"
        assert seen.isdisjoint(holdout_ids), "a fit saw holdout rows"
    assert MODEL_INPUT_ROWS[key] == [len(ids) for ids in cv_fits]

    final = compiler.fit_final(plan, dev, "churned", spec, profile)
    final_fit = FIT_LOG[key][-1]
    assert final_fit == frozenset(int(i) for i in dev[ROW_ID])
    assert final_fit.isdisjoint(holdout_ids), "the final fit saw holdout rows"

    n_fits_before = len(FIT_LOG[key])
    seal.evaluate(final, get_metric(plan.metric), caller="spy-test")
    assert len(FIT_LOG[key]) == n_fits_before, "evaluating the holdout must not fit anything"
    assert all(fit.isdisjoint(holdout_ids) for fit in FIT_LOG[key])


def test_negative_control_the_spy_catches_a_leaky_cv_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Swap in a CV loop that fits on all development rows: the spy must notice."""
    key = "negative-control"
    FIT_LOG.pop(key, None)
    monkeypatch.setattr(compiler, "compile_plan", _spied(key, compiler.compile_plan))

    def leaky_cross_validate(
        pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, groups: Any, cv: Any, metric: str
    ) -> tuple[list[float], float]:
        scores = []
        for _, valid in cv.split(X, y, groups):
            fitted = clone(pipeline).fit(X, y)  # the bug: fits on validation rows too
            scores.append(portable.score(metric, fitted, X.iloc[valid], y.iloc[valid]))
        return scores, 0.0

    monkeypatch.setattr(portable, "cross_validate_scores", leaky_cross_validate)
    frame = _dataset()
    spec = SPECS["stratified"]
    state = RunState.create(tmp_path, run_seed=5)
    _, dev = SealedHoldout.seal(frame, "churned", spec, state, data_sha256="sha")
    X, y, groups = portable.cv_inputs(dev, "churned", spec.to_config())
    valid_sets = [
        frozenset(int(i) for i in X[ROW_ID].iloc[valid])
        for _, valid in compiler.cv_from_split(spec).split(X, y, groups)
    ]

    compiler.evaluate_cv(
        ModelPlan.model_validate(PLANS["target_encoding_kbest"]),
        dev,
        "churned",
        spec,
        profile_schema(frame, "churned"),
    )
    leaked = [
        not seen.isdisjoint(valid) for seen, valid in zip(FIT_LOG[key], valid_sets, strict=True)
    ]
    assert all(leaked), "the spy failed to detect fits on validation rows"
