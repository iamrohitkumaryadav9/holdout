"""compile_plan structure, every model/encoder/selector combination, CV and errors."""

from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder

from holdout.core.errors import EvaluationError, PlanValidationError
from holdout.core.schemas import MetricName, SplitStrategy
from holdout.data.profile import profile_schema
from holdout.data.split import SplitSpec, apply_split
from holdout.plan.compiler import (
    compile_plan,
    cv_from_split,
    evaluate_cv,
    fit_final,
    plan_to_config,
)
from holdout.plan.defaults import baseline_plan, default_plan
from holdout.plan.schema import ModelPlan
from tests.helpers import plan_dict


def test_every_learned_step_lives_inside_the_pipeline(churn_df: pd.DataFrame) -> None:
    profile = profile_schema(churn_df, "churned")
    plan = ModelPlan.model_validate(
        plan_dict(
            numeric={"impute": "mean", "scale": "standard"},
            categorical={"impute": "constant", "encode": "target_cv"},
            feature_selection={"method": "k_best", "k": 4},
        )
    )
    pipeline = compile_plan(plan, profile, run_seed=1)
    assert [name for name, _ in pipeline.steps] == ["preprocess", "select", "model"]
    preprocess = pipeline.named_steps["preprocess"]
    assert isinstance(preprocess, ColumnTransformer)
    branches = {name: step for name, step, _columns in preprocess.transformers}
    numeric = branches["numeric"]
    categorical = branches["categorical"]
    assert [n for n, _ in numeric.steps] == ["impute", "scale"]
    assert [n for n, _ in categorical.steps] == ["as_text", "impute", "encode"]
    assert isinstance(categorical.named_steps["encode"], TargetEncoder)
    assert preprocess.remainder == "drop"


def test_config_routes_columns_by_kind_and_derives_seeds(churn_df: pd.DataFrame) -> None:
    profile = profile_schema(churn_df, "churned")
    config = plan_to_config(default_plan(profile), profile, run_seed=7)
    assert config["numeric_columns"] == ["age", "tenure_months", "monthly_spend", "support_tickets"]
    assert config["categorical_columns"] == ["plan_tier", "signup_channel", "region", "has_promo"]
    assert len({config["model_seed"], config["encoder_seed"], config["selector_seed"]}) == 3
    assert config == plan_to_config(default_plan(profile), profile, run_seed=7)


def test_invalid_plan_is_refused_with_typed_issues(churn_df: pd.DataFrame) -> None:
    profile = profile_schema(churn_df, "churned")
    plan = ModelPlan.model_validate(plan_dict(metric="rmse"))
    with pytest.raises(PlanValidationError, match="metric_task_mismatch") as info:
        compile_plan(plan, profile, run_seed=1)
    assert info.value.issues


CLASSIFICATION_MODELS: list[dict[str, Any]] = [
    {"name": "dummy"},
    {"name": "logreg", "C": 0.5},
    {"name": "random_forest", "n_estimators": 20, "max_depth": 4, "max_features": "all"},
    {"name": "hist_gb", "max_iter": 20},
    {"name": "lightgbm", "n_estimators": 20, "subsample": 0.8},
]


@pytest.mark.parametrize("model", CLASSIFICATION_MODELS)
@pytest.mark.parametrize("encode", ["onehot", "ordinal", "target_cv"])
def test_every_classifier_and_encoder_cross_validates(
    churn_df: pd.DataFrame, model: dict[str, Any], encode: str
) -> None:
    profile = profile_schema(churn_df, "churned")
    weight = "none" if model["name"] == "dummy" else "balanced"
    plan = ModelPlan.model_validate(
        plan_dict(
            model=model,
            class_weight=weight,
            numeric={"scale": "standard"},
            categorical={"encode": encode},
        )
    )
    spec = SplitSpec(strategy=SplitStrategy.STRATIFIED, seed=3)
    dev, _ = apply_split(churn_df, spec, "churned")
    result = evaluate_cv(plan, dev, "churned", spec, profile)
    assert result.n_folds == 5
    assert result.n_rows == len(dev)
    assert all(0.0 <= s <= 1.0 for s in result.fold_scores)
    assert result.mean == pytest.approx(np.mean(result.fold_scores))


@pytest.mark.parametrize(
    "model",
    [
        {"name": "dummy"},
        {"name": "ridge", "alpha": 3.0},
        {"name": "random_forest", "n_estimators": 20},
        {"name": "hist_gb", "max_iter": 20},
        {"name": "lightgbm", "n_estimators": 20},
    ],
)
@pytest.mark.parametrize(
    "selection", [{"method": "none"}, {"method": "k_best", "k": 3}, {"method": "model_based"}]
)
def test_every_regressor_and_selector_cross_validates(
    sales_df: pd.DataFrame, model: dict[str, Any], selection: dict[str, Any]
) -> None:
    profile = profile_schema(sales_df, "units_sold")
    plan = ModelPlan.model_validate(
        plan_dict(
            task="regression",
            metric="mae",
            model=model,
            feature_selection=selection,
            drop_columns=[{"column": "date", "reason": "raw timestamp"}],
        )
    )
    spec = SplitSpec(strategy=SplitStrategy.TIME, time_col="date", seed=3)
    dev, _ = apply_split(sales_df, spec, "units_sold")
    result = evaluate_cv(plan, dev, "units_sold", spec, profile)
    assert result.splitter == "TimeSeriesSplit"
    assert all(s >= 0 for s in result.fold_scores)


def test_plan_beats_the_dummy_baseline(churn_df: pd.DataFrame) -> None:
    profile = profile_schema(churn_df, "churned")
    plan = default_plan(profile)
    spec = SplitSpec(strategy=SplitStrategy.STRATIFIED, seed=3)
    dev, _ = apply_split(churn_df, spec, "churned")
    baseline = evaluate_cv(baseline_plan(plan), dev, "churned", spec, profile, label="baseline")
    candidate = evaluate_cv(plan, dev, "churned", spec, profile)
    assert baseline.mean == pytest.approx(0.5)
    assert candidate.mean > baseline.mean + 0.05
    assert baseline_plan(plan).model.name == "dummy"


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (SplitSpec(strategy=SplitStrategy.STRATIFIED, seed=1), StratifiedKFold),
        (SplitSpec(strategy=SplitStrategy.GROUP, group_col="g", seed=1), GroupKFold),
        (SplitSpec(strategy=SplitStrategy.TIME, time_col="t", seed=1), TimeSeriesSplit),
        (SplitSpec(strategy=SplitStrategy.RANDOM, seed=1), KFold),
    ],
)
def test_cv_from_split_mirrors_the_holdout_strategy(spec: SplitSpec, expected: type) -> None:
    splitter = cv_from_split(spec)
    assert type(splitter) is expected
    assert splitter.get_n_splits() == 5


def test_fold_failures_name_the_fold() -> None:
    # Time-ordered labels: the first forward-chaining fold trains on a single class.
    frame = pd.DataFrame(
        {
            "t": np.arange(120),
            "x": np.random.default_rng(0).normal(size=120),
            "y": [0] * 60 + [1] * 60,
        }
    )
    profile = profile_schema(frame, "y")
    plan = ModelPlan.model_validate(
        plan_dict(model={"name": "logreg"}, drop_columns=[{"column": "t", "reason": "time index"}])
    )
    spec = SplitSpec(strategy=SplitStrategy.TIME, time_col="t", seed=1)
    dev, _ = apply_split(frame, spec, "y")
    with pytest.raises(EvaluationError, match="fold 1"):
        evaluate_cv(plan, dev, "y", spec, profile)
    single_class = dev.iloc[:30]
    with pytest.raises(EvaluationError, match="fitting the final pipeline failed"):
        fit_final(plan, single_class, "y", spec, profile)


def test_fit_final_fits_on_all_dev_rows(churn_df: pd.DataFrame) -> None:
    profile = profile_schema(churn_df, "churned")
    spec = SplitSpec(strategy=SplitStrategy.STRATIFIED, seed=3)
    dev, _ = apply_split(churn_df, spec, "churned")
    fitted = fit_final(default_plan(profile), dev, "churned", spec, profile)
    assert isinstance(fitted, Pipeline)
    assert fitted.predict_proba(dev.drop(columns=["churned"])).shape == (len(dev), 2)


def test_multiclass_target_encoding(churn_df: pd.DataFrame) -> None:
    churn_df["tier_label"] = churn_df.pop("plan_tier")
    profile = profile_schema(churn_df, "tier_label")
    plan = ModelPlan.model_validate(
        plan_dict(
            metric="f1",
            categorical={"encode": "target_cv"},
            model={"name": "hist_gb", "max_iter": 20},
        )
    )
    spec = SplitSpec(strategy=SplitStrategy.STRATIFIED, seed=3)
    dev, _ = apply_split(churn_df, spec, "tier_label")
    assert evaluate_cv(plan, dev, "tier_label", spec, profile).metric is MetricName.F1
