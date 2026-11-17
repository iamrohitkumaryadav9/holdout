"""ModelPlan shape validation (bounds, discriminated unions) and validate_plan issues."""

from typing import Any

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from holdout.core.schemas import MetricName, Task
from holdout.data.profile import DatasetProfile, profile_schema
from holdout.plan.defaults import default_plan
from holdout.plan.schema import (
    KBestSelection,
    LightGBMModel,
    LogRegModel,
    ModelPlan,
    PlanIssue,
    Severity,
    validate_plan,
)
from tests.helpers import RATIONALE, plan_dict


def codes(issues: list[PlanIssue], severity: Severity | None = None) -> set[str]:
    return {i.code for i in issues if severity is None or i.severity is severity}


@pytest.fixture
def imbalanced_profile() -> DatasetProfile:
    """Binary target with a 10% minority class."""
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "x": rng.normal(size=500),
            "color": rng.choice(["r", "g", "b"], 500),
            "y": np.r_[np.ones(50, dtype=int), np.zeros(450, dtype=int)],
        }
    )
    return profile_schema(frame, "y")


# ------------------------------------------------------------------ shape validation


def test_minimal_plan_parses_with_safe_defaults() -> None:
    plan = ModelPlan.model_validate(plan_dict())
    assert plan.numeric.impute == "median"
    assert plan.categorical.encode == "onehot"
    assert plan.feature_selection.method == "none"
    assert plan.class_weight == "none"


@pytest.mark.parametrize(
    "model",
    [
        {"name": "logreg", "C": 0.0},
        {"name": "logreg", "C": 1e5},
        {"name": "ridge", "alpha": -1.0},
        {"name": "random_forest", "n_estimators": 5},
        {"name": "random_forest", "max_depth": 0},
        {"name": "hist_gb", "learning_rate": 2.0},
        {"name": "hist_gb", "max_leaf_nodes": 1},
        {"name": "lightgbm", "num_leaves": 5000},
        {"name": "lightgbm", "subsample": 0.0},
    ],
)
def test_out_of_range_hyperparameters_are_rejected(model: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ModelPlan.model_validate(plan_dict(model=model))


@pytest.mark.parametrize(
    "model",
    [
        {"name": "xgboost"},
        {"name": "hist_gb", "n_estimators": 100},  # a LightGBM parameter on the wrong model
        {"C": 1.0},
    ],
)
def test_unknown_models_and_parameters_are_rejected(model: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ModelPlan.model_validate(plan_dict(model=model))


def test_discriminated_union_builds_the_right_submodel() -> None:
    plan = ModelPlan.model_validate(plan_dict(model={"name": "lightgbm", "num_leaves": 63}))
    assert isinstance(plan.model, LightGBMModel)
    assert plan.model.num_leaves == 63
    selection = ModelPlan.model_validate(
        plan_dict(feature_selection={"method": "k_best", "k": 5})
    ).feature_selection
    assert isinstance(selection, KBestSelection)
    with pytest.raises(ValidationError):
        ModelPlan.model_validate(plan_dict(feature_selection={"method": "k_best"}))


@pytest.mark.parametrize(
    "overrides",
    [
        {"metric": "logloss"},
        {"task": "ranking"},
        {"numeric": {"impute": "knn"}},
        {"categorical": {"encode": "hashing"}},
        {"class_weight": "auto"},
        {"drop_columns": [{"column": "x"}]},
        {"drop_columns": [{"column": "x", "reason": ""}]},
        {"drop_columns": [{"column": "x", "reason": "a"}, {"column": "x", "reason": "b c"}]},
        {"rationale": {**RATIONALE, "model": "two\nlines"}},
        {"rationale": {**RATIONALE, "model": "x" * 201}},
        {"rationale": {k: v for k, v in RATIONALE.items() if k != "metric"}},
        {"surprise": True},
    ],
)
def test_malformed_plans_are_rejected(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ModelPlan.model_validate(plan_dict(**overrides))


def test_plan_round_trips_through_json() -> None:
    plan = ModelPlan.model_validate(plan_dict(model={"name": "logreg", "C": 0.5}))
    again = ModelPlan.model_validate_json(plan.model_dump_json())
    assert again == plan
    assert isinstance(again.model, LogRegModel)


# ------------------------------------------------------------------ validate_plan


def test_accuracy_is_rejected_on_a_10_percent_minority(imbalanced_profile: DatasetProfile) -> None:
    plan = ModelPlan.model_validate(plan_dict(metric="accuracy"))
    issues = validate_plan(plan, imbalanced_profile)
    assert "accuracy_on_imbalanced_target" in codes(issues, Severity.ERROR)
    message = next(i.message for i in issues if i.code == "accuracy_on_imbalanced_target")
    assert "10.0%" in message
    assert "balanced_accuracy" in message


def test_accuracy_is_allowed_on_balanced_targets(churn_df: pd.DataFrame) -> None:
    churn_df["churned"] = np.arange(len(churn_df)) % 2
    profile = profile_schema(churn_df, "churned")
    plan = default_plan(profile).model_copy(update={"metric": MetricName.ACCURACY})
    assert "accuracy_on_imbalanced_target" not in codes(validate_plan(plan, profile))


def test_metric_must_match_task(imbalanced_profile: DatasetProfile) -> None:
    plan = ModelPlan.model_validate(plan_dict(metric="rmse"))
    assert "metric_task_mismatch" in codes(validate_plan(plan, imbalanced_profile), Severity.ERROR)


def test_pr_auc_needs_a_binary_target() -> None:
    frame = pd.DataFrame({"x": np.arange(90.0), "y": ["a", "b", "c"] * 30})
    profile = profile_schema(frame, "y")
    plan = ModelPlan.model_validate(plan_dict(metric="pr_auc"))
    assert "metric_needs_binary" in codes(validate_plan(plan, profile), Severity.ERROR)


@pytest.mark.parametrize(
    ("task", "metric", "model"),
    [("classification", "roc_auc", "ridge"), ("regression", "rmse", "logreg")],
)
def test_model_must_support_the_task(
    task: str, metric: str, model: str, churn_df: pd.DataFrame
) -> None:
    profile = profile_schema(churn_df, "churned")
    plan = ModelPlan.model_validate(plan_dict(task=task, metric=metric, model={"name": model}))
    assert "model_task_mismatch" in codes(validate_plan(plan, profile), Severity.ERROR)


@pytest.mark.parametrize(
    ("task", "metric", "model"),
    [("regression", "rmse", "hist_gb"), ("classification", "roc_auc", "dummy")],
)
def test_balanced_class_weight_needs_a_weightable_classifier(
    task: str, metric: str, model: str, churn_df: pd.DataFrame
) -> None:
    profile = profile_schema(churn_df, "churned")
    plan = ModelPlan.model_validate(
        plan_dict(task=task, metric=metric, model={"name": model}, class_weight="balanced")
    )
    assert "class_weight_not_applicable" in codes(validate_plan(plan, profile), Severity.ERROR)


def test_task_differing_from_hint_is_a_warning(churn_df: pd.DataFrame) -> None:
    profile = profile_schema(churn_df, "churned")
    plan = ModelPlan.model_validate(plan_dict(task="regression", metric="rmse"))
    assert "task_differs_from_hint" in codes(validate_plan(plan, profile), Severity.WARNING)


def test_drop_columns_must_exist_and_not_be_the_target(churn_df: pd.DataFrame) -> None:
    profile = profile_schema(churn_df, "churned")
    plan = ModelPlan.model_validate(
        plan_dict(
            drop_columns=[
                {"column": "churned", "reason": "oops"},
                {"column": "ghost", "reason": "typo"},
            ]
        )
    )
    issues = validate_plan(plan, profile)
    assert {"drop_target", "unknown_column"} <= codes(issues, Severity.ERROR)


def test_unmodellable_columns_must_be_dropped(sales_df: pd.DataFrame) -> None:
    profile = profile_schema(sales_df, "units_sold")
    plan = ModelPlan.model_validate(plan_dict(task="regression", metric="rmse"))
    issues = validate_plan(plan, profile)
    message = next(i.message for i in issues if i.code == "unmodellable_column")
    assert "'date' is datetime" in message


def test_dropping_everything_is_an_error(churn_df: pd.DataFrame) -> None:
    profile = profile_schema(churn_df, "churned")
    drops = [{"column": c.name, "reason": "test drop"} for c in profile.feature_columns]
    plan = ModelPlan.model_validate(plan_dict(drop_columns=drops))
    assert "no_features" in codes(validate_plan(plan, profile), Severity.ERROR)


@pytest.mark.parametrize("encode", ["onehot", "ordinal", "target_cv"])
def test_k_best_that_keeps_everything_warns(churn_df: pd.DataFrame, encode: str) -> None:
    profile = profile_schema(churn_df, "churned")
    plan = ModelPlan.model_validate(
        plan_dict(categorical={"encode": encode}, feature_selection={"method": "k_best", "k": 1000})
    )
    assert "k_best_keeps_everything" in codes(validate_plan(plan, profile), Severity.WARNING)
    small = plan.model_copy(update={"feature_selection": KBestSelection(k=2)})
    assert "k_best_keeps_everything" not in codes(validate_plan(small, profile))


def test_errors_are_listed_before_warnings(churn_df: pd.DataFrame) -> None:
    profile = profile_schema(churn_df, "churned")
    plan = ModelPlan.model_validate(plan_dict(task="regression", metric="roc_auc"))
    issues = validate_plan(plan, profile)
    severities = [i.severity for i in issues]
    assert severities == sorted(severities, key=lambda s: s is not Severity.ERROR)
    assert Severity.WARNING in severities


def test_default_plans_validate_cleanly(churn_df: pd.DataFrame, sales_df: pd.DataFrame) -> None:
    for frame, target in ((churn_df, "churned"), (sales_df, "units_sold")):
        profile = profile_schema(frame, target)
        assert codes(validate_plan(default_plan(profile), profile), Severity.ERROR) == set()


def test_default_plan_choices(imbalanced_profile: DatasetProfile, sales_df: pd.DataFrame) -> None:
    plan = default_plan(imbalanced_profile)
    assert plan.metric is MetricName.ROC_AUC
    assert plan.class_weight == "balanced"
    sales_plan = default_plan(profile_schema(sales_df, "units_sold"))
    assert sales_plan.task is Task.REGRESSION
    assert sales_plan.metric is MetricName.RMSE
    assert {d.column for d in sales_plan.drop_columns} == {"date"}
