"""Offline plans: a sensible default, and the dummy baseline every plan is measured against.

``default_plan`` is what runs when no proposer is available (Week 1, or later without an
LLM key). It is deliberately conservative: a robust tree model, explicit drops for every
column the pipeline cannot use, and a metric chosen for the target's balance.
"""

from typing import Literal

from holdout.core import config
from holdout.core.schemas import ColumnKind, MetricName, Task
from holdout.data.profile import ColumnProfile, DatasetProfile
from holdout.plan.schema import (
    CategoricalPreprocessing,
    DropColumn,
    DummyModel,
    HistGBModel,
    ModelPlan,
    NoSelection,
    NumericPreprocessing,
    Rationale,
)

_DROP_REASONS = {
    ColumnKind.ID_LIKE: "identifier; unique per row or entity, so it cannot generalise",
    ColumnKind.DATETIME: "raw timestamp; Week 1 does not engineer date features",
    ColumnKind.TEXT: "free text; Week 1 has no text featurizer",
}


def default_plan(profile: DatasetProfile, task: Task | None = None) -> ModelPlan:
    """Return the offline default plan for ``profile`` (task defaults to the hint)."""
    task = task if task is not None else profile.task_hint
    drops = [d for d in (_drop_for(c) for c in profile.feature_columns) if d is not None]
    dropped = {d.column for d in drops}
    categorical = [
        c
        for c in profile.feature_columns
        if c.name not in dropped and c.kind in (ColumnKind.CATEGORICAL, ColumnKind.BOOLEAN)
    ]
    high_cardinality = any(c.n_unique > config.ONEHOT_MAX_CARDINALITY for c in categorical)
    encode: Literal["onehot", "target_cv"] = "target_cv" if high_cardinality else "onehot"
    metric = _default_metric(profile, task)
    minority = profile.minority_class_fraction
    imbalanced = (
        task is Task.CLASSIFICATION
        and minority is not None
        and minority < config.ACCURACY_MIN_MINORITY_FRACTION
    )
    return ModelPlan(
        task=task,
        metric=metric,
        drop_columns=drops,
        numeric=NumericPreprocessing(impute="median", scale="none"),
        categorical=CategoricalPreprocessing(impute="most_frequent", encode=encode),
        feature_selection=NoSelection(),
        model=HistGBModel(),
        class_weight="balanced" if imbalanced else "none",
        rationale=Rationale(
            metric=_metric_reason(metric),
            drop_columns=(
                f"dropped {len(drops)} column(s) the pipeline cannot use"
                if drops
                else "no drops needed"
            ),
            preprocessing=(
                "median imputation; trees need no scaling; "
                + (
                    "target encoding for high-cardinality categories"
                    if high_cardinality
                    else "one-hot categories"
                )
            ),
            feature_selection="none: gradient boosting is robust to irrelevant features",
            model="histogram gradient boosting: strong, fast default for tabular data",
            class_weight=(
                "balanced: minority class under 20%"
                if imbalanced
                else "classes reasonably balanced or regression"
            ),
        ),
    )


def baseline_plan(plan: ModelPlan) -> ModelPlan:
    """The dummy baseline for ``plan``: same task, metric and drops, a dummy model.

    Every reported score is read relative to this, so "0.71 ROC AUC" is never quoted
    without "vs 0.50 for predicting the prior".
    """
    return plan.model_copy(
        update={
            "model": DummyModel(),
            "feature_selection": NoSelection(),
            "class_weight": "none",
            "rationale": plan.rationale.model_copy(
                update={
                    "model": "dummy baseline: predicts the prior or the mean",
                    "feature_selection": "none: the baseline ignores features",
                    "class_weight": "none: the dummy model cannot use weights",
                }
            ),
        }
    )


def _drop_for(column: ColumnProfile) -> DropColumn | None:
    if column.kind in _DROP_REASONS:
        return DropColumn(column=column.name, reason=_DROP_REASONS[column.kind])
    if column.missing_rate == 1.0:
        return DropColumn(column=column.name, reason="entirely missing")
    if column.n_unique <= 1:
        return DropColumn(column=column.name, reason="constant; carries no information")
    return None


def _default_metric(profile: DatasetProfile, task: Task) -> MetricName:
    if task is Task.REGRESSION:
        return MetricName.RMSE
    if profile.n_classes is not None and profile.n_classes > 2:
        return MetricName.BALANCED_ACCURACY
    return MetricName.ROC_AUC


def _metric_reason(metric: MetricName) -> str:
    return {
        MetricName.RMSE: "rmse: penalises large errors, in the target's own units",
        MetricName.BALANCED_ACCURACY: "balanced accuracy: multiclass and robust to imbalance",
        MetricName.ROC_AUC: "roc_auc: threshold-free and insensitive to class balance",
    }[metric]
