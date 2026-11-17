"""Compile a validated plan into a leakage-safe pipeline, and cross-validate it.

Leakage safety is structural: every learned step (imputers, scaler, encoder, selector,
model) is a step of one scikit-learn ``Pipeline``, and each CV fold fits a fresh clone on
that fold's training rows only. The splitter is derived from the same ``SplitSpec`` as the
holdout, so a group or time split can never be cross-validated with shuffled KFold.

The pipeline itself is built by :mod:`holdout.core.portable` from a plain config; this
module validates and translates. The exported ``pipeline.py`` embeds that same kernel, so it
rebuilds the identical pipeline by construction.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from holdout.core import config, portable
from holdout.core.errors import EvaluationError, PlanValidationError
from holdout.core.schemas import ColumnKind, CVResult
from holdout.core.seeds import derive_seed
from holdout.data.profile import DatasetProfile
from holdout.data.split import SplitSpec
from holdout.plan.metrics import get_metric
from holdout.plan.schema import KBestSelection, ModelPlan, Severity, validate_plan

_CATEGORICAL_KINDS = {ColumnKind.CATEGORICAL, ColumnKind.BOOLEAN}


def plan_to_config(
    plan: ModelPlan, profile: DatasetProfile, *, run_seed: int
) -> portable.PipelineConfig:
    """Translate a plan into the portable kernel's plain config.

    Raises:
        PlanValidationError: if :func:`validate_plan` reports any error.
    """
    issues = validate_plan(plan, profile)
    errors = [i for i in issues if i.severity is Severity.ERROR]
    if errors:
        details = "; ".join(f"[{i.code}] {i.message}" for i in errors)
        raise PlanValidationError(f"plan rejected: {details}", list(errors))
    kept = [c for c in profile.feature_columns if c.name not in plan.dropped]
    selection = plan.feature_selection
    return portable.PipelineConfig(
        task=plan.task.value,
        numeric_columns=[c.name for c in kept if c.kind is ColumnKind.NUMERIC],
        categorical_columns=[c.name for c in kept if c.kind in _CATEGORICAL_KINDS],
        numeric_impute=plan.numeric.impute,
        numeric_scale=plan.numeric.scale,
        categorical_impute=plan.categorical.impute,
        categorical_encode=plan.categorical.encode,
        selection_method=selection.method,
        selection_k=selection.k if isinstance(selection, KBestSelection) else None,
        model_name=plan.model.name,
        model_params=plan.model.model_dump(exclude={"name"}),
        class_weight=plan.class_weight,
        model_seed=derive_seed(run_seed, "model"),
        encoder_seed=derive_seed(run_seed, "target_encoder"),
        selector_seed=derive_seed(run_seed, "feature_selection"),
    )


def compile_plan(plan: ModelPlan, profile: DatasetProfile, *, run_seed: int) -> Pipeline:
    """Build the unfitted pipeline for ``plan`` (validating it first)."""
    return portable.build_pipeline(plan_to_config(plan, profile, run_seed=run_seed))


def cv_from_split(spec: SplitSpec) -> Any:
    """Return the 5-fold splitter matching the holdout split, seeded via ``derive_seed``.

    stratified -> StratifiedKFold, group -> GroupKFold, time -> TimeSeriesSplit,
    random -> KFold. Never a shuffled KFold for group or time data.
    """
    return portable.make_cv(spec.strategy.value, config.N_FOLDS, derive_seed(spec.seed, "cv"))


def evaluate_cv(
    plan: ModelPlan,
    dev_df: pd.DataFrame,
    target: str,
    spec: SplitSpec,
    profile: DatasetProfile,
    *,
    label: str = "plan",
) -> CVResult:
    """Cross-validate ``plan`` on the development rows only.

    Raises:
        PlanValidationError: if the plan is invalid for this data.
        EvaluationError: if a fold cannot be fitted or scored (e.g. a fold without both
            classes), with the fold number in the message.
    """
    pipeline = compile_plan(plan, profile, run_seed=spec.seed)
    X, y, groups = portable.cv_inputs(dev_df, target, spec.to_config())
    metric = get_metric(plan.metric)
    try:
        scores, seconds = portable.cross_validate_scores(
            pipeline, X, y, groups, cv_from_split(spec), metric.name.value
        )
    except (RuntimeError, ValueError) as exc:
        raise EvaluationError(f"cross-validation of {label!r} failed: {exc}") from exc
    return CVResult(
        plan_label=label,
        metric=metric.name,
        direction=metric.direction,
        mean=float(np.mean(scores)),
        std=float(np.std(scores)),
        fold_scores=scores,
        fit_seconds=seconds,
        n_rows=len(X),
        n_folds=len(scores),
        splitter=type(cv_from_split(spec)).__name__,
    )


def fit_final(
    plan: ModelPlan, dev_df: pd.DataFrame, target: str, spec: SplitSpec, profile: DatasetProfile
) -> Pipeline:
    """Fit ``plan`` on all development rows, for the single holdout evaluation."""
    pipeline = compile_plan(plan, profile, run_seed=spec.seed)
    X, y, _ = portable.cv_inputs(dev_df, target, spec.to_config())
    try:
        fitted: Pipeline = pipeline.fit(X, y)
    except ValueError as exc:
        raise EvaluationError(f"fitting the final pipeline failed: {exc}") from exc
    return fitted
