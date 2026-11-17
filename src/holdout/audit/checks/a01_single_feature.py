"""A1 single-feature leakage: can one column alone do (almost) the whole job?

For each feature, a depth-limited decision tree on that feature alone is cross-validated
with the *same splitter as the run*. Its lift over the dummy is compared with the full
plan's lift (relative lift) and with the distance to a perfect score (normalized score).

Severity reasoning: some clean data really is near-perfectly predictable from one honest
feature (odor in the mushroom data), and statistics alone cannot tell that apart from a
leak. So A1 alone is at most a warning; it becomes critical only when corroborated by A2
(the name or values suggest a post-outcome column) or A3 (the column is ID-like). A
train/validation lift gap is deliberately *not* a corroborator: a genuine leak generalises
within development CV and shows no gap; a gap signals memorisation, which is A3's job.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, TargetEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from holdout.audit import cv as audit_cv
from holdout.audit.context import AuditContext
from holdout.audit.sampling import feature_cap, sample_rows, sampling_info, select_features
from holdout.audit.scoring import normalized, relative_lift
from holdout.audit.types import (
    AuditFlag,
    Category,
    CheckId,
    DropColumns,
    SamplingInfo,
    Severity,
    SingleFeatureEvidence,
)
from holdout.core import portable
from holdout.core.schemas import ColumnKind, Task
from holdout.plan.compiler import cv_from_split


@dataclass(frozen=True)
class FeatureScore:
    column: str
    score: float
    relative_lift: float | None
    normalized_score: float
    used_by_plan: bool


@dataclass(frozen=True)
class SingleFeatureResult:
    scores: dict[str, FeatureScore]
    sampling: SamplingInfo
    splitter: str


def measure(ctx: AuditContext) -> SingleFeatureResult:
    """Cross-validate a single-feature tree for every (capped) feature."""
    sample, row_method = sample_rows(ctx, "a01")
    columns, feature_method = select_features(
        sample, ctx.target, ctx.task, ctx.features, feature_cap(ctx)
    )
    X, y, groups = portable.cv_inputs(sample, ctx.target, ctx.spec.to_config())
    splitter = cv_from_split(ctx.spec)
    metric = ctx.metric
    used = ctx.used_features
    scores: dict[str, FeatureScore] = {}
    for column in columns:
        pipeline = _pipeline(ctx, column, X[column])
        folds = audit_cv.cross_validate(pipeline, X[[column]], y, groups, splitter, [metric.name])
        score = folds.mean(metric.name)
        if score is None:
            continue
        scores[column] = FeatureScore(
            column=column,
            score=score,
            relative_lift=relative_lift(metric, score, ctx.dummy_score, ctx.full_score),
            normalized_score=normalized(metric, score, ctx.dummy_score),
            used_by_plan=column in used,
        )
    return SingleFeatureResult(
        scores=scores,
        sampling=sampling_info(ctx, sample, row_method, len(columns), feature_method),
        splitter=type(splitter).__name__,
    )


def fires(ctx: AuditContext, score: FeatureScore) -> bool:
    """Whether one feature alone looks too strong (before corroboration)."""
    t = ctx.thresholds.a01_single_feature
    full_normalized = normalized(ctx.metric, ctx.full_score, ctx.dummy_score)
    relative_rule = (
        score.relative_lift is not None
        and full_normalized >= t.min_full_normalized_lift
        and score.relative_lift >= t.warn_relative_lift
    )
    return relative_rule or score.normalized_score >= t.warn_normalized_score


def flags(
    ctx: AuditContext,
    result: SingleFeatureResult,
    corroboration: Mapping[str, list[CheckId]],
) -> list[AuditFlag]:
    """Flags for the features that fire, upgraded to critical when corroborated."""
    metric = ctx.metric
    out: list[AuditFlag] = []
    for column, score in sorted(result.scores.items()):
        if not fires(ctx, score):
            continue
        corroborators = sorted(corroboration.get(column, []))
        if not score.used_by_plan:
            severity = Severity.INFO
        elif corroborators:
            severity = Severity.CRITICAL
        else:
            severity = Severity.WARNING
        share = (
            f"{score.relative_lift:.0%} of the plan's lift"
            if score.relative_lift is not None
            else f"{score.normalized_score:.0%} of the way to a perfect score"
        )
        message = (
            f"{column!r} alone reaches {metric.name.value} {score.score:.3f} "
            f"(dummy {ctx.dummy_score:.3f}, full plan {ctx.full_score:.3f}): {share}."
        )
        if severity is Severity.CRITICAL:
            names = " and ".join(c.value for c in corroborators)
            message += f" {names} also point to leakage, so treat it as a leak."
        elif severity is Severity.WARNING:
            message += (
                " This can be an honest, very strong feature;"
                " confirm it is known at prediction time."
            )
        else:
            message += " The plan already drops it."
        out.append(
            AuditFlag(
                check=CheckId.A1,
                category=Category.TARGET_LEAKAGE,
                severity=severity,
                columns=[column],
                evidence=SingleFeatureEvidence(
                    metric=metric.name,
                    single_score=score.score,
                    dummy_score=ctx.dummy_score,
                    full_score=ctx.full_score,
                    relative_lift=score.relative_lift,
                    normalized_score=score.normalized_score,
                    used_by_plan=score.used_by_plan,
                    corroborated_by=corroborators,
                    splitter=result.splitter,
                    sampling=result.sampling,
                ),
                message=message,
                fix=DropColumns(
                    columns=[column],
                    reason=(
                        "single-feature predictive power suggests information"
                        " from after the outcome"
                    ),
                ),
            )
        )
    return out


def _as_float(X: pd.DataFrame) -> np.ndarray:
    return np.asarray(X, dtype="float64")


def _time_as_float(X: pd.DataFrame) -> np.ndarray:
    """Stateless: parse each value as a timestamp and use nanoseconds as a number."""
    frame = pd.DataFrame(X)
    columns = []
    for name in frame.columns:
        key = portable.time_order_key(frame[name])
        if pd.api.types.is_datetime64_any_dtype(key):
            numeric = key.astype("int64").astype("float64").where(key.notna())
            columns.append(numeric.to_numpy())
        else:
            columns.append(key.to_numpy(dtype="float64"))
    return np.column_stack(columns)


def _pipeline(ctx: AuditContext, column: str, values: pd.Series) -> Pipeline:
    t = ctx.thresholds.a01_single_feature
    seed = ctx.seed(f"a01/{column}")
    tree: DecisionTreeClassifier | DecisionTreeRegressor = (
        DecisionTreeClassifier(max_depth=t.tree_max_depth, random_state=seed)
        if ctx.task is Task.CLASSIFICATION
        else DecisionTreeRegressor(max_depth=t.tree_max_depth, random_state=seed)
    )
    kind = _kind(ctx, column)
    if pd.api.types.is_numeric_dtype(values):
        prep: list[tuple[str, object]] = [("as_float", FunctionTransformer(_as_float))]
    elif kind is ColumnKind.DATETIME:
        prep = [("as_time", FunctionTransformer(_time_as_float))]
    else:
        prep = [
            ("as_text", FunctionTransformer(portable.to_string_categories)),
            (
                "impute_text",
                SimpleImputer(strategy="constant", fill_value=portable.MISSING_CATEGORY),
            ),
            ("encode", TargetEncoder(cv=5, shuffle=True, random_state=seed)),
        ]
    steps = [
        *prep,
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("tree", tree),
    ]
    return Pipeline(steps)


def _kind(ctx: AuditContext, column: str) -> ColumnKind | None:
    try:
        return ctx.profile.column(column).kind
    except KeyError:
        return None
