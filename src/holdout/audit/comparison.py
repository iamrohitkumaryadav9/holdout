"""Compare the plan's CV score under a reference splitter and a stricter alternative.

Used by A5 (GroupKFold) and A6 (forward-chaining folds). The drop is expressed relative
to the plan's lift over the dummy under the reference splitter, so "0.4" means the
stricter split removes 40% of what the plan claims to have learned, on any metric.
"""

from typing import Any

import pandas as pd

from holdout.audit import cv as audit_cv
from holdout.audit.context import AuditContext
from holdout.audit.scoring import lift
from holdout.audit.types import SamplingInfo, Severity, SplitComparison

_EPSILON = 1e-12


def plan_cv(
    ctx: AuditContext,
    pipeline: Any,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series | None,
    splitter: Any,
) -> audit_cv.FoldScores:
    return audit_cv.cross_validate(pipeline, X, y, groups, splitter, [ctx.metric.name])


def compare(
    ctx: AuditContext,
    reference_name: str,
    reference: audit_cv.FoldScores,
    alternative_name: str,
    alternative: audit_cv.FoldScores,
    sampling: SamplingInfo,
) -> SplitComparison | None:
    """Build the comparison, or None when either side produced no scorable fold."""
    metric = ctx.metric
    ref = reference.mean(metric.name)
    alt = alternative.mean(metric.name)
    if ref is None or alt is None:
        return None
    absolute = metric.improvement(ref, alt)
    reference_lift = lift(metric, ref, ctx.dummy_score)
    return SplitComparison(
        reference_splitter=reference_name,
        reference_score=ref,
        alternative_splitter=alternative_name,
        alternative_score=alt,
        dummy_score=ctx.dummy_score,
        absolute_drop=absolute,
        relative_drop=absolute / reference_lift if reference_lift > _EPSILON else None,
        n_usable_folds=min(
            len(reference.scores[metric.name]), len(alternative.scores[metric.name])
        ),
        sampling=sampling,
    )


def drop_severity(
    comparison: SplitComparison | None, warn: float, critical: float
) -> Severity | None:
    """Critical / warning by relative drop; None when below the warning level or unknown."""
    if comparison is None or comparison.relative_drop is None:
        return None
    if comparison.relative_drop >= critical:
        return Severity.CRITICAL
    if comparison.relative_drop >= warn:
        return Severity.WARNING
    return None
