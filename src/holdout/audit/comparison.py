"""Compare the plan's CV score under a reference splitter and a stricter alternative.

Used by A5 (GroupKFold) and A6 (forward-chaining folds). The drop is expressed relative
to the plan's lift over the dummy under the reference splitter, so "0.4" means the
stricter split removes 40% of what the plan claims to have learned, on any metric.

Noise calibration: on small data with a weak model the relative drop is noisy (measured
on clean classification twins: sd ~0.16, max ~0.34 when the true drop is zero). So the
reference is run a second time under a different shuffle; the difference between the two
estimates the noise, and a drop only counts when it clears both the threshold and a
multiple of that noise.
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
    reference_repeat: audit_cv.FoldScores,
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
    repeat = reference_repeat.mean(metric.name)
    absolute = metric.improvement(ref, alt)
    reference_lift = lift(metric, ref, ctx.dummy_score)
    usable = reference_lift > _EPSILON
    return SplitComparison(
        reference_splitter=reference_name,
        reference_score=ref,
        alternative_splitter=alternative_name,
        alternative_score=alt,
        dummy_score=ctx.dummy_score,
        absolute_drop=absolute,
        relative_drop=absolute / reference_lift if usable else None,
        reference_repeat_score=repeat,
        relative_noise=abs(ref - repeat) / reference_lift
        if usable and repeat is not None
        else None,
        n_usable_folds=min(
            len(reference.scores[metric.name]), len(alternative.scores[metric.name])
        ),
        sampling=sampling,
    )


def drop_severity(
    comparison: SplitComparison | None, warn: float, critical: float, min_drop_to_noise: float
) -> Severity | None:
    """Critical / warning by relative drop, provided it clears ``min_drop_to_noise`` times
    the measured noise; None when below the warning level, within noise, or unknown."""
    if comparison is None or comparison.relative_drop is None:
        return None
    drop = comparison.relative_drop
    noise = comparison.relative_noise if comparison.relative_noise is not None else 0.0
    if drop < min_drop_to_noise * noise:
        return None
    if drop >= critical:
        return Severity.CRITICAL
    if drop >= warn:
        return Severity.WARNING
    return None
