"""A7 metric fit: is the chosen metric able to tell a good model from a useless one?

Classification metric problems are found by Week 1's ``validate_plan`` (single source of
truth); A7 wraps those issues into audit flags. Accuracy on an imbalanced target is a
pre-seal warning and an A7 *critical*: the run shows, as secondary CV metrics,
balanced accuracy and (binary) PR AUC for the plan and the dummy, so the user sees how
misleading accuracy is. Regression adds one check: R2 on a heavy-tailed target is
dominated by a few extreme rows, so MAE is suggested.
"""

import numpy as np

from holdout.audit import cv as audit_cv
from holdout.audit.checks import CheckOutcome
from holdout.audit.context import AuditContext
from holdout.audit.sampling import sample_rows, sampling_info
from holdout.audit.types import (
    AuditFlag,
    Category,
    ChangeMetric,
    CheckId,
    MetricEvidence,
    SamplingInfo,
    Severity,
)
from holdout.core import portable
from holdout.core.schemas import MetricName, Task
from holdout.plan.compiler import compile_plan, cv_from_split
from holdout.plan.defaults import baseline_plan
from holdout.plan.schema import validate_plan

#: validate_plan issue codes that concern the metric.
METRIC_ISSUE_CODES = frozenset(
    {"accuracy_on_imbalanced_target", "metric_task_mismatch", "metric_needs_binary"}
)


def run(ctx: AuditContext) -> CheckOutcome:
    flags: list[AuditFlag] = []
    sampling: SamplingInfo | None = None
    issues = [i for i in validate_plan(ctx.plan, ctx.profile) if i.code in METRIC_ISSUE_CODES]
    for issue in issues:
        secondary: dict[str, float] = {}
        secondary_dummy: dict[str, float] = {}
        if issue.code == "accuracy_on_imbalanced_target":
            secondary, secondary_dummy, sampling = _secondary_scores(ctx)
        shown = ", ".join(
            f"{name} {value:.3f} (dummy {secondary_dummy[name]:.3f})"
            for name, value in secondary.items()
        )
        message = issue.message[0].upper() + issue.message[1:] + "."
        if shown:
            message = (
                f"Accuracy {ctx.full_score:.3f} vs dummy {ctx.dummy_score:.3f} looks good only "
                f"because one class dominates; {shown}."
            )
        flags.append(
            AuditFlag(
                check=CheckId.A7,
                category=Category.METRIC_MISMATCH,
                severity=Severity.CRITICAL,
                columns=[ctx.target],
                evidence=MetricEvidence(
                    metric=ctx.plan.metric,
                    plan_issue_code=issue.code,
                    minority_class_fraction=ctx.profile.minority_class_fraction,
                    secondary_scores=secondary,
                    secondary_dummy_scores=secondary_dummy,
                    tail_ratio=None,
                ),
                message=message,
                fix=ChangeMetric(metric=MetricName.BALANCED_ACCURACY),
            )
        )
    if ctx.task is Task.REGRESSION and ctx.plan.metric is MetricName.R2:
        ratio = tail_ratio(ctx.dev[ctx.target].to_numpy(dtype="float64"))
        if ratio is not None and ratio >= ctx.thresholds.a07_metric.heavy_tail_ratio:
            flags.append(
                AuditFlag(
                    check=CheckId.A7,
                    category=Category.METRIC_MISMATCH,
                    severity=Severity.WARNING,
                    columns=[ctx.target],
                    evidence=MetricEvidence(
                        metric=ctx.plan.metric,
                        plan_issue_code=None,
                        minority_class_fraction=None,
                        secondary_scores={},
                        secondary_dummy_scores={},
                        tail_ratio=ratio,
                    ),
                    message=(
                        f"{ctx.target!r} is heavy-tailed (tail ratio {ratio:.1f}); R2 is "
                        "dominated by a few extreme rows. MAE reflects typical errors better."
                    ),
                    fix=ChangeMetric(metric=MetricName.MAE),
                )
            )
    return CheckOutcome(flags=flags, sampling=sampling, detail=f"metric {ctx.plan.metric.value}")


def tail_ratio(values: np.ndarray) -> float | None:
    """(p99 - p50) / (p75 - p25); None when the IQR is zero."""
    p25, p50, p75, p99 = np.percentile(values, [25, 50, 75, 99])
    iqr = p75 - p25
    return float((p99 - p50) / iqr) if iqr > 0 else None


def _secondary_scores(
    ctx: AuditContext,
) -> tuple[dict[str, float], dict[str, float], SamplingInfo]:
    """Plan and dummy CV means for balanced accuracy and (binary) PR AUC."""
    sample, row_method = sample_rows(ctx, "a07")
    X, y, groups = portable.cv_inputs(sample, ctx.target, ctx.spec.to_config())
    metrics = [MetricName.BALANCED_ACCURACY.value]
    if (ctx.profile.n_classes or 0) == 2:
        metrics.append(MetricName.PR_AUC.value)
    splitter = cv_from_split(ctx.spec)
    plan_scores = audit_cv.cross_validate(
        compile_plan(ctx.plan, ctx.profile, run_seed=ctx.spec.seed), X, y, groups, splitter, metrics
    )
    dummy_scores = audit_cv.cross_validate(
        compile_plan(baseline_plan(ctx.plan), ctx.profile, run_seed=ctx.spec.seed),
        X,
        y,
        groups,
        splitter,
        metrics,
    )
    secondary = {m: v for m in metrics if (v := plan_scores.mean(m)) is not None}
    dummy = {m: v for m in metrics if (v := dummy_scores.mean(m)) is not None}
    return secondary, dummy, sampling_info(ctx, sample, row_method, len(ctx.features), "all")
