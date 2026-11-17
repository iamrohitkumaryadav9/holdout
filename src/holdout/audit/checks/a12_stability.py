"""A12 unstable validation: fold scores too spread out to trust the mean.

Two signals, both warnings: the fold scores spread by a large share of the lift being
claimed (std / |mean - dummy|; the plain coefficient of variation std / |mean| is also
recorded), or the worst fold keeps little of the mean lift over the dummy. The spread is
measured against the lift rather than the raw score because a raw-score CV is
scale-dependent (an AUC near 0.5 or an R2 near 0 makes it meaningless).
"""

import numpy as np

from holdout.audit.checks import CheckOutcome
from holdout.audit.context import AuditContext
from holdout.audit.scoring import lift
from holdout.audit.types import AuditFlag, Category, CheckId, NoAction, Severity, StabilityEvidence

_EPSILON = 1e-12


def run(ctx: AuditContext) -> CheckOutcome:
    t = ctx.thresholds.a12_stability
    metric = ctx.metric
    scores = np.asarray(ctx.cv_result.fold_scores, dtype="float64")
    mean, std = float(scores.mean()), float(scores.std())
    mean_lift = lift(metric, mean, ctx.dummy_score)
    plain_cv = std / abs(mean) if abs(mean) > _EPSILON else None
    if mean_lift <= _EPSILON:
        return CheckOutcome(detail="plan does not beat the dummy; stability not assessed")
    lift_cv = std / mean_lift
    worst = min(scores, key=lambda s: lift(metric, float(s), ctx.dummy_score))
    worst_share = lift(metric, float(worst), ctx.dummy_score) / mean_lift
    reasons: list[str] = []
    if lift_cv >= t.max_lift_coefficient_of_variation:
        reasons.append(f"fold scores spread by {lift_cv:.0%} of the claimed lift")
    if worst_share <= t.min_worst_fold_lift_share:
        reasons.append(f"the worst fold ({worst:.3f}) keeps only {worst_share:.0%} of the lift")
    if not reasons:
        return CheckOutcome(detail=f"lift CV {lift_cv:.2f}, worst fold share {worst_share:.2f}")
    flag = AuditFlag(
        check=CheckId.A12,
        category=Category.UNSTABLE_VALIDATION,
        severity=Severity.WARNING,
        columns=[],
        evidence=StabilityEvidence(
            fold_scores=[float(s) for s in scores],
            dummy_score=ctx.dummy_score,
            coefficient_of_variation=plain_cv,
            lift_coefficient_of_variation=lift_cv,
            worst_fold_lift_share=worst_share,
        ),
        message="Validation is unstable: " + "; ".join(reasons) + ".",
        fix=NoAction(reason="get more data or use repeated CV before trusting the mean"),
    )
    return CheckOutcome(flags=[flag])
