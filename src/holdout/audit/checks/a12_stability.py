"""A12 unstable validation: could the fold-to-fold spread change the conclusion?

The conclusion a run claims is "the plan beats the dummy baseline". A12 tests it per fold:
the plan and the dummy are cross-validated on the same seeded folds, so their per-fold
differences are paired. The confidence interval of the mean difference uses the
Nadeau-Bengio corrected resampled t-test (Nadeau & Bengio, "Inference for the
Generalization Error", Machine Learning 52, 2003): CV folds share training rows, so the
naive variance s^2/k is too small; the corrected variance is

    (1/k + n_test/n_train) * s^2,  with k - 1 degrees of freedom,

where n_test/n_train is averaged over the run's actual folds.

Warning only when the spread could change the conclusion: some fold scores at or below
the dummy, or the interval crosses zero. Otherwise info, with the spread recorded.
"""

import numpy as np
from scipy import stats

from holdout.audit.checks import CheckOutcome
from holdout.audit.context import AuditContext
from holdout.audit.scoring import lift
from holdout.audit.types import AuditFlag, Category, CheckId, NoAction, Severity, StabilityEvidence
from holdout.core import portable
from holdout.plan.compiler import cv_from_split

_EPSILON = 1e-12


def nadeau_bengio_interval(
    differences: np.ndarray, test_train_ratio: float, confidence: float
) -> tuple[float, float, float]:
    """(mean, low, high) of the corrected resampled t-interval for paired fold differences."""
    k = len(differences)
    mean = float(np.mean(differences))
    if k < 2:
        return mean, float("-inf"), float("inf")
    variance = (1.0 / k + test_train_ratio) * float(np.var(differences, ddof=1))
    half = float(stats.t.ppf(0.5 + confidence / 2.0, df=k - 1)) * float(np.sqrt(variance))
    return mean, mean - half, mean + half


def fold_test_train_ratio(ctx: AuditContext) -> float:
    """Mean n_test / n_train over the run's folds (exact, from the run's own splitter)."""
    X, y, groups = portable.cv_inputs(ctx.dev, ctx.target, ctx.spec.to_config())
    ratios = [
        len(valid) / len(train) for train, valid in cv_from_split(ctx.spec).split(X, y, groups)
    ]
    return float(np.mean(ratios))


def run(ctx: AuditContext) -> CheckOutcome:
    t = ctx.thresholds.a12_stability
    metric = ctx.metric
    scores = np.asarray(ctx.cv_result.fold_scores, dtype="float64")
    dummy = np.asarray(ctx.baseline.fold_scores, dtype="float64")
    if len(dummy) != len(scores):  # not paired: fall back to the dummy's mean
        dummy = np.full_like(scores, ctx.dummy_score)
    mean, std = float(scores.mean()), float(scores.std())
    mean_lift = lift(metric, mean, ctx.dummy_score)
    if mean_lift <= _EPSILON:
        return CheckOutcome(detail="plan does not beat the dummy; stability not assessed")
    lifts = np.array([lift(metric, float(s), float(d)) for s, d in zip(scores, dummy, strict=True)])
    ratio = fold_test_train_ratio(ctx)
    _, low, high = nadeau_bengio_interval(lifts, ratio, t.confidence)
    at_or_below = int(np.sum(lifts <= 0.0))
    reasons: list[str] = []
    if at_or_below:
        reasons.append(f"{at_or_below} fold(s) score at or below the dummy baseline")
    if low <= 0.0:
        reasons.append(
            f"the {t.confidence:.0%} interval of the lift over the dummy ({low:+.3f} to "
            f"{high:+.3f}, Nadeau-Bengio corrected) includes zero"
        )
    severity = Severity.WARNING if reasons else Severity.INFO
    if reasons:
        message = "Validation is unstable: " + "; ".join(reasons) + "."
    else:
        message = (
            f"Fold scores vary (std {std:.3f}), but every fold beats the dummy and the "
            f"{t.confidence:.0%} interval of the lift ({low:+.3f} to {high:+.3f}) excludes zero."
        )
    flag = AuditFlag(
        check=CheckId.A12,
        category=Category.UNSTABLE_VALIDATION,
        severity=severity,
        columns=[],
        evidence=StabilityEvidence(
            fold_scores=[float(s) for s in scores],
            dummy_score=ctx.dummy_score,
            dummy_fold_scores=[float(d) for d in dummy],
            fold_lifts=[float(v) for v in lifts],
            coefficient_of_variation=std / abs(mean) if abs(mean) > _EPSILON else None,
            lift_coefficient_of_variation=std / mean_lift,
            test_train_ratio=ratio,
            lift_interval_low=low,
            lift_interval_high=high,
            confidence=t.confidence,
        ),
        message=message,
        fix=NoAction(
            reason="get more data or use repeated CV before trusting the mean"
            if reasons
            else "spread recorded; it does not change the conclusion"
        ),
    )
    return CheckOutcome(flags=[flag])
