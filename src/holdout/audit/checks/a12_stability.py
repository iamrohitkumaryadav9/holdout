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

from collections.abc import Sequence

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
    below = mean_lift <= _EPSILON
    v11 = ctx.thresholds_v11
    if below and (v11 is None or not v11.a12_stability.warn_below_baseline):
        return CheckOutcome(detail="plan does not beat the dummy; stability not assessed")
    lifts = np.array([lift(metric, float(s), float(d)) for s, d in zip(scores, dummy, strict=True)])
    ratio = fold_test_train_ratio(ctx)
    _, low, high = nadeau_bengio_interval(lifts, ratio, t.confidence)
    if below:
        return CheckOutcome(flags=[_below_baseline(ctx, scores, dummy, lifts, ratio, low, high)])
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


#: The plain statement shown in the report summary and the app (v1.1).
BELOW_BASELINE: str = (
    "The model does not beat a baseline that ignores the features; do not rely on it."
)
NOT_RELIABLY_ABOVE_BASELINE: str = (
    "The model does not reliably beat a baseline that ignores the features (the interval "
    "of its lift includes zero); do not rely on it."
)


def baseline_statement(flags: Sequence[AuditFlag]) -> str | None:
    """The sentence a report must lead with when A12 warns that the plan does not beat the
    dummy (mean lift at or below zero), or not reliably (the lift interval includes zero)."""
    for flag in flags:
        evidence = flag.evidence
        if (
            flag.check is CheckId.A12
            and flag.severity is Severity.WARNING
            and isinstance(evidence, StabilityEvidence)
            and evidence.lift_interval_low <= 0.0
        ):
            mean_lift = float(np.mean(evidence.fold_lifts)) if evidence.fold_lifts else 0.0
            return BELOW_BASELINE if mean_lift <= _EPSILON else NOT_RELIABLY_ABOVE_BASELINE
    return None


def _below_baseline(
    ctx: AuditContext,
    scores: np.ndarray,
    dummy: np.ndarray,
    lifts: np.ndarray,
    ratio: float,
    low: float,
    high: float,
) -> AuditFlag:
    """v1.1: the selected plan does not beat the dummy (mean lift <= 0, so the interval
    cannot exclude zero on the good side). A warning, not a block: see docs/design.md."""
    t = ctx.thresholds.a12_stability
    std = float(scores.std())
    mean = float(scores.mean())
    return AuditFlag(
        check=CheckId.A12,
        category=Category.UNSTABLE_VALIDATION,
        severity=Severity.WARNING,
        columns=[],
        evidence=StabilityEvidence(
            fold_scores=[float(s) for s in scores],
            dummy_score=ctx.dummy_score,
            dummy_fold_scores=[float(d) for d in dummy],
            fold_lifts=[float(v) for v in lifts],
            coefficient_of_variation=std / abs(mean) if abs(mean) > _EPSILON else None,
            lift_coefficient_of_variation=None,
            test_train_ratio=ratio,
            lift_interval_low=low,
            lift_interval_high=high,
            confidence=t.confidence,
        ),
        message=(
            f"{BELOW_BASELINE} Cross-validated {ctx.metric.name.value} {mean:.3f} vs "
            f"{ctx.dummy_score:.3f} for the dummy; the {t.confidence:.0%} interval of the lift "
            f"({low:+.3f} to {high:+.3f}) does not exclude zero."
        ),
        fix=NoAction(
            reason="warned, not blocked: the holdout gives an honest estimate of a weak model"
        ),
    )
