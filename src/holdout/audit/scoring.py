"""Metric-independent score arithmetic shared by the checks.

``lift`` is direction-aware improvement over the dummy baseline; ``normalized`` divides by
the distance from the dummy to a perfect score, so 0.95 means "95% of the way to perfect"
for AUC, R2 and RMSE alike.
"""

from holdout.core.schemas import MetricName
from holdout.plan.metrics import Metric

_PERFECT: dict[MetricName, float] = {
    MetricName.ROC_AUC: 1.0,
    MetricName.PR_AUC: 1.0,
    MetricName.F1: 1.0,
    MetricName.BALANCED_ACCURACY: 1.0,
    MetricName.ACCURACY: 1.0,
    MetricName.R2: 1.0,
    MetricName.RMSE: 0.0,
    MetricName.MAE: 0.0,
}

_EPSILON = 1e-12


def perfect_score(metric: Metric) -> float:
    return _PERFECT[metric.name]


def lift(metric: Metric, score: float, dummy: float) -> float:
    return metric.improvement(score, dummy)


def normalized(metric: Metric, score: float, dummy: float) -> float:
    """Share of the way from ``dummy`` to a perfect score (0 when dummy is already perfect)."""
    span = abs(perfect_score(metric) - dummy)
    return lift(metric, score, dummy) / span if span > _EPSILON else 0.0


def relative_lift(metric: Metric, single: float, dummy: float, full: float) -> float | None:
    """``lift(single) / lift(full)``; None when the full plan does not beat the dummy."""
    full_lift = lift(metric, full, dummy)
    if full_lift <= _EPSILON:
        return None
    return lift(metric, single, dummy) / full_lift
