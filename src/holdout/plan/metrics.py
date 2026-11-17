"""The metric registry: one place that knows each metric's task, direction and scorer.

Scores are reported in their natural sign (an RMSE of 3.2 is ``3.2``, not scikit-learn's
``-3.2``), and comparisons go through :meth:`Metric.improvement`, so no caller ever has to
remember which way is better.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

import pandas as pd

from holdout.core import portable
from holdout.core.schemas import Direction, MetricName, Task


@dataclass(frozen=True)
class Metric:
    """A metric and everything needed to use it correctly.

    Instances are scikit-learn compatible scorers: ``metric(estimator, X, y)``.
    """

    name: MetricName
    task: Task
    direction: Direction
    binary_only: bool
    threshold_free: bool
    description: str

    def score(self, estimator: Any, X: pd.DataFrame, y: pd.Series) -> float:
        """Score a fitted estimator (the implementation lives in the portable kernel)."""
        return portable.score(self.name.value, estimator, X, y)

    def __call__(self, estimator: Any, X: pd.DataFrame, y: pd.Series) -> float:
        return self.score(estimator, X, y)

    def improvement(self, candidate: float, reference: float) -> float:
        """How much better ``candidate`` is than ``reference`` (positive means better)."""
        if self.direction is Direction.HIGHER:
            return candidate - reference
        return reference - candidate


_C = Task.CLASSIFICATION
_R = Task.REGRESSION
_HI = Direction.HIGHER
_LO = Direction.LOWER

METRICS: Final[Mapping[MetricName, Metric]] = MappingProxyType(
    {
        m.name: m
        for m in (
            Metric(MetricName.ROC_AUC, _C, _HI, False, True, "ROC AUC; one-vs-rest if multiclass"),
            Metric(MetricName.PR_AUC, _C, _HI, True, True, "average precision of classes_[1]"),
            Metric(MetricName.F1, _C, _HI, False, False, "F1 of classes_[1]; macro if multiclass"),
            Metric(MetricName.BALANCED_ACCURACY, _C, _HI, False, False, "mean per-class recall"),
            Metric(MetricName.ACCURACY, _C, _HI, False, False, "share of correct predictions"),
            Metric(MetricName.RMSE, _R, _LO, False, False, "root mean squared error"),
            Metric(MetricName.MAE, _R, _LO, False, False, "mean absolute error"),
            Metric(MetricName.R2, _R, _HI, False, False, "coefficient of determination"),
        )
    }
)


def get_metric(name: MetricName | str) -> Metric:
    """Look up a metric by name.

    Raises:
        KeyError: naming the valid metrics.
    """
    try:
        return METRICS[MetricName(name)]
    except ValueError as exc:
        raise KeyError(
            f"unknown metric {name!r}; choose from {[m.value for m in METRICS]}"
        ) from exc
