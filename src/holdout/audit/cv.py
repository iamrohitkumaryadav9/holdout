"""Cross-validation loops for audits: several metrics per fit, tolerant of degenerate folds.

Audits compare splitters (e.g. forward-chaining folds) where an individual fold can be
unscorable, such as a validation fold with a single class. Such folds are counted and
skipped rather than aborting the check; the comparison reports how many folds it used.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone

from holdout.core import portable


@dataclass(frozen=True)
class FoldScores:
    scores: dict[str, list[float]]
    n_folds: int
    n_failed: int

    def mean(self, metric: str) -> float | None:
        values = self.scores.get(metric, [])
        return float(np.mean(values)) if values else None


def cross_validate(
    pipeline: Any,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series | None,
    splitter: Any,
    metrics: Sequence[str],
) -> FoldScores:
    """Fit a clone per fold on training rows only; score every metric on validation rows."""
    scores: dict[str, list[float]] = {m: [] for m in metrics}
    n_folds = n_failed = 0
    for train, valid in splitter.split(X, y, groups):
        n_folds += 1
        try:
            estimator = clone(pipeline).fit(X.iloc[train], y.iloc[train])
            fold = {m: portable.score(m, estimator, X.iloc[valid], y.iloc[valid]) for m in metrics}
        except ValueError:
            n_failed += 1
            continue
        for metric, value in fold.items():
            scores[metric].append(value)
    return FoldScores(scores=scores, n_folds=n_folds, n_failed=n_failed)
