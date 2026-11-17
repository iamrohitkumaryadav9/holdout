"""Test helpers shared across unit, property and integration tests."""

from typing import Any

import numpy as np
import pandas as pd

from holdout.core.schemas import MetricName

RATIONALE: dict[str, str] = {
    "metric": "chosen for the test",
    "drop_columns": "chosen for the test",
    "preprocessing": "chosen for the test",
    "feature_selection": "chosen for the test",
    "model": "chosen for the test",
    "class_weight": "chosen for the test",
}


def plan_dict(**overrides: Any) -> dict[str, Any]:
    """A minimal valid plan as a dict; override any field to test one change."""
    base: dict[str, Any] = {
        "task": "classification",
        "metric": "roc_auc",
        "model": {"name": "hist_gb"},
        "rationale": RATIONALE,
    }
    base.update(overrides)
    return base


def make_grouped_df(
    n_customers: int = 40, rows_per_customer: int = 6, seed: int = 0
) -> pd.DataFrame:
    """Repeated measurements per customer: the case a random split gets wrong."""
    rng = np.random.default_rng(seed)
    customer = np.repeat([f"c{i:03d}" for i in range(n_customers)], rows_per_customer)
    base = np.repeat(rng.normal(size=n_customers), rows_per_customer)
    x = base + rng.normal(scale=0.3, size=len(customer))
    return pd.DataFrame(
        {
            "customer_id": customer,
            "x": x,
            "segment": rng.choice(["a", "b", "c"], len(customer)),
            "label": (base + rng.normal(scale=0.5, size=len(customer)) > 0).astype(int),
        }
    )


class AccuracyScorer:
    """Minimal Scorer for seal tests, independent of the plan layer."""

    @property
    def name(self) -> MetricName:
        return MetricName.ACCURACY

    def score(self, estimator: Any, X: pd.DataFrame, y: pd.Series) -> float:
        return float((estimator.predict(X) == y.to_numpy()).mean())
