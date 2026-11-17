"""The profile is computed before sealing, so it must be blind to feature-target relations.

Shuffling the target's values destroys every feature-target relation while keeping the
target's own distribution. If ``profile_schema`` is identical before and after the shuffle,
it cannot contain any feature-target statistic. A positive control proves the check would
catch a profiler that did compute one.
"""

from typing import Any

import numpy as np
import pandas as pd
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from holdout.data.examples import make_classification_example, make_timeseries_example
from holdout.data.profile import profile_schema
from tests.helpers import make_grouped_df
from tests.tiers import examples

DATASETS = {
    "churn": (make_classification_example(n_rows=300, seed=1), "churned"),
    "sales": (make_timeseries_example(n_days=40, seed=1), "units_sold"),
    "grouped": (make_grouped_df(), "label"),
    "churn_regression": (make_classification_example(n_rows=300, seed=2), "monthly_spend"),
}


def _shuffle_target(frame: pd.DataFrame, target: str, seed: int) -> pd.DataFrame:
    shuffled = frame.copy()
    rng = np.random.default_rng(seed)
    shuffled[target] = rng.permutation(frame[target].to_numpy())
    return shuffled


def _comparable(profile_dump: dict[str, Any]) -> dict[str, Any]:
    """Everything except target class frequencies (which a shuffle preserves anyway)."""
    return {
        k: v
        for k, v in profile_dump.items()
        if k not in ("target_class_frequencies", "target_class_counts")
    }


@settings(max_examples=examples(25, 8), deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(name=st.sampled_from(sorted(DATASETS)), seed=st.integers(0, 2**32 - 1))
def test_profile_is_invariant_to_shuffling_the_target(name: str, seed: int) -> None:
    frame, target = DATASETS[name]
    original = profile_schema(frame, target).model_dump()
    shuffled = profile_schema(_shuffle_target(frame, target, seed), target).model_dump()
    assert _comparable(shuffled) == _comparable(original)
    # Class frequencies are a property of the target alone, so they match too.
    assert shuffled["target_class_frequencies"] == original["target_class_frequencies"]


def test_positive_control_a_leaky_statistic_is_detected() -> None:
    """A correlation-with-target 'profile' changes under the shuffle, so the check has teeth."""
    frame, target = DATASETS["churn"]

    def leaky_profile(data: pd.DataFrame) -> dict[str, float]:
        return {
            c: round(float(data[c].corr(data[target])), 12)
            for c in ("tenure_months", "support_tickets")
        }

    assert leaky_profile(frame) != leaky_profile(_shuffle_target(frame, target, seed=0))
