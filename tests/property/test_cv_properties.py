"""CV folds never contain holdout rows; the same seed reproduces split and scores exactly."""

from pathlib import Path

import numpy as np
import pandas as pd
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from holdout.core import portable
from holdout.core.schemas import SplitStrategy
from holdout.core.state import RunState
from holdout.data.profile import profile_schema
from holdout.data.seal import SealedHoldout
from holdout.data.split import SplitSpec, apply_split
from holdout.plan.compiler import cv_from_split, evaluate_cv
from holdout.plan.schema import ModelPlan
from tests.helpers import plan_dict
from tests.tiers import examples


def _frame(seed: int, n_rows: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n_rows)
    days = rng.integers(0, 400, n_rows).astype("timedelta64[D]")
    return pd.DataFrame(
        {
            "x": x,
            "color": rng.choice(["r", "g", "b"], n_rows),
            "entity": rng.integers(0, 25, n_rows).astype(str),
            "when": pd.Series(np.datetime64("2024-01-01") + days).dt.strftime("%Y-%m-%d"),
            "y": np.r_[[0] * 6, [1] * 6, (x[12:] + rng.normal(size=n_rows - 12) > 0).astype(int)],
        }
    )


def _spec(strategy: SplitStrategy, seed: int) -> SplitSpec:
    return SplitSpec(
        strategy=strategy,
        seed=seed,
        group_col="entity" if strategy is SplitStrategy.GROUP else None,
        time_col="when" if strategy is SplitStrategy.TIME else None,
    )


PLAN = ModelPlan.model_validate(
    plan_dict(
        model={"name": "hist_gb", "max_iter": 15},
        drop_columns=[
            {"column": "entity", "reason": "entity id"},
            {"column": "when", "reason": "raw timestamp"},
        ],
    )
)


@settings(
    max_examples=examples(40, 10),
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    data_seed=st.integers(0, 2**32 - 1),
    run_seed=st.integers(0, 2**31 - 1),
    n_rows=st.integers(120, 400),
    strategy=st.sampled_from(list(SplitStrategy)),
)
def test_no_holdout_row_ever_appears_in_a_cv_fold(
    tmp_path: Path, data_seed: int, run_seed: int, n_rows: int, strategy: SplitStrategy
) -> None:
    frame = _frame(data_seed, n_rows)
    spec = _spec(strategy, run_seed)
    state = RunState.create(tmp_path / "runs", run_seed)
    _, dev = SealedHoldout.seal(frame, "y", spec, state, data_sha256="sha")
    holdout_ids = set(pd.read_parquet(state.holdout_path).index)

    X, y, groups = portable.cv_inputs(dev, "y", spec.to_config())
    seen: set[int] = set()
    for train, valid in cv_from_split(spec).split(X, y, groups):
        train_ids = set(X.index[train])
        valid_ids = set(X.index[valid])
        assert train_ids.isdisjoint(holdout_ids)
        assert valid_ids.isdisjoint(holdout_ids)
        assert train_ids.isdisjoint(valid_ids)
        if strategy is SplitStrategy.GROUP:
            assert set(groups.loc[list(train_ids)]).isdisjoint(set(groups.loc[list(valid_ids)]))
        if strategy is SplitStrategy.TIME:
            times = pd.to_datetime(X["when"])
            assert times.loc[list(train_ids)].max() <= times.loc[list(valid_ids)].min()
        seen |= train_ids | valid_ids
    assert seen <= set(dev.index)


@settings(max_examples=examples(8, 3), deadline=None)
@given(
    data_seed=st.integers(0, 2**32 - 1),
    run_seed=st.integers(0, 2**31 - 1),
    strategy=st.sampled_from(list(SplitStrategy)),
)
def test_same_seed_gives_identical_split_and_cv_scores(
    data_seed: int, run_seed: int, strategy: SplitStrategy
) -> None:
    frame = _frame(data_seed, 200)
    profile = profile_schema(frame, "y")
    spec = _spec(strategy, run_seed)
    results = []
    for _ in range(2):
        dev, holdout = apply_split(frame, spec, "y")
        results.append(
            (dev.index.tolist(), holdout.index.tolist(), evaluate_cv(PLAN, dev, "y", spec, profile))
        )
    (dev_a, hold_a, cv_a), (dev_b, hold_b, cv_b) = results
    assert dev_a == dev_b
    assert hold_a == hold_b
    assert cv_a.fold_scores == cv_b.fold_scores  # exact, not approximate
