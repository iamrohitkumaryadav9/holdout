"""Property tests for apply_split: partition, determinism, group and time guarantees."""

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from holdout.core.schemas import SplitStrategy
from holdout.data.split import SplitSpec, apply_split
from tests.tiers import examples

SEEDS = st.integers(0, 2**31 - 1)
TEST_SIZES = st.floats(0.1, 0.4)


@st.composite
def frames(draw: st.DrawFn) -> pd.DataFrame:
    """Random frames with a binary target, an entity column and a (shuffled) time column."""
    n_rows = draw(st.integers(40, 300))
    n_groups = draw(st.integers(12, 40))
    seed = draw(st.integers(0, 2**32 - 1))
    rng = np.random.default_rng(seed)
    start = np.datetime64("2023-01-01")
    times = start + rng.integers(0, 500, n_rows).astype("timedelta64[D]")
    return pd.DataFrame(
        {
            "x": rng.normal(size=n_rows),
            "entity": rng.integers(0, n_groups, n_rows),
            "when": pd.Series(times).dt.strftime("%Y-%m-%d"),
            "y": np.r_[[0, 0, 1, 1], rng.integers(0, 2, n_rows - 4)],
        }
    )


def _spec(strategy: SplitStrategy, seed: int, test_size: float) -> SplitSpec:
    return SplitSpec(
        strategy=strategy,
        seed=seed,
        test_size=test_size,
        group_col="entity" if strategy is SplitStrategy.GROUP else None,
        time_col="when" if strategy is SplitStrategy.TIME else None,
    )


@settings(max_examples=examples(60, 20), deadline=None)
@given(frame=frames(), strategy=st.sampled_from(list(SplitStrategy)), seed=SEEDS, size=TEST_SIZES)
def test_split_partitions_rows_and_is_deterministic(
    frame: pd.DataFrame, strategy: SplitStrategy, seed: int, size: float
) -> None:
    spec = _spec(strategy, seed, size)
    dev, holdout = apply_split(frame, spec, "y")
    assert set(dev.index).isdisjoint(holdout.index)
    assert sorted([*dev.index, *holdout.index]) == list(frame.index)

    dev_again, holdout_again = apply_split(frame, spec, "y")
    pd.testing.assert_frame_equal(dev, dev_again)
    pd.testing.assert_frame_equal(holdout, holdout_again)


@settings(max_examples=examples(60, 20), deadline=None)
@given(frame=frames(), seed=SEEDS, size=TEST_SIZES)
def test_group_split_never_shares_a_group(frame: pd.DataFrame, seed: int, size: float) -> None:
    dev, holdout = apply_split(frame, _spec(SplitStrategy.GROUP, seed, size), "y")
    assert set(dev["entity"]).isdisjoint(set(holdout["entity"]))


@settings(max_examples=examples(60, 20), deadline=None)
@given(frame=frames(), seed=SEEDS, size=TEST_SIZES)
def test_time_split_never_trains_on_the_future(frame: pd.DataFrame, seed: int, size: float) -> None:
    dev, holdout = apply_split(frame, _spec(SplitStrategy.TIME, seed, size), "y")
    assert pd.to_datetime(dev["when"]).max() <= pd.to_datetime(holdout["when"]).min()


@settings(max_examples=examples(30, 10), deadline=None)
@given(frame=frames(), seed=SEEDS)
def test_different_seeds_change_random_splits(frame: pd.DataFrame, seed: int) -> None:
    first = apply_split(frame, _spec(SplitStrategy.RANDOM, seed, 0.2), "y")[1].index
    second = apply_split(frame, _spec(SplitStrategy.RANDOM, seed + 1, 0.2), "y")[1].index
    # Not a guarantee for tiny frames, but for n >= 40 a collision is ~impossible.
    assert set(first) != set(second)
