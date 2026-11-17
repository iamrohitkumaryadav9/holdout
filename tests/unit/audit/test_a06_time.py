"""A6: hidden time order with drift is critical and invalidates; drift-free twins and
files sorted by an ordinary feature never produce a time-leak critical."""

import dataclasses

import pandas as pd
import pytest

from bench.traps import build_case, clean_twin
from bench.traps.synthetic import easy_clean, regression
from bench.traps.types import clean_case
from holdout.audit.checks import a06_time as a6
from holdout.audit.types import Severity
from holdout.core.schemas import SplitStrategy, Task
from holdout.data.split import SplitSpec
from tests.audit_harness import context_for, trap_context, twin_context
from tests.tiers import SEEDS


@pytest.mark.parametrize("seed", SEEDS)
def test_hidden_row_order_with_drift_is_critical(seed: int) -> None:
    outcome = a6.run(trap_context("time_order", seed))
    flag = outcome.flags[0]
    assert flag.severity is Severity.CRITICAL
    assert flag.invalidates_holdout
    evidence = flag.evidence
    assert evidence.order_source == "row_order"  # type: ignore[union-attr]
    assert evidence.z_score >= 4  # type: ignore[union-attr,operator]
    assert flag.fix.kind == "change_split"
    assert flag.fix.new_spec is None  # type: ignore[union-attr]
    assert "--split time:row_number" in flag.fix.instruction  # type: ignore[union-attr]


@pytest.mark.parametrize("seed", SEEDS[:2])
def test_renamed_date_column_is_found_by_parsing_not_by_name(seed: int) -> None:
    ctx = context_for(build_case("time_order", seed, params={"mode": "rename"}), seed)
    flags = a6.run(ctx).flags
    assert len(flags) == 1  # the row order is the same order: no duplicate flag
    assert flags[0].columns == ["period"]
    assert flags[0].severity is Severity.CRITICAL
    spec = flags[0].fix.new_spec  # type: ignore[union-attr]
    assert spec is not None
    assert (spec.strategy, spec.time_col) == (SplitStrategy.TIME, "period")


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("trap", ["time_order", "direct_leak", "group_leakage"])
def test_quiet_on_clean_twins(trap: str, seed: int) -> None:
    assert [
        f for f in a6.run(twin_context(trap, seed)).flags if f.severity is not Severity.INFO
    ] == []


def _sorted_case(seed: int, column: str):  # type: ignore[no-untyped-def]
    frame, target, task = easy_clean(n_rows=1200, seed=seed)
    frame = frame.sort_values(column, kind="stable").reset_index(drop=True)
    return clean_case(frame, target, task)


@pytest.mark.parametrize("seed", SEEDS)
def test_clean_data_sorted_by_a_category_is_never_a_time_critical(seed: int) -> None:
    flags = a6.run(context_for(_sorted_case(seed, "odor"), seed)).flags
    assert all(f.severity is not Severity.CRITICAL for f in flags)
    assert not any(f.invalidates_holdout for f in flags)
    assert flags[0].severity is Severity.WARNING
    assert flags[0].evidence.sorted_by == "odor"  # type: ignore[union-attr]
    assert "appears sorted by 'odor'" in flags[0].message


@pytest.mark.parametrize("seed", SEEDS[:2])
def test_clean_data_sorted_by_a_continuous_feature_is_a_sort_warning(seed: int) -> None:
    frame, target, task = regression(seed=seed)  # score_a dominates the target
    frame = frame.sort_values("score_a", kind="stable").reset_index(drop=True)
    flags = a6.run(context_for(clean_case(frame, target, task), seed)).flags
    assert all(f.severity is not Severity.CRITICAL for f in flags)
    assert [f.evidence.sorted_by for f in flags] == ["score_a"]  # type: ignore[union-attr]


@pytest.mark.parametrize("seed", SEEDS[:2])
def test_weak_sorting_feature_produces_no_flag_at_all(seed: int) -> None:
    case = clean_twin(build_case("noisy_proxy", seed))
    frame = case.df.sort_values("score_a", kind="stable").reset_index(drop=True)
    assert (
        a6.run(context_for(clean_case(frame, case.target, Task.CLASSIFICATION), seed)).flags == []
    )


def test_time_split_runs_skip_a6() -> None:
    case = build_case("time_order", 0, params={"mode": "rename"})
    spec = SplitSpec(strategy=SplitStrategy.TIME, time_col="period", seed=0)
    assert a6.run(context_for(case, 0, spec=spec)).skipped


def test_row_cap_samples_systematically() -> None:
    ctx = trap_context("time_order", 0)
    tight = ctx.thresholds.model_copy(
        update={"sampling": ctx.thresholds.sampling.model_copy(update={"max_rows": 500})}
    )
    flag = a6.run(dataclasses.replace(ctx, thresholds=tight)).flags[0]
    sampling = flag.evidence.comparison.sampling  # type: ignore[union-attr]
    assert (sampling.n_rows_used, sampling.row_method) == (500, "systematic")


def test_lag_statistics() -> None:
    assert a6._lag_statistic(
        pd.Series([0, 0, 1, 1]).to_numpy(dtype="float64"), Task.CLASSIFICATION
    ) == pytest.approx(2 / 3)
    assert a6._lag_statistic(
        pd.Series(range(10)).to_numpy(dtype="float64"), Task.REGRESSION
    ) == pytest.approx(1.0)
    assert a6._lag_statistic(pd.Series([1.0, 1.0, 1.0]).to_numpy(), Task.REGRESSION) == 0.0
