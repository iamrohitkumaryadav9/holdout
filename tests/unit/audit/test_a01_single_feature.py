"""A1: fires on planted leaks (warning alone, critical when corroborated); quiet on twins."""

import dataclasses

import pytest

from holdout.audit.checks import a01_single_feature as a1
from holdout.audit.types import CheckId, Severity
from holdout.plan.schema import DropColumn
from tests.audit_harness import trap_context, twin_context
from tests.tiers import SEEDS


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize(
    ("trap", "column"), [("direct_leak", "refund_issued"), ("noisy_proxy", "signal_strength")]
)
def test_fires_as_warning_alone_and_critical_when_corroborated(
    trap: str, column: str, seed: int
) -> None:
    ctx = trap_context(trap, seed)
    result = a1.measure(ctx)
    alone = {f.columns[0]: f for f in a1.flags(ctx, result, {})}
    assert alone[column].severity is Severity.WARNING
    assert alone[column].evidence.kind == "single_feature"
    corroborated = {f.columns[0]: f for f in a1.flags(ctx, result, {column: [CheckId.A2]})}
    assert corroborated[column].severity is Severity.CRITICAL
    assert corroborated[column].evidence.corroborated_by == [CheckId.A2]  # type: ignore[union-attr]
    assert corroborated[column].fix.kind == "drop_columns"
    assert not corroborated[column].invalidates_holdout


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("trap", ["direct_leak", "noisy_proxy", "id_with_signal"])
def test_quiet_on_clean_twins(trap: str, seed: int) -> None:
    ctx = twin_context(trap, seed)
    result = a1.measure(ctx)
    # Even with every column corroborated, nothing on a clean twin may rise above info.
    everything = {c: [CheckId.A2] for c in result.scores}
    assert [f for f in a1.flags(ctx, result, everything) if f.severity is not Severity.INFO] == []


def test_a_leak_the_plan_already_drops_is_info() -> None:
    ctx = trap_context("direct_leak", 0)
    plan = ctx.plan.model_copy(
        update={
            "drop_columns": [
                *ctx.plan.drop_columns,
                DropColumn(column="refund_issued", reason="known leak"),
            ]
        }
    )
    dropped = dataclasses.replace(ctx, plan=plan)
    flag = next(
        f for f in a1.flags(dropped, a1.measure(dropped), {}) if f.columns == ["refund_issued"]
    )
    assert flag.severity is Severity.INFO
    assert "already drops" in flag.message


def test_evidence_records_sampling_and_splitter() -> None:
    ctx = trap_context("direct_leak", 0)
    result = a1.measure(ctx)
    assert result.sampling.n_rows_used == len(ctx.dev)
    assert result.sampling.row_method == "none"
    assert result.sampling.feature_method == "all"
    assert result.splitter == "StratifiedKFold"
    assert set(result.scores) == set(ctx.features)


def test_sampling_caps_rows_and_features() -> None:
    ctx = trap_context("direct_leak", 0)
    tight = ctx.thresholds.model_copy(
        update={
            "sampling": ctx.thresholds.sampling.model_copy(
                update={"max_rows": 400, "max_features": 3}
            )
        }
    )
    small = dataclasses.replace(ctx, thresholds=tight)
    result = a1.measure(small)
    assert result.sampling.n_rows_used == 400
    assert result.sampling.row_method == "stratified"
    assert result.sampling.feature_method == "top_univariate"
    assert "refund_issued" in result.scores  # the leak survives the feature cap
    assert len(result.scores) == 3
