"""A3: an identifier issued in target order is critical; random identifiers stay info."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from holdout.audit.checks import a01_single_feature as a1
from holdout.audit.checks import a03_id_like as a3
from holdout.audit.types import Severity
from holdout.plan.schema import DropColumn
from tests.audit_harness import trap_context, twin_context
from tests.tiers import SEEDS


@pytest.mark.parametrize("seed", SEEDS)
def test_planted_id_with_signal_is_critical(seed: int) -> None:
    ctx = trap_context("id_with_signal", seed)
    outcome, id_like = a3.run(ctx, a1.measure(ctx))
    assert "record_number" in id_like
    flag = next(f for f in outcome.flags if f.columns == ["record_number"])
    assert flag.severity is Severity.CRITICAL
    assert flag.evidence.kind == "id_like"
    assert {"near_unique", "integer_sequence"} <= set(flag.evidence.signals)  # type: ignore[union-attr]


@pytest.mark.parametrize("seed", SEEDS)
def test_random_identifier_on_clean_twin_is_info(seed: int) -> None:
    ctx = twin_context("id_with_signal", seed)
    outcome, id_like = a3.run(ctx, a1.measure(ctx))
    assert "record_number" in id_like
    assert all(f.severity is Severity.INFO for f in outcome.flags)


def test_dropped_identifier_with_signal_is_a_warning() -> None:
    ctx = trap_context("id_with_signal", 0)
    plan = ctx.plan.model_copy(
        update={
            "drop_columns": [
                *ctx.plan.drop_columns,
                DropColumn(column="record_number", reason="identifier"),
            ]
        }
    )
    dropped = dataclasses.replace(ctx, plan=plan)
    outcome, _ = a3.run(dropped, a1.measure(dropped))
    flag = next(f for f in outcome.flags if f.columns == ["record_number"])
    assert flag.severity is Severity.WARNING


def test_structure_signals() -> None:
    codes = pd.Series([f"AB-{i:04d}" for i in range(100)])
    assert a3._uniform_code(codes, 0.95)
    assert not a3._uniform_code(pd.Series(["x", "yy", "zzz"] * 30), 0.95)
    assert a3._integer_sequence(pd.Series(np.arange(10, 110)), 1.1)
    assert not a3._integer_sequence(pd.Series(np.arange(0, 1000, 10)), 1.1)
    assert not a3._integer_sequence(pd.Series(np.linspace(0, 1, 100)), 1.1)
    assert a3._row_order_spearman(pd.Series(np.arange(50))) == pytest.approx(1.0)
    assert a3._row_order_spearman(pd.Series(["a"] * 50)) is None
    assert a3._row_order_spearman(pd.Series([1.0] * 50)) is None


def test_a3_without_a1_reports_no_signal() -> None:
    ctx = trap_context("id_with_signal", 0)
    outcome, _ = a3.run(ctx, None)
    assert all(f.severity is Severity.INFO for f in outcome.flags)
