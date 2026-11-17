"""A7: accuracy on a 4% minority is critical with secondary metrics; regression tails."""

import dataclasses

import numpy as np
import pytest

from holdout.audit.checks import a07_metric as a7
from holdout.audit.types import Severity
from holdout.core.schemas import MetricName
from tests.audit_harness import trap_context, twin_context
from tests.tiers import SEEDS


@pytest.mark.parametrize("seed", SEEDS)
def test_accuracy_on_imbalanced_target_is_critical_with_secondary_metrics(seed: int) -> None:
    ctx = trap_context("imbalance_accuracy", seed)
    outcome = a7.run(ctx)
    flag = outcome.flags[0]
    assert flag.severity is Severity.CRITICAL
    assert flag.fix.kind == "change_metric"
    assert flag.fix.metric is MetricName.BALANCED_ACCURACY  # type: ignore[union-attr]
    evidence = flag.evidence
    assert evidence.plan_issue_code == "accuracy_on_imbalanced_target"  # type: ignore[union-attr]
    secondary = evidence.secondary_scores  # type: ignore[union-attr]
    dummy = evidence.secondary_dummy_scores  # type: ignore[union-attr]
    assert set(secondary) == {"balanced_accuracy", "pr_auc"}
    assert dummy["balanced_accuracy"] == pytest.approx(0.5)
    # Accuracy looks great; the balanced metrics tell the real story.
    assert ctx.full_score > 0.9
    assert secondary["balanced_accuracy"] < ctx.full_score
    assert "balanced_accuracy" in flag.message
    assert "pr_auc" in flag.message
    assert outcome.sampling is not None


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("trap", ["imbalance_accuracy", "direct_leak"])
def test_quiet_on_clean_twins(trap: str, seed: int) -> None:
    assert a7.run(twin_context(trap, seed)).flags == []


def test_heavy_tailed_regression_target_with_r2_warns() -> None:
    ctx = trap_context("time_order", 0)
    heavy = ctx.dev.copy()
    rng = np.random.default_rng(0)
    heavy[ctx.target] = rng.lognormal(0, 2.0, len(heavy))
    plan = ctx.plan.model_copy(update={"metric": MetricName.R2})
    flag = a7.run(dataclasses.replace(ctx, dev=heavy, plan=plan)).flags[0]
    assert flag.severity is Severity.WARNING
    assert flag.fix.metric is MetricName.MAE  # type: ignore[union-attr]
    assert flag.evidence.tail_ratio > 10  # type: ignore[union-attr,operator]
    rmse = dataclasses.replace(ctx, dev=heavy)  # the same target with RMSE: no R2 warning
    assert a7.run(rmse).flags == []


def test_tail_ratio() -> None:
    assert a7.tail_ratio(np.ones(10)) is None
    assert a7.tail_ratio(np.arange(100.0)) == pytest.approx(0.98, abs=0.01)
