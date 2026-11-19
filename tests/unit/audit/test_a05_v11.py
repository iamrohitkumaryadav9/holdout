"""v1.1 A5: an entity key must be key-like and have entity-constant features.

Post-benchmark fix from docs/error_analysis.md, section 3 (A5 treated telco's
``number_vmail_messages``, a 0-52 count, as an entity key on two clean tasks)."""

import numpy as np
import pandas as pd
import pytest

from holdout.audit.checks.a05_group import candidates
from holdout.audit.context import AuditContext
from holdout.audit.thresholds import load_thresholds, load_thresholds_v11
from holdout.core.schemas import SplitStrategy, Task
from holdout.data.profile import profile_schema
from holdout.data.split import SplitSpec
from holdout.plan.compiler import evaluate_cv
from holdout.plan.defaults import baseline_plan, default_plan

SPEC = SplitSpec(strategy=SplitStrategy.STRATIFIED, seed=3)


def _ctx(
    frame: pd.DataFrame, target: str, v11: bool, task: Task = Task.CLASSIFICATION
) -> AuditContext:
    profile = profile_schema(frame, target, task=task)
    plan = default_plan(profile, task)
    spec = SPEC if task is Task.CLASSIFICATION else SplitSpec(strategy=SplitStrategy.RANDOM, seed=3)
    dummy = evaluate_cv(baseline_plan(plan), frame, target, spec, profile)
    return AuditContext(dev=frame, target=target, profile=profile, spec=spec, plan=plan,
                        cv_result=dummy, baseline=dummy, thresholds=load_thresholds()[0],
                        thresholds_v11=load_thresholds_v11()[0] if v11 else None)  # fmt: skip


def _frame(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    accounts = rng.integers(0, 300, n)
    region = rng.integers(0, 5, 300)  # an attribute of the account: constant within it
    vmail = np.where(rng.random(n) < 0.7, 0, rng.integers(1, 52, n))
    return pd.DataFrame({
        "account": [f"ACC-{a:04d}" for a in accounts],  # a string entity key
        "customer_id": accounts.astype("int64"),  # the same entity, as an integer key
        "region": region[accounts],
        "number_vmail_messages": vmail.astype("int64"),  # a count: not an entity
        # A coarse attribute that happens to be constant within each count value.
        "vmail_plan": (vmail > 0).astype("int64"),
        "usage": rng.normal(size=n),
        "churn": rng.integers(0, 2, n),
    })  # fmt: skip


def test_a_count_column_is_no_longer_an_entity_key() -> None:
    frame = _frame()
    v10 = {c.column for c in candidates(_ctx(frame, "churn", v11=False))}
    v11 = {c.column for c in candidates(_ctx(frame, "churn", v11=True))}
    assert "number_vmail_messages" in v10  # the v1.0 false positive
    assert "number_vmail_messages" not in v11
    assert {"account", "customer_id"} <= v11  # real keys, string and key-named integer


def test_a_string_column_without_entity_constant_features_is_not_a_key() -> None:
    rng = np.random.default_rng(1)
    n = 2000
    frame = pd.DataFrame({
        "diagnosis": [f"D{c}" for c in rng.integers(0, 200, n)],  # repeats, nothing constant
        "age": rng.integers(18, 90, n),
        "usage": rng.normal(size=n),
        "readmitted": rng.integers(0, 2, n),
    })  # fmt: skip
    v10 = {c.column for c in candidates(_ctx(frame, "readmitted", v11=False))}
    v11 = {c.column for c in candidates(_ctx(frame, "readmitted", v11=True))}
    assert "diagnosis" in v10
    assert "diagnosis" not in v11


# --------------------------------------------------------------- the benchmark's frames


def _datasets_cached() -> bool:
    try:
        from bench.data.openml import cached_path
        from bench.data.registry import load_registry
    except ImportError:
        return False
    return all(cached_path(s).exists() for s in load_registry().values() if s.pool == "final")


def _benchmark_candidates(task_id: str, seed: int = 101) -> set[str]:
    from bench.harness import partition
    from bench.run import Unit, prepare

    p = prepare(Unit("final_v1", task_id, "audits_only", seed, "planted"))
    dev, _, _ = partition(p.frame, p.target, p.rule, seed)
    return {c.column for c in candidates(_ctx(dev, p.target, v11=True, task=p.task))}


@pytest.mark.full_only
@pytest.mark.skipif(not _datasets_cached(), reason="needs the OpenML files (no network in CI)")
def test_the_benchmark_count_column_is_dropped_and_the_planted_keys_remain() -> None:
    for clean in ("group_leakage:clean@telco_churn", "causal_question:clean@telco_churn"):
        assert "number_vmail_messages" not in _benchmark_candidates(clean), clean
    for trap in (
        "group_leakage@telco_churn",
        "group_leakage@house_16h",
        "group_leakage@qsar_biodeg",
    ):
        assert "account" in _benchmark_candidates(trap), trap
