"""v1.1 fixes found on the app's group_leakage demo follow-up run (group split on 'account').

1. A plan that does not beat the dummy is a warning, said plainly (it does not block).
2. One entity-key rule for the split proposer and A5; the run's own group key is recorded.
3. The group key is not a feature when predictions are for new entities.

Post-benchmark (v1.1 code paths and thresholds_v11.toml only)."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from holdout.agent.graph import drop_entity_key
from holdout.audit.checks import a05_group, a12_stability
from holdout.audit.checks.a12_stability import (
    BELOW_BASELINE,
    NOT_RELIABLY_ABOVE_BASELINE,
    baseline_statement,
)
from holdout.audit.context import AuditContext
from holdout.audit.thresholds import load_thresholds, load_thresholds_v11
from holdout.audit.types import CheckId, Severity
from holdout.core.schemas import CVResult, Direction, EntityScope, MetricName, SplitStrategy, Task
from holdout.data.profile import profile_schema
from holdout.data.split import SplitSpec
from holdout.data.structure import entity_keys
from holdout.plan.compiler import evaluate_cv
from holdout.plan.defaults import baseline_plan, default_plan

V11 = load_thresholds_v11()[0]


def _entity_frame(n_entities: int = 100, rows: int = 6, seed: int = 0) -> pd.DataFrame:
    """The demo's shape: a string key with entity-constant attributes, a count that is not
    an entity, and a target that depends on the entity only (nothing a new entity has)."""
    rng = np.random.default_rng(seed)
    ent = np.repeat(np.arange(n_entities), rows)
    return pd.DataFrame({
        "account": [f"A{e:03d}" for e in ent],
        "region_code": rng.integers(0, 6, n_entities)[ent],
        "account_vintage": rng.integers(2000, 2024, n_entities)[ent],
        "visits": rng.integers(0, 30, len(ent)),  # a count: not an entity
        "noise": rng.normal(size=len(ent)),
        "converted": rng.integers(0, 2, n_entities)[ent],
    })  # fmt: skip


def _ctx(frame: pd.DataFrame, spec: SplitSpec, *, v11: bool = True, plan=None) -> AuditContext:  # type: ignore[no-untyped-def]
    profile = profile_schema(frame, "converted", task=Task.CLASSIFICATION)
    plan = plan or default_plan(profile, Task.CLASSIFICATION)
    cv = evaluate_cv(plan, frame, "converted", spec, profile)
    dummy = evaluate_cv(baseline_plan(plan), frame, "converted", spec, profile)
    return AuditContext(dev=frame, target="converted", profile=profile, spec=spec, plan=plan,
                        cv_result=cv, baseline=dummy, thresholds=load_thresholds()[0],
                        thresholds_v11=V11 if v11 else None)  # fmt: skip


# --------------------------------------------------------------------- 1. below baseline


def _cv(fold_scores: list[float], label: str = "plan") -> CVResult:
    return CVResult(plan_label=label, metric=MetricName.ROC_AUC, direction=Direction.HIGHER,
                    mean=float(np.mean(fold_scores)), std=float(np.std(fold_scores)),
                    fold_scores=fold_scores, fit_seconds=0.0, n_rows=600, n_folds=5,
                    splitter="GroupKFold")  # fmt: skip


def _a12(ctx: AuditContext, plan_folds: list[float]):  # type: ignore[no-untyped-def]
    ctx = dataclasses.replace(ctx, cv_result=_cv(plan_folds), baseline=_cv([0.5] * 5, "dummy"))
    return a12_stability.run(ctx)


def test_a_plan_below_the_dummy_is_a_warning_in_v11() -> None:
    frame = _entity_frame()
    spec = SplitSpec(strategy=SplitStrategy.GROUP, group_col="account", seed=1,
                     entity_scope=EntityScope.NEW)  # fmt: skip
    folds = [0.38, 0.44, 0.47, 0.41, 0.45]  # the demo: 0.429 vs 0.500
    (flag,) = _a12(_ctx(frame, spec), folds).flags
    assert flag.severity is Severity.WARNING
    assert flag.message.startswith(BELOW_BASELINE)
    assert "does not exclude zero" in flag.message
    assert baseline_statement([flag]) == BELOW_BASELINE
    # v1.0 behaviour is kept without the v1.1 layer.
    v10 = _a12(_ctx(frame, spec, v11=False), folds)
    assert v10.flags == []
    assert v10.detail == "plan does not beat the dummy; stability not assessed"


def test_an_unreliable_lift_gets_the_softer_sentence() -> None:
    frame = _entity_frame()
    spec = SplitSpec(strategy=SplitStrategy.STRATIFIED, seed=1)
    (flag,) = _a12(_ctx(frame, spec), [0.62, 0.45, 0.58, 0.47, 0.55]).flags
    assert flag.severity is Severity.WARNING
    assert baseline_statement([flag]) == NOT_RELIABLY_ABOVE_BASELINE
    (good,) = _a12(_ctx(frame, spec), [0.80, 0.82, 0.79, 0.81, 0.83]).flags
    assert good.severity is Severity.INFO
    assert baseline_statement([good]) is None


# --------------------------------------------------------------------- 2. one entity rule


def test_the_proposer_and_a5_use_the_same_entity_rule() -> None:
    frame = _entity_frame()
    rule = V11.entity_rule()
    profile = profile_schema(frame, "converted", task=Task.CLASSIFICATION, entity_rule=rule)
    t = load_thresholds()[0].a05_group
    a5 = entity_keys(frame, ["converted"], rule, min_mean_rows_per_value=t.min_mean_rows_per_value,
                     min_groups=t.min_groups, max_unique_ratio=t.max_unique_ratio,
                     max_candidates=10)  # fmt: skip
    assert [e.column for e in profile.structural_entities] == [c.column for c in a5] == ["account"]
    # v1.0 profile (no rule) still asks about the count column too.
    v10 = profile_schema(frame, "converted", task=Task.CLASSIFICATION)
    assert "account" in {e.column for e in v10.structural_entities}


def test_the_runs_own_group_key_is_recorded_not_dropped_silently() -> None:
    frame = _entity_frame()
    spec = SplitSpec(strategy=SplitStrategy.GROUP, group_col="account", seed=1,
                     entity_scope=EntityScope.NEW)  # fmt: skip
    outcome = a05_group.run(_ctx(frame, spec))
    (flag,) = [f for f in outcome.flags if f.check is CheckId.A5]
    assert flag.severity is Severity.INFO
    assert flag.columns == ["account"]
    assert "the split groups by it" in flag.message
    assert outcome.detail == "split groups by entity key 'account'; no other entity-like columns"
    assert a05_group.run(_ctx(frame, spec, v11=False)).detail == "no entity-like columns"


# --------------------------------------------------------------------- 3. key not a feature


@pytest.mark.parametrize(
    ("scope", "dropped"),
    [(EntityScope.NEW, True), (None, True), (EntityScope.KNOWN, False)],
)
def test_the_group_key_is_dropped_for_new_entities(
    scope: EntityScope | None, dropped: bool
) -> None:
    frame = _entity_frame()
    profile = profile_schema(frame, "converted", task=Task.CLASSIFICATION)
    plan = default_plan(profile, Task.CLASSIFICATION)
    assert "account" not in plan.dropped
    spec = SplitSpec(strategy=SplitStrategy.GROUP, group_col="account", seed=1, entity_scope=scope)
    assert ("account" in drop_entity_key(plan, spec).dropped) is dropped
    row_split = SplitSpec(strategy=SplitStrategy.STRATIFIED, seed=1)
    assert drop_entity_key(plan, row_split) == plan
