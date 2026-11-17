"""Build a real AuditContext from a trap case, through the same path as a CLI run.

CSV round trip -> load -> target-blind profile -> split proposal -> seal -> dummy and plan
CV. Contexts are cached per (trap, seed, variant) because several checks reuse them.
"""

import tempfile
from functools import cache
from pathlib import Path
from typing import Any

from bench.traps import TrapCase, build_case, clean_twin
from holdout.audit.context import AuditContext
from holdout.audit.thresholds import load_thresholds
from holdout.core.schemas import DataType, EntityScope
from holdout.core.state import RunState
from holdout.data.load import load_dataset
from holdout.data.profile import profile_schema
from holdout.data.seal import SealedHoldout
from holdout.data.split import SplitSpec, propose_split
from holdout.plan.compiler import evaluate_cv
from holdout.plan.defaults import baseline_plan, default_plan
from holdout.plan.schema import ModelPlan

SEEDS = (0, 1, 2, 3, 4)

_WORKDIR = Path(tempfile.mkdtemp(prefix="holdout-audit-tests-"))


#: Per-check tests use the default plan with a lighter booster, to keep ~100 audit
#: contexts fast. End-to-end tests use the untouched default plan through the CLI.
FAST_MODEL = {"name": "hist_gb", "max_iter": 60}


def plan_for(case: TrapCase, profile: Any, *, fast: bool = True) -> ModelPlan:
    plan = default_plan(profile, case.task)
    overrides = dict(case.plan_overrides)
    if fast:
        overrides.setdefault("model", FAST_MODEL)
    if not overrides:
        return plan
    return ModelPlan.model_validate({**plan.model_dump(mode="json"), **overrides})


def context_for(
    case: TrapCase,
    seed: int,
    *,
    spec: SplitSpec | None = None,
    entity_scope: EntityScope | None = None,
    workdir: Path | None = None,
) -> AuditContext:
    root = workdir if workdir is not None else Path(tempfile.mkdtemp(dir=_WORKDIR))
    csv = root / "data.csv"
    case.df.to_csv(csv, index=False)
    loaded = load_dataset(csv, case.target)
    profile = profile_schema(loaded.frame, case.target, task=case.task)
    proposal = propose_split(
        profile, case.task, DataType.TABULAR, seed=seed, entity_scope=entity_scope
    )
    chosen = spec if spec is not None else proposal.spec
    state = RunState.create(root / "runs", seed)
    seal, dev = SealedHoldout.seal(loaded.frame, case.target, chosen, state, loaded.sha256)
    plan = plan_for(case, profile)
    baseline = evaluate_cv(baseline_plan(plan), dev, case.target, chosen, profile, label="dummy")
    result = evaluate_cv(plan, dev, case.target, chosen, profile, label=plan.model.name)
    return AuditContext(
        dev=dev,
        target=case.target,
        profile=profile,
        spec=chosen,
        plan=plan,
        cv_result=result,
        baseline=baseline,
        thresholds=load_thresholds()[0],
        overlap_source=seal,
        goal_text=case.goal_text,
    )


@cache
def trap_context(name: str, seed: int) -> AuditContext:
    return context_for(build_case(name, seed), seed)


@cache
def twin_context(name: str, seed: int) -> AuditContext:
    return context_for(clean_twin(build_case(name, seed)), seed)
