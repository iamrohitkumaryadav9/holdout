"""Run orchestration: load -> profile -> propose split -> seal -> baseline -> plan -> unseal.

Kept separate from the CLI so the same sequence can be driven by tests today and by an
agent loop later (an LLM proposing plans in place of ``default_plan``). The order of the
steps is the guarantee: the profile is target-blind, the plan is validated before the seal
so a rejected plan never leaves a half-finished run, and the holdout is opened at most once,
at the very end, through :meth:`SealedHoldout.evaluate`.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from holdout.core.errors import PlanValidationError, SplitError
from holdout.core.schemas import CVResult, DataType, HoldoutResult, SplitStrategy, Task
from holdout.core.state import RunState
from holdout.data.load import load_dataset
from holdout.data.profile import DatasetProfile, profile_schema
from holdout.data.seal import SealedHoldout
from holdout.data.split import SplitProposal, SplitSpec, propose_split
from holdout.export.manifest import (
    NOT_UNSEALED,
    DataRecord,
    PlanRecord,
    ProfileSummary,
    RunManifest,
    SplitRecord,
    capture_environment,
    write_manifest,
    write_report,
)
from holdout.export.pipeline_export import write_pipeline_script
from holdout.plan.compiler import evaluate_cv, fit_final
from holdout.plan.defaults import baseline_plan, default_plan
from holdout.plan.metrics import get_metric
from holdout.plan.schema import ModelPlan, PlanIssue, Severity, validate_plan

#: Caller name recorded in the holdout access log for CLI unsealing.
UNSEAL_CALLER = "holdout run --unseal"


@dataclass(frozen=True)
class RunOptions:
    csv: Path
    target: str
    task: Task | None = None
    data_type: DataType = DataType.TABULAR
    split: str = "auto"
    seed: int = 42
    plan_path: Path | None = None
    unseal: bool = False
    out_dir: Path = Path("runs")


@dataclass(frozen=True)
class RunOutcome:
    state: RunState
    manifest: RunManifest
    proposal: SplitProposal
    spec: SplitSpec
    plan_issues: list[PlanIssue]
    baseline: CVResult
    result: CVResult
    holdout: HoldoutResult | None


class RunObserver(Protocol):
    """Progress hooks, so the CLI can print the split reasoning before CV starts."""

    def on_stage(self, message: str) -> None: ...

    def on_split(self, proposal: SplitProposal, spec: SplitSpec) -> None: ...


class _SilentObserver:
    def on_stage(self, message: str) -> None:
        del message

    def on_split(self, proposal: SplitProposal, spec: SplitSpec) -> None:
        del proposal, spec


def parse_split_option(text: str, seed: int) -> SplitSpec | None:
    """Parse ``--split``: ``auto`` (None), ``random``, ``stratified``, ``group:COL``, ``time:COL``.

    Raises:
        SplitError: for an unknown strategy or a missing/unexpected column.
    """
    if text == "auto":
        return None
    name, separator, column = text.partition(":")
    try:
        strategy = SplitStrategy(name)
    except ValueError as exc:
        raise SplitError(
            f"unknown --split {text!r}; use auto, random, stratified, group:COL or time:COL"
        ) from exc
    takes_column = strategy in (SplitStrategy.GROUP, SplitStrategy.TIME)
    if takes_column and not column:
        raise SplitError(f"--split {name} needs a column, e.g. {name}:my_column")
    if not takes_column and separator:
        raise SplitError(f"--split {text!r}: only group and time splits take a column")
    return SplitSpec(
        strategy=strategy,
        seed=seed,
        group_col=column if strategy is SplitStrategy.GROUP else None,
        time_col=column if strategy is SplitStrategy.TIME else None,
    )


def load_plan(path: Path) -> ModelPlan:
    """Read a plan JSON file, turning schema errors into one readable message."""
    try:
        return ModelPlan.model_validate_json(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise PlanValidationError(f"cannot read plan {path}: {exc}", []) from exc
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']}" for e in exc.errors()
        )
        raise PlanValidationError(f"plan {path} is malformed: {details}", []) from exc


def _checked_plan(
    options: RunOptions, profile: DatasetProfile, task: Task
) -> tuple[ModelPlan, list[PlanIssue]]:
    plan = load_plan(options.plan_path) if options.plan_path else default_plan(profile, task)
    if plan.task is not task:
        raise PlanValidationError(
            f"the plan's task is {plan.task.value} but this run is {task.value}; "
            "pass --task to match the plan",
            [],
        )
    issues = validate_plan(plan, profile)
    errors = [i for i in issues if i.severity is Severity.ERROR]
    if errors:
        details = "; ".join(f"[{i.code}] {i.message}" for i in errors)
        raise PlanValidationError(f"plan rejected: {details}", list(errors))
    return plan, issues


def execute_run(options: RunOptions, observer: RunObserver | None = None) -> RunOutcome:
    """Run the full Week 1 flow and write all artifacts into ``out_dir/<run_id>/``."""
    watch = observer if observer is not None else _SilentObserver()
    started_at = datetime.now(UTC)
    watch.on_stage(f"loading {options.csv}")
    loaded = load_dataset(options.csv, options.target)
    frame = loaded.frame

    watch.on_stage("profiling schema (target-blind)")
    profile = profile_schema(frame, options.target, task=options.task)
    task = options.task if options.task is not None else profile.task_hint
    proposal = propose_split(profile, task, options.data_type, seed=options.seed)
    spec = parse_split_option(options.split, options.seed) or proposal.spec
    watch.on_split(proposal, spec)
    plan, issues = _checked_plan(options, profile, task)

    state = RunState.create(options.out_dir, options.seed, now=started_at)
    watch.on_stage(f"sealing the holdout in {state.run_dir}")
    seal, dev = SealedHoldout.seal(frame, options.target, spec, state, loaded.sha256)

    watch.on_stage("cross-validating the dummy baseline")
    baseline = evaluate_cv(
        baseline_plan(plan), dev, options.target, spec, profile, label="dummy baseline"
    )
    watch.on_stage(f"cross-validating the plan ({plan.model.name})")
    result = evaluate_cv(plan, dev, options.target, spec, profile, label=plan.model.name)

    holdout: HoldoutResult | None = None
    if options.unseal:
        watch.on_stage("fitting on all development rows and evaluating the holdout once")
        fitted = fit_final(plan, dev, options.target, spec, profile)
        holdout = seal.evaluate(fitted, get_metric(plan.metric), caller=UNSEAL_CALLER)

    write_pipeline_script(
        state.pipeline_path, plan=plan, profile=profile, spec=spec, data_sha256=loaded.sha256
    )
    artifacts = [
        p.name
        for p in (
            state.holdout_path,
            state.seal_state_path,
            state.access_log_path,
            state.pipeline_path,
            state.manifest_path,
            state.report_path,
        )
        if p.exists() or p in (state.manifest_path, state.report_path)
    ]
    metric = get_metric(plan.metric)
    manifest = RunManifest(
        run_id=state.run_id,
        started_at=started_at,
        finished_at=datetime.now(UTC),
        data=DataRecord(
            path=str(options.csv),
            sha256=loaded.sha256,
            n_rows=len(frame),
            n_cols=frame.shape[1],
            n_dropped_missing_target=loaded.n_dropped_missing_target,
        ),
        target=options.target,
        task=task,
        metric=metric.name,
        direction=metric.direction,
        profile=ProfileSummary.from_profile(profile),
        split=SplitRecord.build(
            spec,
            proposal,
            n_dev_rows=len(dev),
            n_holdout_rows=seal.n_rows,
            holdout_sha256=seal.holdout_sha256,
            reuse_warnings=seal.reuse_warnings,
        ),
        run_seed=options.seed,
        derived_seeds=state.seeds,
        plans=[
            PlanRecord(label=baseline.plan_label, plan=baseline_plan(plan), issues=[], cv=baseline),
            PlanRecord(label=result.plan_label, plan=plan, issues=issues, cv=result),
        ],
        selected_plan=result.plan_label,
        validation_score=result.mean,
        holdout=holdout if holdout is not None else NOT_UNSEALED,
        optimism_gap=None if holdout is None else result.mean - holdout.score,
        artifacts=artifacts,
        environment=capture_environment(),
    )
    write_manifest(state.manifest_path, manifest)
    write_report(state.report_path, manifest)
    return RunOutcome(
        state=state,
        manifest=manifest,
        proposal=proposal,
        spec=spec,
        plan_issues=issues,
        baseline=baseline,
        result=result,
        holdout=holdout,
    )
