"""Run orchestration: load -> profile -> propose split -> seal -> baseline -> plan -> audit
-> (gated) unseal.

Kept separate from the CLI so the same sequence can be driven by tests today and by an
agent loop later (an LLM proposing plans in place of ``default_plan``). The order of the
steps is the guarantee: the profile is target-blind, the plan is validated before the seal
so a rejected plan never leaves a half-finished run, the audit reads development rows only,
and the holdout is opened at most once, at the very end, only if the unseal gate allows it.
"""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from pydantic import ValidationError

from holdout.agent.schemas import AgentRecord
from holdout.audit.gate import UnsealDecision, decide_unseal, validate_risk_reason
from holdout.audit.runner import run_audits
from holdout.audit.thresholds import load_thresholds, load_thresholds_v11
from holdout.audit.types import AuditReport
from holdout.core.errors import HoldoutError, PlanValidationError, SplitError
from holdout.core.schemas import (
    CVResult,
    DataType,
    EntityScope,
    HoldoutResult,
    SplitStrategy,
    Task,
)
from holdout.core.state import RunState
from holdout.data.load import LoadedData, load_dataset
from holdout.data.profile import DatasetProfile, profile_schema
from holdout.data.seal import SealedHoldout, dev_index_sha256
from holdout.data.split import SplitProposal, SplitSpec, apply_split, propose_split
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

#: Caller names recorded in the holdout access log.
UNSEAL_CALLER = "holdout run --unseal"
AUDIT_CALLER = "holdout audit"


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
    goal: str | None = None
    entities: EntityScope | None = None
    ack_warnings: bool = False
    accept_risk: str | None = None


@dataclass(frozen=True)
class RunOutcome:
    state: RunState
    manifest: RunManifest
    proposal: SplitProposal
    spec: SplitSpec
    plan_issues: list[PlanIssue]
    baseline: CVResult
    result: CVResult
    audit: AuditReport
    unseal: UnsealDecision
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
    """Run the full flow and write all artifacts into ``out_dir/<run_id>/``."""
    watch = observer if observer is not None else _SilentObserver()
    started_at = datetime.now(UTC)
    goal = options.goal if options.goal else f"predict {options.target}"
    try:
        validate_risk_reason(options.accept_risk)  # fail before any work is done
    except ValueError as exc:
        raise HoldoutError(str(exc)) from exc
    watch.on_stage(f"loading {options.csv}")
    loaded = load_dataset(options.csv, options.target)
    frame = loaded.frame

    watch.on_stage("profiling schema (target-blind)")
    profile = profile_schema(frame, options.target, task=options.task)
    task = options.task if options.task is not None else profile.task_hint
    proposal = propose_split(
        profile, task, options.data_type, seed=options.seed, entity_scope=options.entities
    )
    explicit = parse_split_option(options.split, options.seed)
    spec = (
        explicit.model_copy(update={"entity_scope": options.entities})
        if explicit is not None
        else proposal.spec
    )
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

    watch.on_stage("auditing the pipeline on development rows")
    thresholds = load_thresholds()
    thresholds_v11 = load_thresholds_v11()
    audit = run_audits(
        state,
        dev,
        plan,
        result,
        baseline=baseline,
        profile=profile,
        spec=spec,
        target=options.target,
        thresholds=thresholds,
        overlap_source=seal,
        goal_text=goal,
        thresholds_v11=thresholds_v11,
    )
    if audit.holdout_invalid:
        seal.mark_invalid(_invalid_reasons(audit), caller=AUDIT_CALLER)
    decision = decide_unseal(
        audit,
        requested=options.unseal,
        ack_warnings=options.ack_warnings,
        accept_risk=options.accept_risk,
    )

    holdout: HoldoutResult | None = None
    if decision.allowed:
        watch.on_stage("fitting on all development rows and evaluating the holdout once")
        fitted = fit_final(plan, dev, options.target, spec, profile)
        holdout = seal.evaluate(fitted, get_metric(plan.metric), caller=UNSEAL_CALLER)

    manifest = finalize_run(
        RunFacts(
            state=state,
            started_at=started_at,
            loaded=loaded,
            target=options.target,
            task=task,
            goal=goal,
            entity_scope=options.entities,
            profile=profile,
            proposal=proposal,
            spec=spec,
            n_dev_rows=len(dev),
            seal=seal,
            plan=plan,
            issues=issues,
            baseline=baseline,
            result=result,
            thresholds_hash=thresholds[1],
            thresholds_v11_hash=thresholds_v11[1],
            audit=audit,
            unseal=decision,
            holdout=holdout,
        )
    )
    return RunOutcome(
        state=state,
        manifest=manifest,
        proposal=proposal,
        spec=spec,
        plan_issues=issues,
        baseline=baseline,
        result=result,
        audit=audit,
        unseal=decision,
        holdout=holdout,
    )


@dataclass(frozen=True)
class RunFacts:
    """Everything a run's artifacts are written from (shared by holdout run and the agent)."""

    state: RunState
    started_at: datetime
    loaded: LoadedData
    target: str
    task: Task
    goal: str
    entity_scope: EntityScope | None
    profile: DatasetProfile
    proposal: SplitProposal
    spec: SplitSpec
    n_dev_rows: int
    seal: SealedHoldout
    plan: ModelPlan
    issues: list[PlanIssue]
    baseline: CVResult
    result: CVResult
    thresholds_hash: str
    audit: AuditReport
    unseal: UnsealDecision
    holdout: HoldoutResult | None
    agent: AgentRecord | None = None
    thresholds_v11_hash: str | None = None


def finalize_run(facts: RunFacts) -> RunManifest:
    """Write pipeline.py, run_manifest.json and report.md; return the manifest."""
    state = facts.state
    write_pipeline_script(
        state.pipeline_path,
        plan=facts.plan,
        profile=facts.profile,
        spec=facts.spec,
        data_sha256=facts.loaded.sha256,
    )
    artifacts = [
        p.name
        for p in (
            state.holdout_path,
            state.seal_state_path,
            state.holdout_digests_path,
            state.access_log_path,
            state.audit_path,
            state.checkpoint_path,
            state.pipeline_path,
            state.manifest_path,
            state.report_path,
        )
        if p.exists() or p in (state.manifest_path, state.report_path)
    ]
    metric = get_metric(facts.plan.metric)
    frame = facts.loaded.frame
    result, holdout = facts.result, facts.holdout
    manifest = RunManifest(
        run_id=state.run_id,
        started_at=facts.started_at,
        finished_at=datetime.now(UTC),
        data=DataRecord(
            path=str(facts.loaded.path),
            sha256=facts.loaded.sha256,
            n_rows=len(frame),
            n_cols=frame.shape[1],
            n_dropped_missing_target=facts.loaded.n_dropped_missing_target,
        ),
        target=facts.target,
        task=facts.task,
        goal_text=facts.goal,
        entity_scope=facts.entity_scope,
        metric=metric.name,
        direction=metric.direction,
        profile=ProfileSummary.from_profile(facts.profile),
        split=SplitRecord.build(
            facts.spec,
            facts.proposal,
            n_dev_rows=facts.n_dev_rows,
            n_holdout_rows=facts.seal.n_rows,
            holdout_sha256=facts.seal.holdout_sha256,
            reuse_warnings=facts.seal.reuse_warnings,
        ),
        run_seed=state.run_seed,
        derived_seeds=state.seeds,
        plans=[
            PlanRecord(
                label=facts.baseline.plan_label,
                plan=baseline_plan(facts.plan),
                issues=[],
                cv=facts.baseline,
            ),
            PlanRecord(label=result.plan_label, plan=facts.plan, issues=facts.issues, cv=result),
        ],
        selected_plan=result.plan_label,
        validation_score=result.mean,
        thresholds_hash=facts.thresholds_hash,
        thresholds_v11_hash=facts.thresholds_v11_hash,
        audit=facts.audit,
        unseal=facts.unseal,
        holdout_invalid=facts.audit.holdout_invalid,
        caveats=facts.seal.caveats,
        holdout=holdout if holdout is not None else NOT_UNSEALED,
        optimism_gap=None if holdout is None else result.mean - holdout.score,
        artifacts=artifacts,
        environment=capture_environment(),
        agent=facts.agent,
    )
    write_manifest(state.manifest_path, manifest)
    write_report(state.report_path, manifest)
    return manifest


def reaudit_run(run_dir: Path) -> AuditReport:
    """Re-run the audits of an existing run (never unseals).

    The development rows are rebuilt by re-applying the recorded split to the recorded
    CSV; both are verified (the CSV's sha256, and the development-row fingerprint recorded
    at seal time), so the audit sees exactly the rows the run used. The holdout file is
    never read. The latest audit replaces ``audit.json``; the manifest keeps the audit made
    at run time. A contaminated holdout is marked invalid in the seal.

    Raises:
        HoldoutError: if the run, its CSV or its seal cannot be matched.
    """
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise HoldoutError(f"no run manifest in {run_dir}")
    raw: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    if raw.get("status", "completed") != "completed":
        raise HoldoutError(
            f"run {raw.get('run_id')} is {raw.get('status')}, not completed; resume it with "
            f"{raw.get('resume_command', 'holdout agent --resume <run_id>')}"
        )
    target = str(raw["target"])
    loaded = load_dataset(Path(raw["data"]["path"]), target)
    if loaded.sha256 != raw["data"]["sha256"]:
        raise HoldoutError(
            f"{loaded.path} changed since the run (sha256 {loaded.sha256}, run used "
            f"{raw['data']['sha256']}); cannot re-audit"
        )
    spec = SplitSpec.model_validate(raw["split"]["spec"])
    plan = ModelPlan.model_validate(raw["plans"][-1]["plan"])
    baseline = CVResult.model_validate(raw["plans"][0]["cv"])
    result = CVResult.model_validate(raw["plans"][-1]["cv"])
    profile = profile_schema(loaded.frame, target, task=Task(raw["task"]))
    dev, _ = apply_split(loaded.frame, spec, target)
    state = RunState(
        run_id=str(raw["run_id"]),
        out_dir=run_dir.parent,
        run_seed=int(raw["run_seed"]),
        created_at=datetime.fromisoformat(raw["started_at"]),
    )
    seal = SealedHoldout.load(state)
    recorded = seal.record.dev_index_sha256
    if recorded is not None and recorded != dev_index_sha256(dev):
        raise HoldoutError("re-applied split does not reproduce the run's development rows")
    report = run_audits(
        state,
        dev,
        plan,
        result,
        baseline=baseline,
        profile=profile,
        spec=spec,
        target=target,
        overlap_source=seal,
        goal_text=str(raw.get("goal_text") or f"predict {target}"),
    )
    if report.holdout_invalid and not seal.holdout_invalid:
        seal.mark_invalid(_invalid_reasons(report), caller=AUDIT_CALLER)
    return report


def _invalid_reasons(report: AuditReport) -> list[str]:
    return [f"{f.check.value}: {f.message}" for f in report.flags if f.invalidates_holdout]
